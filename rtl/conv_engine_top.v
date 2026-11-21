`timescale 1ns/1ps

// ==========================================
// Module: Skew Buffer (平坦輸入 -> 階梯狀)
// 通用型：可用於橫向的 Activation，也可用於縱向的 Psum
// ==========================================
module skew_buffer #(
    parameter ROWS = 16,
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [ROWS*WIDTH-1:0] flat_in,
    output wire [ROWS*WIDTH-1:0] skewed_out
);
    genvar r;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_skew
            if (r == 0) begin : gen_row_0
                assign skewed_out[WIDTH-1:0] = flat_in[WIDTH-1:0];
            end else begin : gen_row_delay  // [修復] Synthesis 必須命名區塊
                reg [WIDTH-1:0] shift_reg [0:r-1];
                integer i; 
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for(i = 0; i < r; i = i + 1) shift_reg[i] <= 0; 
                    end else begin
                        shift_reg[0] <= flat_in[r*WIDTH +: WIDTH];
                        for(i = 1; i < r; i = i + 1) shift_reg[i] <= shift_reg[i-1]; 
                    end
                end
                assign skewed_out[r*WIDTH +: WIDTH] = shift_reg[r-1];
            end
        end
    endgenerate
endmodule

// ==========================================
// Module: De-skew Buffer (階梯狀 -> 平坦輸出)
// ==========================================
module deskew_buffer #(
    parameter COLS = 16,
    parameter WIDTH = 48
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [COLS*WIDTH-1:0] skewed_in,
    output wire [COLS*WIDTH-1:0] flat_out
);
    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : col_deskew
            localparam DELAY = COLS - 1 - c;
            
            if (DELAY == 0) begin : gen_col_0
                assign flat_out[c*WIDTH +: WIDTH] = skewed_in[c*WIDTH +: WIDTH];
            end else begin : gen_col_delay  // [修復] Synthesis 必須命名區塊
                reg [WIDTH-1:0] shift_reg [0:DELAY-1];
                integer i; 
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for(i = 0; i < DELAY; i = i + 1) shift_reg[i] <= 0; 
                    end else begin
                        shift_reg[0] <= skewed_in[c*WIDTH +: WIDTH];
                        for(i = 1; i < DELAY; i = i + 1) shift_reg[i] <= shift_reg[i-1]; 
                    end
                end
                assign flat_out[c*WIDTH +: WIDTH] = shift_reg[DELAY-1];
            end
        end
    endgenerate
endmodule

// ==========================================
// Module: Conv Engine Top (無腦平坦介面 + Tiling 支援)
// ==========================================
module conv_engine_top #(
    parameter ROWS = 16,
    parameter COLS = 16
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire load_wt,
    input  wire act_valid,  
    
    input  wire [ROWS*16-1:0] flat_act_in, 
    input  wire [COLS*16-1:0] flat_wt_in,
    
    // [新增] 支援 Tiling 的外部 Psum 輸入 (平坦輸入)
    input  wire [COLS*48-1:0] flat_psum_in, 
    
    output wire [COLS*48-1:0] flat_psum_out,
    output wire               out_valid   
);

    wire [ROWS*16-1:0] skewed_act;
    wire [COLS*48-1:0] skewed_psum_in;
    wire [COLS*48-1:0] skewed_psum_out;

    // 1. Skew Buffer (負責把橫向的 Activation 切成階梯狀)
    skew_buffer #(.ROWS(ROWS), .WIDTH(16)) u_skew_act (
        .clk(clk), .rst_n(rst_n),
        .flat_in(flat_act_in), .skewed_out(skewed_act)
    );

    // 1.5 Skew Buffer (負責把外部灌進來的 Psum 切成階梯狀，精準對齊管線)
    // 巧妙利用 skew_buffer，把 ROWS 設成 COLS 即可完美轉換
    skew_buffer #(.ROWS(COLS), .WIDTH(48)) u_skew_psum_in (
        .clk(clk), .rst_n(rst_n),
        .flat_in(flat_psum_in), .skewed_out(skewed_psum_in)
    );

    // 2. 脈動陣列核心 (接上剛才更新好的支援 Psum_in 版本)
    systolic_array #(.ROWS(ROWS), .COLS(COLS)) u_mac_array (
        .clk(clk), .rst_n(rst_n), .load_wt(load_wt),
        .act_in_row(skewed_act),
        .wt_in_col(flat_wt_in),
        .psum_in_col(skewed_psum_in),     // [新增連線] 接上階梯狀的 Psum
        .psum_out_col(skewed_psum_out)
    );

    // 3. De-skew Buffer (負責把陣列底部的階梯狀結果推平成平坦輸出)
    deskew_buffer #(.COLS(COLS), .WIDTH(48)) u_deskew (
        .clk(clk), .rst_n(rst_n),
        .skewed_in(skewed_psum_out), .flat_out(flat_psum_out)
    );

    // 4. 管線延遲追蹤器 (Data Path Delay Tracking)
    localparam TOTAL_DELAY = ROWS + COLS - 1;
    reg [TOTAL_DELAY-1:0] valid_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
        end else begin
            valid_pipe <= {valid_pipe[TOTAL_DELAY-2:0], act_valid};
        end
    end

    assign out_valid = valid_pipe[TOTAL_DELAY-1];

endmodule