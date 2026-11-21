`timescale 1ns/1ps

module pe (
    input  wire clk,
    input  wire rst_n,
    input  wire load_wt,  
    
    input  wire signed [15:0] act_in,   
    input  wire signed [15:0] wt_in,    
    input  wire signed [47:0] psum_in,  
    
    output reg  signed [15:0] act_out,  
    output wire signed [15:0] wt_out,   // [FIX] 改成 wire
    output reg  signed [47:0] psum_out  
);
    reg signed [15:0] wt_reg;
    
    // [FIX] 直接把內部的 wt_reg 牽線給下一顆 PE，達成 1 Cycle 傳遞
    assign wt_out = wt_reg;
    
    wire signed [47:0] mult_res = $signed(act_in) * $signed(wt_reg);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_out  <= 0;
            psum_out <= 0;
            wt_reg   <= 0;
        end else if (load_wt) begin
            wt_reg   <= wt_in;
        end else begin
            act_out  <= act_in;
            psum_out <= psum_in + mult_res;
        end
    end
endmodule
module systolic_array #(
    parameter ROWS = 16,
    parameter COLS = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire load_wt,
    
    input  wire [ROWS*16-1:0] act_in_row,   
    input  wire [COLS*16-1:0] wt_in_col,    
    input  wire [COLS*48-1:0] psum_in_col,  // [新增] 為了支援 Tiling 累加而開的輸入
    output wire [COLS*48-1:0] psum_out_col  
);
    wire signed [15:0] act  [0:ROWS][0:COLS];
    wire signed [15:0] wt   [0:ROWS][0:COLS];
    wire signed [47:0] psum [0:ROWS][0:COLS]; 

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : bound_row
            assign act[r][0] = act_in_row[r*16 +: 16];
        end
        for (c = 0; c < COLS; c = c + 1) begin : bound_col
            assign wt[0][c]   = wt_in_col[c*16 +: 16];  
            
            // [關鍵修正] 不再寫死 0，而是接上外部輸入。
            // 算第一塊時，外部給 0；算第二塊時，外部把上一次的輸出餵回來！
            assign psum[0][c] = psum_in_col[c*48 +: 48]; 
            
            assign psum_out_col[c*48 +: 48] = psum[ROWS][c];
        end

        for (r = 0; r < ROWS; r = r + 1) begin : row
            for (c = 0; c < COLS; c = c + 1) begin : col
                pe u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_wt(load_wt),
                    .act_in(act[r][c]),
                    .wt_in(wt[r][c]),
                    .psum_in(psum[r][c]),
                    .act_out(act[r][c+1]),
                    .wt_out(wt[r+1][c]),
                    .psum_out(psum[r+1][c])
                );
            end
        end
    endgenerate
endmodule