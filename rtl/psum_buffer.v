`timescale 1ns/1ps

module psum_buffer #(
    parameter COLS = 16,
    parameter N_TOKENS = 16,     // [保持 16]
    parameter CHUNKS = 8,      
    parameter FRAC_BITS = 11,    // [🔥 對應 W_frac: 11] 用於 MAC 移位
    parameter A_FRAC_BITS = 5,   // [🔥 對應 A_frac: 5]  用於 ReLU6 極限
    parameter USE_RELU6 = 0      
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear_psum,    // [🔥 新增] 來自狀態機的強制清空訊號
    input  wire                 in_valid,
    input  wire [COLS*48-1:0]   flat_psum_in,
    input  wire [COLS*16-1:0]   flat_bias_in,
    output wire                 out_valid,
    output wire [COLS*16-1:0]   flat_out_data
);
    reg [COLS*48-1:0] ram [0:N_TOKENS-1];
    reg [COLS*48-1:0] ram_q_sync, psum_in_d1;
    reg in_valid_d1;
    reg [15:0] rd_addr, rd_addr_d1, chunk_cnt;

    // [🔥 修正] rd_addr 遇到 clear_psum 強制歸零
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_addr <= 0;
        else if (clear_psum) rd_addr <= 0;
        else if (in_valid) rd_addr <= (rd_addr == N_TOKENS-1) ? 0 : rd_addr + 1;
    end

    always @(posedge clk) begin
        ram_q_sync  <= ram[rd_addr];
        psum_in_d1  <= flat_psum_in;
        in_valid_d1 <= in_valid;
        rd_addr_d1  <= rd_addr;
    end

    // [🔥 修正] chunk_cnt 遇到 clear_psum 強制歸零，防止上一輪殘骸污染
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) chunk_cnt <= 0;
        else if (clear_psum) chunk_cnt <= 0;
        else if (in_valid_d1 && rd_addr_d1 == N_TOKENS-1) begin
            chunk_cnt <= (chunk_cnt == CHUNKS-1) ? 0 : chunk_cnt + 1;
        end
    end

    wire [COLS*48-1:0] sum_out;
    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : adder
            wire signed [47:0] in_val  = psum_in_d1[c*48 +: 48];
            // chunk_cnt 歸零時，強制加 0 (這就是我們清除累積值的精髓)
            wire signed [47:0] ram_val = (chunk_cnt == 0) ? 48'd0 : ram_q_sync[c*48 +: 48];
            assign sum_out[c*48 +: 48] = in_val + ram_val;
        end
    endgenerate

    always @(posedge clk) begin
        if (in_valid_d1) ram[rd_addr_d1] <= sum_out;
    end

    /*generate
        for (c = 0; c < COLS; c = c + 1) begin : clip
            wire signed [47:0] val = sum_out[c*48 +: 48];
            // [🔥 移位吃 W_frac (11)]
            wire signed [47:0] shifted = val >>> FRAC_BITS; 
            wire signed [15:0] bias_val = flat_bias_in[c*16 +: 16];
            wire signed [47:0] final_sum = shifted + {{32{bias_val[15]}}, bias_val};
            
            if (USE_RELU6 == 1) begin
                // [🔥 ReLU6 極限吃 A_frac (5)，數值自動變成 192]
                localparam signed [15:0] RELU6_MAX = 16'd6 << A_FRAC_BITS; 
                assign flat_out_data[c*16 +: 16] = (final_sum < 0) ? 16'd0 :
                                                   (final_sum > RELU6_MAX) ? RELU6_MAX : final_sum[15:0];
            end else begin
                assign flat_out_data[c*16 +: 16] = (final_sum > 32767)  ?  16'd32767 :
                                                   (final_sum < -32768) ? -16'd32768 : final_sum[15:0];
            end
        end
    endgenerate*/

    //assign out_valid = in_valid_d1 && (chunk_cnt == CHUNKS-1);
    // =================================================================
    // 🚀 正確答案：保持你的 Bias 順序，僅補上 100% 對齊軟體的 Rounding 電路
    // =================================================================
    generate
        for (c = 0; c < COLS; c = c + 1) begin : clip
            wire signed [47:0] val = sum_out[c*48 +: 48];
            
            // 💡 補上四捨五入加法器：加上 2^(11-1) = 1024
            wire signed [47:0] rounding_adder = (FRAC_BITS > 0) ? (48'sh1 << (FRAC_BITS - 1)) : 48'sh0;
            
            // 先加 1024 再右移 11 bits，此時正負數的四捨五入臨界點與 PyTorch 完全 100% 同步
            wire signed [47:0] shifted = (val + rounding_adder) >>> FRAC_BITS; 
            
            // 完全保留你原本的 Bias 加法通路
            wire signed [15:0] bias_val = flat_bias_in[c*16 +: 16];
            wire signed [47:0] final_sum = shifted + {{32{bias_val[15]}}, bias_val};
            
            if (USE_RELU6 == 1) begin
                localparam signed [15:0] RELU6_MAX = 16'd6 << A_FRAC_BITS; 
                assign flat_out_data[c*16 +: 16] = (final_sum < 0) ? 16'd0 :
                                                   (final_sum > RELU6_MAX) ? RELU6_MAX : final_sum[15:0];
            end else begin
                assign flat_out_data[c*16 +: 16] = (final_sum > 32767)  ?  16'd32767 :
                                                   (final_sum < -32768) ? -16'd32768 : final_sum[15:0];
            end
        end
    endgenerate

    assign out_valid = in_valid_d1 && (chunk_cnt == CHUNKS-1);


endmodule