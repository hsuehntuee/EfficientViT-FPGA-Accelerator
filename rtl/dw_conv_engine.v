/*`timescale 1ns/1ps

module dw_conv_engine #(
    parameter CHANNELS = 128,  // default only; accelerator_top instantiates it with CHANNELS=16
    parameter FRAC_BITS = 6   // Updated to 12 for precision alignment
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire                 load_wt,
    // Dynamically scaled based on CHANNELS
    input  wire [CHANNELS*16-1:0]    flat_wt_in,
    input  wire [CHANNELS*16-1:0]    flat_bias_in,
    
    input  wire                 in_valid,
    // Window size: 25 pixels * CHANNELS * 16 bits
    input  wire [25*CHANNELS*16-1:0] window_data, 
    
    output reg                  out_valid,
    output reg  [CHANNELS*16-1:0]    flat_out_data
);

    // Local parameter to calculate bits per pixel automatically
    localparam PIX_W = CHANNELS * 16;

    integer w_idx, ch_w, out_ch;

    // Weight and Bias Registers
    reg signed [15:0] w_reg [0:24][0:CHANNELS-1];
    reg signed [15:0] b_reg [0:CHANNELS-1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(w_idx = 0; w_idx < 25; w_idx = w_idx + 1)
                for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1)
                    w_reg[w_idx][ch_w] <= 0;
            for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1) 
                b_reg[ch_w] <= 0;
        end else if (load_wt) begin
            for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1) begin
                w_reg[0][ch_w] <= flat_wt_in[ch_w*16 +: 16];
                for(w_idx = 1; w_idx < 25; w_idx = w_idx + 1) begin
                    w_reg[w_idx][ch_w] <= w_reg[w_idx-1][ch_w];
                end
                b_reg[ch_w] <= flat_bias_in[ch_w*16 +: 16]; 
            end
        end
    end

    wire signed [15:0] clip_wire [0:CHANNELS-1];
    
    genvar c;
    generate
        for(c = 0; c < CHANNELS; c = c + 1) begin : mac_tree
            // PIPELINE STAGE 1: Break the 25-input adder tree into 5 smaller chunks
            reg signed [47:0] p1_sum [0:4];
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    p1_sum[0] <= 0; p1_sum[1] <= 0; p1_sum[2] <= 0; p1_sum[3] <= 0; p1_sum[4] <= 0;
                end else if (in_valid) begin
                    // ALL hardcoded 256 replaced with PIX_W to support any CHANNELS
                    p1_sum[0] <= $signed(window_data[ 0*PIX_W + c*16 +: 16]) * $signed(w_reg[ 0][c]) +
                                 $signed(window_data[ 1*PIX_W + c*16 +: 16]) * $signed(w_reg[ 1][c]) +
                                 $signed(window_data[ 2*PIX_W + c*16 +: 16]) * $signed(w_reg[ 2][c]) +
                                 $signed(window_data[ 3*PIX_W + c*16 +: 16]) * $signed(w_reg[ 3][c]) +
                                 $signed(window_data[ 4*PIX_W + c*16 +: 16]) * $signed(w_reg[ 4][c]);

                    p1_sum[1] <= $signed(window_data[ 5*PIX_W + c*16 +: 16]) * $signed(w_reg[ 5][c]) +
                                 $signed(window_data[ 6*PIX_W + c*16 +: 16]) * $signed(w_reg[ 6][c]) +
                                 $signed(window_data[ 7*PIX_W + c*16 +: 16]) * $signed(w_reg[ 7][c]) +
                                 $signed(window_data[ 8*PIX_W + c*16 +: 16]) * $signed(w_reg[ 8][c]) +
                                 $signed(window_data[ 9*PIX_W + c*16 +: 16]) * $signed(w_reg[ 9][c]);

                    p1_sum[2] <= $signed(window_data[10*PIX_W + c*16 +: 16]) * $signed(w_reg[10][c]) +
                                 $signed(window_data[11*PIX_W + c*16 +: 16]) * $signed(w_reg[11][c]) +
                                 $signed(window_data[12*PIX_W + c*16 +: 16]) * $signed(w_reg[12][c]) +
                                 $signed(window_data[13*PIX_W + c*16 +: 16]) * $signed(w_reg[13][c]) +
                                 $signed(window_data[14*PIX_W + c*16 +: 16]) * $signed(w_reg[14][c]);

                    p1_sum[3] <= $signed(window_data[15*PIX_W + c*16 +: 16]) * $signed(w_reg[15][c]) +
                                 $signed(window_data[16*PIX_W + c*16 +: 16]) * $signed(w_reg[16][c]) +
                                 $signed(window_data[17*PIX_W + c*16 +: 16]) * $signed(w_reg[17][c]) +
                                 $signed(window_data[18*PIX_W + c*16 +: 16]) * $signed(w_reg[18][c]) +
                                 $signed(window_data[19*PIX_W + c*16 +: 16]) * $signed(w_reg[19][c]);

                    p1_sum[4] <= $signed(window_data[20*PIX_W + c*16 +: 16]) * $signed(w_reg[20][c]) +
                                 $signed(window_data[21*PIX_W + c*16 +: 16]) * $signed(w_reg[21][c]) +
                                 $signed(window_data[22*PIX_W + c*16 +: 16]) * $signed(w_reg[22][c]) +
                                 $signed(window_data[23*PIX_W + c*16 +: 16]) * $signed(w_reg[23][c]) +
                                 $signed(window_data[24*PIX_W + c*16 +: 16]) * $signed(w_reg[24][c]);
                end
            end

            // PIPELINE STAGE 2: Final addition, shift, and clipping
            wire signed [47:0] total_sum = p1_sum[0] + p1_sum[1] + p1_sum[2] + p1_sum[3] + p1_sum[4];
            wire signed [47:0] shifted = total_sum >>> FRAC_BITS;
            wire signed [47:0] b_extended = {{32{b_reg[c][15]}}, b_reg[c]};
            wire signed [47:0] final_val = shifted + b_extended;
            
            assign clip_wire[c] = (final_val > 32767)  ?  16'd32767 :
                                  (final_val < -32768) ? -16'd32768 : final_val[15:0];
        end
    endgenerate

    // Delay valid signal by 1 clock cycle to match the pipeline stage
    reg in_valid_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_valid_d1 <= 0;
        else in_valid_d1 <= in_valid;
    end

    // Final Output Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 0;
            flat_out_data <= 0;
        end else begin
            out_valid <= in_valid_d1;
            if (in_valid_d1) begin
                for (out_ch = 0; out_ch < CHANNELS; out_ch = out_ch + 1) begin
                    flat_out_data[out_ch*16 +: 16] <= clip_wire[out_ch];
                end
            end
        end
    end

endmodule*/
`timescale 1ns/1ps

module dw_conv_engine #(
    parameter CHANNELS = 128,  // default only; accelerator_top instantiates it with CHANNELS=16
    parameter FRAC_BITS = 6    // Updated to 12 for precision alignment
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire                  load_wt,
    // Dynamically scaled based on CHANNELS
    input  wire [CHANNELS*16-1:0]    flat_wt_in,
    input  wire [CHANNELS*16-1:0]    flat_bias_in,
    
    input  wire                  in_valid,
    // Window size: 25 pixels * CHANNELS * 16 bits
    input  wire [25*CHANNELS*16-1:0] window_data, 
    
    output reg                   out_valid,
    output reg  [CHANNELS*16-1:0]    flat_out_data
);

    // Local parameter to calculate bits per pixel automatically
    localparam PIX_W = CHANNELS * 16;

    integer w_idx, ch_w, out_ch;

    // Weight and Bias Registers
    reg signed [15:0] w_reg [0:24][0:CHANNELS-1];
    reg signed [15:0] b_reg [0:CHANNELS-1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(w_idx = 0; w_idx < 25; w_idx = w_idx + 1)
                for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1)
                    w_reg[w_idx][ch_w] <= 0;
            for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1) 
                b_reg[ch_w] <= 0;
        end else if (load_wt) begin
            for(ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1) begin
                w_reg[0][ch_w] <= flat_wt_in[ch_w*16 +: 16];
                for(w_idx = 1; w_idx < 25; w_idx = w_idx + 1) begin
                    w_reg[w_idx][ch_w] <= w_reg[w_idx-1][ch_w];
                end
                b_reg[ch_w] <= flat_bias_in[ch_w*16 +: 16]; 
            end
        end
    end

    wire signed [15:0] clip_wire [0:CHANNELS-1];
    
    genvar c;
    generate
        for(c = 0; c < CHANNELS; c = c + 1) begin : mac_tree
            // PIPELINE STAGE 1: Break the 25-input adder tree into 5 smaller chunks
            reg signed [47:0] p1_sum [0:4];
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    p1_sum[0] <= 0; p1_sum[1] <= 0; p1_sum[2] <= 0; p1_sum[3] <= 0; p1_sum[4] <= 0;
                end else if (in_valid) begin
                    // ALL hardcoded 256 replaced with PIX_W to support any CHANNELS
                    p1_sum[0] <= $signed(window_data[ 0*PIX_W + c*16 +: 16]) * $signed(w_reg[24][c]) +
                                 $signed(window_data[ 1*PIX_W + c*16 +: 16]) * $signed(w_reg[23][c]) +
                                 $signed(window_data[ 2*PIX_W + c*16 +: 16]) * $signed(w_reg[22][c]) +
                                 $signed(window_data[ 3*PIX_W + c*16 +: 16]) * $signed(w_reg[21][c]) +
                                 $signed(window_data[ 4*PIX_W + c*16 +: 16]) * $signed(w_reg[20][c]);

                    p1_sum[1] <= $signed(window_data[ 5*PIX_W + c*16 +: 16]) * $signed(w_reg[19][c]) +
                                 $signed(window_data[ 6*PIX_W + c*16 +: 16]) * $signed(w_reg[18][c]) +
                                 $signed(window_data[ 7*PIX_W + c*16 +: 16]) * $signed(w_reg[17][c]) +
                                 $signed(window_data[ 8*PIX_W + c*16 +: 16]) * $signed(w_reg[16][c]) +
                                 $signed(window_data[ 9*PIX_W + c*16 +: 16]) * $signed(w_reg[15][c]);

                    p1_sum[2] <= $signed(window_data[10*PIX_W + c*16 +: 16]) * $signed(w_reg[14][c]) +
                                 $signed(window_data[11*PIX_W + c*16 +: 16]) * $signed(w_reg[13][c]) +
                                 $signed(window_data[12*PIX_W + c*16 +: 16]) * $signed(w_reg[12][c]) +
                                 $signed(window_data[13*PIX_W + c*16 +: 16]) * $signed(w_reg[11][c]) +
                                 $signed(window_data[14*PIX_W + c*16 +: 16]) * $signed(w_reg[10][c]);

                    p1_sum[3] <= $signed(window_data[15*PIX_W + c*16 +: 16]) * $signed(w_reg[ 9][c]) +
                                 $signed(window_data[16*PIX_W + c*16 +: 16]) * $signed(w_reg[ 8][c]) +
                                 $signed(window_data[17*PIX_W + c*16 +: 16]) * $signed(w_reg[ 7][c]) +
                                 $signed(window_data[18*PIX_W + c*16 +: 16]) * $signed(w_reg[ 6][c]) +
                                 $signed(window_data[19*PIX_W + c*16 +: 16]) * $signed(w_reg[ 5][c]);

                    p1_sum[4] <= $signed(window_data[20*PIX_W + c*16 +: 16]) * $signed(w_reg[ 4][c]) +
                                 $signed(window_data[21*PIX_W + c*16 +: 16]) * $signed(w_reg[ 3][c]) +
                                 $signed(window_data[22*PIX_W + c*16 +: 16]) * $signed(w_reg[ 2][c]) +
                                 $signed(window_data[23*PIX_W + c*16 +: 16]) * $signed(w_reg[ 1][c]) +
                                 $signed(window_data[24*PIX_W + c*16 +: 16]) * $signed(w_reg[ 0][c]);
                end
            end

            // PIPELINE STAGE 2: Final addition, shift, and clipping
            wire signed [47:0] total_sum = p1_sum[0] + p1_sum[1] + p1_sum[2] + p1_sum[3] + p1_sum[4];
            
            // 💡 【核心修正】補上 100% 對齊 PyTorch 的四捨五入 Rounding 加法器
            // 當頂層傳入 5 bits 時，會先加上 2^(5-1) = 16 再右移 5 bits，完美重現收斂修約
            wire signed [47:0] rounding_adder = (FRAC_BITS > 0) ? (48'sh1 << (FRAC_BITS - 1)) : 48'sh0;
            wire signed [47:0] shifted = (total_sum + rounding_adder) >>> FRAC_BITS;
            
            wire signed [47:0] b_extended = {{32{b_reg[c][15]}}, b_reg[c]};
            wire signed [47:0] final_val = shifted + b_extended;
            
            assign clip_wire[c] = (final_val > 32767)  ?  16'd32767 :
                                  (final_val < -32768) ? -16'd32768 : final_val[15:0];
        end
    endgenerate

    // Delay valid signal by 1 clock cycle to match the pipeline stage
    reg in_valid_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_valid_d1 <= 0;
        else in_valid_d1 <= in_valid;
    end

    // Final Output Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 0;
            flat_out_data <= 0;
        end else begin
            out_valid <= in_valid_d1;
            if (in_valid_d1) begin
                for (out_ch = 0; out_ch < CHANNELS; out_ch = out_ch + 1) begin
                    flat_out_data[out_ch*16 +: 16] <= clip_wire[out_ch];
                end
            end
        end
    end

endmodule