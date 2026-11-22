`timescale 1ns/1ps

// ============================================================================
// Depthwise 5x5 path, the MS-branch group 1x1 engine, the int16 -> int8 input
// conversion and the saturating adder (v6, INT8).
//
// Every multiplier here is int8 x int8 and is built from LUTs (use_dsp = "no"):
// two such products cannot share a DSP48E2 because they have no common operand,
// and one product per DSP is what v5 spent 1,152 DSPs on.
// Every consumer counts the valid pulses, so latencies need no change in the sequencing.
// ============================================================================

// ---------------------------------------------------------------- 5x5 window generator (line buffer), zero padding by position
module dw_line_buffer #(
    parameter IMG_SIZE = 16,
    parameter CHANNELS = 16,
    parameter LW = 8                 // bits per channel
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    input  wire [CHANNELS*LW-1:0]      in_data,
    output reg                         out_valid,
    output wire [25*CHANNELS*LW-1:0]   window_data
);
    localparam PIX_W        = CHANNELS * LW;
    localparam SR_SIZE      = 4 * IMG_SIZE + 5;           // 5x5 needs 4 full rows + 5 pixels
    localparam CENTER_DELAY = 2 * IMG_SIZE + 2;           // cycles for the first pixel to reach the centre
    localparam TOTAL_PIXELS = IMG_SIZE * IMG_SIZE;

    reg [PIX_W-1:0] sr [0:SR_SIZE-1];
    integer i;
    reg [15:0] in_cnt;
    reg [15:0] out_r, out_c;
    reg [15:0] out_cnt;
    reg        out_active;
    wire shift_en = in_valid || out_active;

    always @(posedge clk) begin
        if (shift_en) begin
            sr[0] <= in_valid ? in_data : {PIX_W{1'b0}};
            for (i = 1; i < SR_SIZE; i = i + 1) sr[i] <= sr[i-1];
        end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            in_cnt <= 0; out_r <= 0; out_c <= 0; out_cnt <= 0; out_active <= 0; out_valid <= 0;
        end else begin
            out_valid <= 0;
            if (shift_en) begin
                if (in_valid) in_cnt <= in_cnt + 16'd1;
                if (in_cnt == CENTER_DELAY || out_active) begin
                    out_active <= 1;
                    out_valid <= 1;
                    if (out_active) begin
                        if (out_c == IMG_SIZE - 1) begin out_c <= 0; out_r <= out_r + 16'd1; end
                        else out_c <= out_c + 16'd1;
                        out_cnt <= out_cnt + 16'd1;
                        if (out_cnt == TOTAL_PIXELS - 1) begin
                            out_active <= 0; out_valid <= 0; out_cnt <= 0;
                            out_r <= 0; out_c <= 0; in_cnt <= 0;
                        end
                    end
                end
            end
        end
    end

    wire [4:0] pad_row = {(out_r + 2 >= IMG_SIZE), (out_r + 1 >= IMG_SIZE), 1'b0, (out_r < 1), (out_r < 2)};
    wire [4:0] pad_col = {(out_c + 2 >= IMG_SIZE), (out_c + 1 >= IMG_SIZE), 1'b0, (out_c < 1), (out_c < 2)};
    genvar ky, kx;
    generate
        for (ky = 0; ky < 5; ky = ky + 1) begin : wy
            for (kx = 0; kx < 5; kx = kx + 1) begin : wx
                // tap ky*5 + kx is the pixel at (row + ky - 2, col + kx - 2): the oldest pixel is the top-left tap
                assign window_data[(ky*5 + kx)*PIX_W +: PIX_W] =
                    (pad_row[ky] || pad_col[kx]) ? {PIX_W{1'b0}} : sr[(4 - ky)*IMG_SIZE + (4 - kx)];
            end
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- 5x5 depthwise MAC, CHANNELS channels in parallel
// Output per channel: v = (sum of 25 products + bias') >>> shift saturated to 12 bits, and the channel's fsel.
// The MS branch saturates v to int8; the MBConv depthwise feeds it to the hardswish table.
// out_valid comes 8 cycles after in_valid:
// window register | products | five 5-term sums | two partial totals | total | + bias' | shift | saturate.
(* use_dsp = "no" *)
module dw_conv_engine8 #(
    parameter CHANNELS = 16
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        load_wt,
    input  wire [CHANNELS*8-1:0]       flat_wt_in,
    input  wire [CHANNELS*32-1:0]      flat_prm_in,     // {shift[4:0], fsel[1:0], bias'[24:0]} per channel, stable while a chunk runs
    input  wire                        in_valid,
    input  wire [25*CHANNELS*8-1:0]    window_data,
    output reg                         out_valid,
    output wire [CHANNELS*12-1:0]      flat_out_p,
    output wire [CHANNELS*2-1:0]       flat_out_fsel
);
    localparam PIX_W = CHANNELS * 8;
    integer w_idx, ch_w;

    reg signed [7:0] w_reg [0:24][0:CHANNELS-1];
    always @(posedge clk) begin
        if (load_wt) begin
            for (ch_w = 0; ch_w < CHANNELS; ch_w = ch_w + 1) begin
                w_reg[0][ch_w] <= flat_wt_in[ch_w*8 +: 8];
                for (w_idx = 1; w_idx < 25; w_idx = w_idx + 1) w_reg[w_idx][ch_w] <= w_reg[w_idx-1][ch_w];
            end
        end
    end

    reg [6:0] vp;
    always @(posedge clk) begin
        if (!rst_n) begin vp <= 0; out_valid <= 0; end
        else begin vp <= {vp[5:0], in_valid}; out_valid <= vp[6]; end
    end

    genvar c, t;
    generate
        for (c = 0; c < CHANNELS; c = c + 1) begin : mac_tree
            reg signed [15:0] prod [0:24];
            for (t = 0; t < 25; t = t + 1) begin : tap
                reg signed [7:0] px;
                always @(posedge clk) begin
                    px      <= window_data[t*PIX_W + c*8 +: 8];
                    prod[t] <= px * w_reg[24 - t][c];
                end
            end
            // 25 products of at most 2^14 in magnitude: every intermediate value fits 20 bits
            reg signed [19:0] p1_sum [0:4];
            reg signed [19:0] t_a, t_b, tot;
            integer g;
            always @(posedge clk) begin
                for (g = 0; g < 5; g = g + 1)
                    p1_sum[g] <= prod[5*g] + prod[5*g+1] + prod[5*g+2] + prod[5*g+3] + prod[5*g+4];
                t_a <= p1_sum[0] + p1_sum[1] + p1_sum[2];
                t_b <= p1_sum[3] + p1_sum[4];
                tot <= t_a + t_b;
            end
            wire signed [26:0] v;
            wire [1:0]         fsel;
            requant_lane #(.AW(26)) u_rq (.clk(clk), .acc({{6{tot[19]}}, tot}), .prm(flat_prm_in[c*32 +: 32]), .v(v), .fsel(fsel));
            wire ov12 = (v[26:11] != {16{v[26]}});
            reg signed [11:0] p12;
            reg [1:0]         f3;
            always @(posedge clk) begin
                p12 <= ov12 ? (v[26] ? 12'h800 : 12'h7FF) : v[11:0];
                f3  <= fsel;
            end
            assign flat_out_p[c*12 +: 12]   = p12;
            assign flat_out_fsel[c*2 +: 2]  = f3;
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- weight-stationary PE of the MS-branch 1x1 engine
(* use_dsp = "no" *)
module pe_ws8 (
    input  wire               clk,
    input  wire               load_wt,
    input  wire signed [7:0]  act_in,
    input  wire signed [7:0]  wt_in,
    input  wire signed [19:0] psum_in,
    output reg  signed [7:0]  act_out,
    output wire signed [7:0]  wt_out,
    output reg  signed [19:0] psum_out
);
    reg signed [7:0]  wt_reg;
    reg signed [15:0] mult_r;
    assign wt_out = wt_reg;
    always @(posedge clk) begin
        if (load_wt) wt_reg <= wt_in;
        act_out  <= act_in;
        mult_r   <= act_in * wt_reg;
        psum_out <= psum_in + mult_r;
    end
endmodule

// ---------------------------------------------------------------- MS-branch group 1x1: one 16x16 array, weights loaded between chunks
// Output: int8 = saturate((sum + bias') >>> shift). out_valid comes ROWS + COLS + 3 cycles after act_valid.
// Nothing is in flight while load_wt is high.
module ms_pw_engine8 #(parameter ROWS = 16, parameter COLS = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 load_wt,
    input  wire                 act_valid,
    input  wire [ROWS*8-1:0]    flat_act_in,
    input  wire [COLS*8-1:0]    flat_wt_in,
    input  wire [COLS*32-1:0]   flat_prm_in,
    output wire [COLS*8-1:0]    flat_out_data,
    output wire                 out_valid
);
    wire [ROWS*8-1:0] skewed_act;
    skew_buffer #(.ROWS(ROWS), .WIDTH(8)) u_skew_act (.clk(clk), .flat_in(flat_act_in), .skewed_out(skewed_act));

    wire signed [7:0]  act  [0:ROWS-1][0:COLS];
    wire signed [7:0]  wt   [0:ROWS][0:COLS-1];
    wire signed [19:0] psum [0:ROWS][0:COLS-1];
    wire [COLS*20-1:0] skewed_psum, flat_psum;
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : bound_row
            assign act[r][0] = skewed_act[r*8 +: 8];
        end
        for (c = 0; c < COLS; c = c + 1) begin : bound_col
            assign wt[0][c]   = flat_wt_in[c*8 +: 8];
            assign psum[0][c] = 20'sd0;
            assign skewed_psum[c*20 +: 20] = psum[ROWS][c];
        end
        for (r = 0; r < ROWS; r = r + 1) begin : row
            for (c = 0; c < COLS; c = c + 1) begin : col
                pe_ws8 u_pe (.clk(clk), .load_wt(load_wt), .act_in(act[r][c]), .wt_in(wt[r][c]), .psum_in(psum[r][c]),
                             .act_out(act[r][c+1]), .wt_out(wt[r+1][c]), .psum_out(psum[r+1][c]));
            end
        end
    endgenerate
    deskew_buffer #(.COLS(COLS), .WIDTH(20)) u_deskew (.clk(clk), .skewed_in(skewed_psum), .flat_out(flat_psum));

    generate
        for (c = 0; c < COLS; c = c + 1) begin : rq
            wire signed [26:0] v;
            wire [1:0]         fsel_unused;
            wire signed [19:0] ps = flat_psum[c*20 +: 20];
            requant_lane #(.AW(26)) u_rq (.clk(clk), .acc({{6{ps[19]}}, ps}), .prm(flat_prm_in[c*32 +: 32]), .v(v), .fsel(fsel_unused));
            wire ov8 = (v[26:7] != {20{v[26]}});
            reg signed [7:0] y;
            always @(posedge clk) y <= ov8 ? (v[26] ? 8'h80 : 8'h7F) : v[7:0];
            assign flat_out_data[c*8 +: 8] = y;
        end
    endgenerate

    localparam TOTAL_DELAY = ROWS + COLS + 3;
    reg [TOTAL_DELAY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (!rst_n) valid_pipe <= 0;
        else        valid_pipe <= {valid_pipe[TOTAL_DELAY-2:0], act_valid};
    end
    assign out_valid = valid_pipe[TOTAL_DELAY-1];
endmodule

// ---------------------------------------------------------------- 16 saturating int16 adds
module adder_array (input wire [255:0] a, input wire [255:0] b, output wire [255:0] c);
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : add_blk
        wire signed [15:0] va = a[i*16 +: 16];
        wire signed [15:0] vb = b[i*16 +: 16];
        wire signed [16:0] s  = va + vb;
        assign c[i*16 +: 16] = (s > 32767) ? 16'd32767 : (s < -32768) ? -16'd32768 : s[15:0];
    end endgenerate
endmodule

// ---------------------------------------------------------------- 16 x (int16 Q10.5 -> int8): (x + 2^(s-1)) >>> s, saturate; s = 0..2 per lane
// Combinational; the caller registers the result.
module to_int8_array (input wire [255:0] x, input wire [31:0] sh, output wire [127:0] y);
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : cv
        wire signed [15:0] v  = x[i*16 +: 16];
        wire [1:0]         s  = sh[i*2 +: 2];
        wire signed [16:0] ve = v;
        wire signed [16:0] r1 = (ve + 17'sd1) >>> 1;
        wire signed [16:0] r2 = (ve + 17'sd2) >>> 2;
        wire signed [16:0] r  = (s == 2'd0) ? ve : (s == 2'd1) ? r1 : r2;
        wire ov = (r[16:7] != {10{r[16]}});
        assign y[i*8 +: 8] = ov ? (r[16] ? 8'h80 : 8'h7F) : r[7:0];
    end endgenerate
endmodule
