`timescale 1ns/1ps

// ReLU linear attention with two context banks: phase 1 of head h+1 (k/v accumulation into the
// free bank) runs while phase 2 of head h (q stream through the MAC + divider pipeline) reads the
// other bank. q tokens of consecutive heads stream back to back; each q carries its bank id through
// the pipeline. Arithmetic identical to litemla_core.v.
//
// Protocol (driven by accelerator_top):
//   p1_begin  : pulse, clear bank p1_bank and start accepting kv_valid tokens into it
//   kv_valid  : one (k, v) token per cycle, N_TOKENS of them
//   q_valid   : one q token per cycle; q_bank says which bank it uses
//   out_valid : one result per cycle, ~70 cycles after q_valid
module litemla_core2 #(
    parameter DIM = 16,
    parameter N_TOKENS = 256,
    parameter FRAC_BITS = 5
)(
    input  wire clk,
    input  wire rst_n,
    input  wire                     p1_begin,
    input  wire                     p1_bank,
    input  wire                     kv_valid,
    input  wire [DIM*16-1:0]        k_in,
    input  wire [(DIM+1)*16-1:0]    v_in,
    input  wire                     q_valid,
    input  wire                     q_bank,
    input  wire [DIM*16-1:0]        q_in,
    output reg  [DIM*16-1:0]        out_data,
    output reg                      out_valid
);
    localparam DIV_STAGES = 32;
    localparam signed [64:0] ROUND_ADD = 65'sd1 <<< (16 - FRAC_BITS - 1);

    reg signed [47:0] ctx0 [0:DIM-1][0:DIM];
    reg signed [47:0] ctx1 [0:DIM-1][0:DIM];
    reg               fill_bank;

    wire signed [15:0] q_vec [0:DIM-1];
    wire signed [15:0] k_vec [0:DIM-1];
    wire signed [15:0] v_vec [0:DIM];
    genvar gi;
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : unpack_qk
            wire signed [15:0] raw_q = q_in[gi*16 +: 16];
            wire signed [15:0] raw_k = k_in[gi*16 +: 16];
            assign q_vec[gi] = (raw_q > 0) ? raw_q : 16'd0;
            assign k_vec[gi] = (raw_k > 0) ? raw_k : 16'd0;
        end
        for (gi = 0; gi <= DIM; gi = gi + 1) begin : unpack_v
            assign v_vec[gi] = v_in[gi*16 +: 16];
        end
    endgenerate

    integer r, c, i, s;

    // ---------------------------------------------------------------- phase 1: accumulate into fill_bank
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_bank <= 0;
            for (r = 0; r < DIM; r = r + 1)
                for (c = 0; c <= DIM; c = c + 1) begin ctx0[r][c] <= 0; ctx1[r][c] <= 0; end
        end else begin
            if (p1_begin) begin
                fill_bank <= p1_bank;
                for (r = 0; r < DIM; r = r + 1)
                    for (c = 0; c <= DIM; c = c + 1)
                        if (p1_bank) ctx1[r][c] <= 0; else ctx0[r][c] <= 0;
            end else if (kv_valid) begin
                for (r = 0; r < DIM; r = r + 1)
                    for (c = 0; c <= DIM; c = c + 1) begin
                        if (fill_bank) ctx1[r][c] <= ctx1[r][c] + ($signed({{32{k_vec[r][15]}}, k_vec[r]}) * $signed({{32{v_vec[c][15]}}, v_vec[c]}));
                        else           ctx0[r][c] <= ctx0[r][c] + ($signed({{32{k_vec[r][15]}}, k_vec[r]}) * $signed({{32{v_vec[c][15]}}, v_vec[c]}));
                    end
            end
        end
    end

    // ---------------------------------------------------------------- phase 2 pipeline
    reg                a_v, a_bank;
    reg signed [15:0]  a_q [0:DIM-1];
    reg                b_v;
    reg signed [63:0]  b_dot [0:DIM];
    reg                c_v;
    reg        [63:0]  c_num  [0:DIM-1];
    reg                c_sign [0:DIM-1];
    reg        [64:0]  c_den;
    reg                d_v    [0:DIV_STAGES];
    reg        [64:0]  d_rem  [0:DIV_STAGES][0:DIM-1];
    reg        [63:0]  d_quo  [0:DIV_STAGES][0:DIM-1];
    reg        [63:0]  d_num  [0:DIV_STAGES][0:DIM-1];
    reg                d_sign [0:DIV_STAGES][0:DIM-1];
    reg        [64:0]  d_den  [0:DIV_STAGES];

    reg signed [63:0] mac_sum [0:DIM];
    reg signed [47:0] ctx_rd;
    reg signed [63:0] norm_val;
    reg signed [63:0] shifted [0:DIM-1];
    reg        [64:0] nr1, nr2, rem_t;
    reg        [63:0] quo_t;
    reg signed [64:0] final_val, rounded_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_v <= 0; a_bank <= 0; b_v <= 0; c_v <= 0; out_valid <= 0;
            for (s = 0; s <= DIV_STAGES; s = s + 1) d_v[s] <= 0;
        end else begin
            // A: capture q + bank
            a_v <= q_valid; a_bank <= q_bank;
            for (i = 0; i < DIM; i = i + 1) a_q[i] <= q_vec[i];
            // B: q . ctx[bank]
            b_v <= a_v;
            for (c = 0; c <= DIM; c = c + 1) begin
                mac_sum[c] = 64'd0;
                for (r = 0; r < DIM; r = r + 1) begin
                    ctx_rd = a_bank ? ctx1[r][c] : ctx0[r][c];
                    mac_sum[c] = mac_sum[c] + ($signed({{48{a_q[r][15]}}, a_q[r]}) * $signed({{16{ctx_rd[47]}}, ctx_rd}));
                end
                b_dot[c] <= mac_sum[c];
            end
            // C
            c_v <= b_v;
            norm_val = (b_dot[DIM] <= 0) ? 64'sd1 : b_dot[DIM];
            c_den <= (norm_val < 0) ? -norm_val : norm_val;
            for (i = 0; i < DIM; i = i + 1) begin
                shifted[i] = b_dot[i] <<< 16;
                c_num[i]  <= (shifted[i] < 0) ? -shifted[i] : shifted[i];
                c_sign[i] <= shifted[i][63] ^ norm_val[63];
            end
            // D0
            d_v[0] <= c_v; d_den[0] <= c_den;
            for (i = 0; i < DIM; i = i + 1) begin
                d_rem[0][i] <= 65'd0; d_quo[0][i] <= 64'd0; d_num[0][i] <= c_num[i]; d_sign[0][i] <= c_sign[i];
            end
            // D1..D32
            for (s = 0; s < DIV_STAGES; s = s + 1) begin
                d_v[s+1] <= d_v[s]; d_den[s+1] <= d_den[s];
                for (i = 0; i < DIM; i = i + 1) begin
                    quo_t = d_quo[s][i];
                    nr1 = {d_rem[s][i][63:0], d_num[s][i][63 - 2*s]};
                    if (nr1 >= d_den[s]) begin rem_t = nr1 - d_den[s]; quo_t[63 - 2*s] = 1'b1; end
                    else begin rem_t = nr1; quo_t[63 - 2*s] = 1'b0; end
                    nr2 = {rem_t[63:0], d_num[s][i][62 - 2*s]};
                    if (nr2 >= d_den[s]) begin rem_t = nr2 - d_den[s]; quo_t[62 - 2*s] = 1'b1; end
                    else begin rem_t = nr2; quo_t[62 - 2*s] = 1'b0; end
                    d_rem[s+1][i] <= rem_t; d_quo[s+1][i] <= quo_t;
                    d_num[s+1][i] <= d_num[s][i]; d_sign[s+1][i] <= d_sign[s][i];
                end
            end
            // E
            out_valid <= d_v[DIV_STAGES];
            for (i = 0; i < DIM; i = i + 1) begin
                if (d_sign[DIV_STAGES][i] && d_rem[DIV_STAGES][i] != 0)
                    final_val = -$signed({1'b0, d_quo[DIV_STAGES][i]}) - 65'sd1;
                else if (d_sign[DIV_STAGES][i])
                    final_val = -$signed({1'b0, d_quo[DIV_STAGES][i]});
                else
                    final_val = $signed({1'b0, d_quo[DIV_STAGES][i]});
                rounded_val = (final_val + ROUND_ADD) >>> (16 - FRAC_BITS);
                if (rounded_val > 32767)       out_data[i*16 +: 16] <= 16'd32767;
                else if (rounded_val < -32768) out_data[i*16 +: 16] <= -16'd32768;
                else                           out_data[i*16 +: 16] <= rounded_val[15:0];
            end
        end
    end
endmodule
