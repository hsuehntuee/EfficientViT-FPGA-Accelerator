`timescale 1ns/1ps

// ============================================================================
// litemla_core8 (v6, INT8): ReLU linear attention for one 16-dimensional head at a time.
//
//   ctx[r][c] = sum over tokens of relu(k[r]) * v[c]          c = 0..16, v[16] = 2^f1 (the "1" of the denominator)
//   num[c]    = sum over r of relu(q[r]) * ctx[r][c]          c = 0..15
//   den       = sum over r of relu(q[r]) * ctx[r][16]         (1 when it is not positive)
//   y[c]      = saturate_int8((floor(num[c] * 2^12 / den) + 2^(s[c]-1)) >>> s[c])
//
// q, k, v are int8. The context fits 26 bits, so q * context is one DSP multiply, and the quotient is below
// 2^20 in magnitude (|num| <= 128 * den / 2^f1), so the divider needs 20 quotient bits: ten stages of two bits.
// The k * v products are LUT multipliers.
//
// Protocol (driven by accel_core), unchanged from v5:
//   ctx_clr   pulse before the first k/v token of a pass: accumulator <= 0
//   kv_valid  one (k, v) token per cycle; it is added 4 cycles later
//   ctx_swap  pulse after the last k/v token of a head has been added and after
//             the previous head's last query has read the holding registers
//   q_valid   one q token per cycle; it reads the holding registers 2 cycles later
//   out_valid one result per cycle, 21 cycles after q_valid
// sh_in (16 x 4 bits) must be stable while the results of a head leave.
// ============================================================================
module litemla_core8 #(
    parameter DIM = 16,
    parameter N_TOKENS = 256,
    parameter QBITS = 12
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     ctx_clr,
    input  wire                     ctx_swap,
    input  wire                     kv_valid,
    input  wire [DIM*8-1:0]         k_in,
    input  wire [DIM*8-1:0]         v_in,
    input  wire [10:0]              one_in,          // 2^f1 for the head whose k / v arrive
    input  wire                     q_valid,
    input  wire [DIM*8-1:0]         q_in,
    input  wire [DIM*4-1:0]         sh_in,           // final right shift per output channel
    output reg  [DIM*8-1:0]         out_data,
    output reg                      out_valid
);
    localparam DIV_STAGES = 10;                      // 2 quotient bits per stage
    localparam NW = 38 + QBITS;                      // numerator width (magnitude)
    localparam DW = 38;                              // denominator width
    localparam QW = 2 * DIV_STAGES;                  // quotient bits

    integer i, s;
    genvar gr, gc;

    // ---------------------------------------------------------------- inputs (ReLU on q and k)
    reg signed [7:0]  kq [0:DIM-1];
    reg signed [10:0] vq [0:DIM];
    reg signed [7:0]  a_q [0:DIM-1];
    reg               kv_vr, q_vr, kv_v0, kv_v1, a_v, m_v, p_v, t1_v, t2_v, b_v;
    reg [DIM*8-1:0]   q_r, k_r, v_r;              // the SRAM outputs, registered before they fan out to the multipliers
    reg [10:0]        one_r;
    always @(posedge clk) begin q_r <= q_in; k_r <= k_in; v_r <= v_in; one_r <= one_in; end
    generate
        for (gr = 0; gr < DIM; gr = gr + 1) begin : in_qk
            wire signed [7:0] raw_q = q_r[gr*8 +: 8];
            wire signed [7:0] raw_k = k_r[gr*8 +: 8];
            wire signed [7:0] raw_v = v_r[gr*8 +: 8];
            always @(posedge clk) begin
                a_q[gr] <= (raw_q > 0) ? raw_q : 8'sd0;
                kq[gr]  <= (raw_k > 0) ? raw_k : 8'sd0;
                vq[gr]  <= raw_v;
            end
        end
    endgenerate
    always @(posedge clk) vq[DIM] <= {1'b0, one_r[9:0]};

    // ---------------------------------------------------------------- phase 1: context accumulate (LUT multipliers)
    wire [DIM*(DIM+1)*26-1:0] hold_flat;
    generate
        for (gr = 0; gr < DIM; gr = gr + 1) begin : mac_r
            for (gc = 0; gc <= DIM; gc = gc + 1) begin : mac_c
                wire signed [25:0] hold_w;
                att_kv_mac u_kv (.clk(clk), .clr(ctx_clr || ctx_swap), .add(kv_v1), .swap(ctx_swap), .k(kq[gr]), .v(vq[gc]), .hold(hold_w));
                assign hold_flat[(gr*(DIM+1) + gc)*26 +: 26] = hold_w;
            end
        end
    endgenerate

    // ---------------------------------------------------------------- phase 2 front: q . context (one DSP each), 16-term sums
    reg signed [37:0] b_dot [0:DIM];
    generate
        for (gc = 0; gc <= DIM; gc = gc + 1) begin : dot
            reg signed [33:0] m  [0:DIM-1];
            reg signed [33:0] pp [0:DIM-1];
            reg signed [35:0] sg [0:3];
            reg signed [37:0] th;
            integer r, g;
            always @(posedge clk) begin
                for (r = 0; r < DIM; r = r + 1) begin
                    m[r]  <= a_q[r] * $signed(hold_flat[(r*(DIM+1) + gc)*26 +: 26]);
                    pp[r] <= m[r];
                end
                for (g = 0; g < 4; g = g + 1) sg[g] <= pp[4*g] + pp[4*g+1] + pp[4*g+2] + pp[4*g+3];
                th <= sg[0] + sg[1] + sg[2] + sg[3];
                b_dot[gc] <= th;
            end
        end
    endgenerate

    // ---------------------------------------------------------------- normalise, divide (2 bits per stage), sign, round
    reg                c_v;
    reg        [NW-1:0] c_num  [0:DIM-1];
    reg                c_sign [0:DIM-1];
    reg        [DW-1:0] c_den;
    reg                d_v    [0:DIV_STAGES];
    reg        [DW:0]  d_rem  [0:DIV_STAGES][0:DIM-1];
    reg        [QW-1:0] d_quo [0:DIV_STAGES][0:DIM-1];
    reg        [QW-1:0] d_low [0:DIV_STAGES][0:DIM-1];     // numerator bits still to bring down
    reg                d_sign [0:DIV_STAGES][0:DIM-1];
    reg        [DW-1:0] d_den [0:DIV_STAGES];
    reg                e_v;
    reg signed [QW+1:0] e_val [0:DIM-1];

    reg signed [37:0]   norm_val;
    reg signed [NW:0]   shifted;
    reg        [DW+1:0] nr1, nr2;
    reg        [DW:0]   rem_t;
    reg        [QW-1:0] quo_t;
    reg signed [QW+2:0] rounded_val;
    reg        [3:0]    shv;

    always @(posedge clk) begin
        if (!rst_n) begin
            kv_vr <= 0; q_vr <= 0; kv_v0 <= 0; kv_v1 <= 0; a_v <= 0; m_v <= 0; p_v <= 0; t1_v <= 0; t2_v <= 0; b_v <= 0;
            c_v <= 0; e_v <= 0; out_valid <= 0;
            for (s = 0; s <= DIV_STAGES; s = s + 1) d_v[s] <= 0;
        end else begin
            kv_vr <= kv_valid; kv_v0 <= kv_vr; kv_v1 <= kv_v0;
            // q_valid | q_r | a_q | m | pp | sg | th | b_dot | c | d0 .. d10 | e | out
            q_vr <= q_valid; a_v <= q_vr; m_v <= a_v; p_v <= m_v; t1_v <= p_v; t2_v <= t1_v; b_v <= t2_v;
            c_v <= b_v;
            d_v[0] <= c_v;
            for (s = 0; s < DIV_STAGES; s = s + 1) d_v[s+1] <= d_v[s];
            e_v <= d_v[DIV_STAGES];
            out_valid <= e_v;
        end
    end

    always @(posedge clk) begin
        // C: magnitudes and sign
        norm_val = (b_dot[DIM] <= 0) ? 38'sd1 : b_dot[DIM];
        c_den <= norm_val[DW-1:0];
        for (i = 0; i < DIM; i = i + 1) begin
            shifted = $signed({b_dot[i], {QBITS{1'b0}}});
            c_num[i]  <= shifted[NW] ? -shifted[NW-1:0] : shifted[NW-1:0];
            c_sign[i] <= shifted[NW];
        end
        // D0: the quotient is below 2^QW, so the upper numerator bits are the first remainder
        d_den[0] <= c_den;
        for (i = 0; i < DIM; i = i + 1) begin
            d_rem[0][i]  <= {{(DW + 1 - (NW - QW)){1'b0}}, c_num[i][NW-1:QW]};
            d_low[0][i]  <= c_num[i][QW-1:0];
            d_quo[0][i]  <= {QW{1'b0}};
            d_sign[0][i] <= c_sign[i];
        end
        // D1..D10
        for (s = 0; s < DIV_STAGES; s = s + 1) begin
            d_den[s+1] <= d_den[s];
            for (i = 0; i < DIM; i = i + 1) begin
                quo_t = d_quo[s][i];
                nr1 = {d_rem[s][i], d_low[s][i][QW-1]};
                if (nr1 >= {2'b00, d_den[s]}) begin rem_t = nr1 - d_den[s]; quo_t = {quo_t[QW-2:0], 1'b1}; end
                else begin rem_t = nr1[DW:0]; quo_t = {quo_t[QW-2:0], 1'b0}; end
                nr2 = {rem_t, d_low[s][i][QW-2]};
                if (nr2 >= {2'b00, d_den[s]}) begin rem_t = nr2 - d_den[s]; quo_t = {quo_t[QW-2:0], 1'b1}; end
                else begin rem_t = nr2[DW:0]; quo_t = {quo_t[QW-2:0], 1'b0}; end
                d_rem[s+1][i] <= rem_t; d_quo[s+1][i] <= quo_t;
                d_low[s+1][i] <= {d_low[s][i][QW-3:0], 2'b00}; d_sign[s+1][i] <= d_sign[s][i];
            end
        end
        // E1: sign (floor toward minus infinity for negative quotients with a remainder)
        for (i = 0; i < DIM; i = i + 1) begin
            if (d_sign[DIV_STAGES][i] && d_rem[DIV_STAGES][i] != 0)
                e_val[i] <= -$signed({2'b00, d_quo[DIV_STAGES][i]}) - 1;
            else if (d_sign[DIV_STAGES][i])
                e_val[i] <= -$signed({2'b00, d_quo[DIV_STAGES][i]});
            else
                e_val[i] <= $signed({2'b00, d_quo[DIV_STAGES][i]});
        end
        // E2: round, saturate
        for (i = 0; i < DIM; i = i + 1) begin
            shv = sh_in[i*4 +: 4];
            rounded_val = (e_val[i] + ((shv == 4'd0) ? 0 : ($signed({{(QW+2){1'b0}}, 1'b1}) <<< (shv - 4'd1)))) >>> shv;
            if (rounded_val > 127)       out_data[i*8 +: 8] <= 8'd127;
            else if (rounded_val < -128) out_data[i*8 +: 8] <= -8'd128;
            else                         out_data[i*8 +: 8] <= rounded_val[7:0];
        end
    end
endmodule

// ---------------------------------------------------------------- one context entry: k * v (LUTs), accumulator, holding register
(* use_dsp = "no" *)
module att_kv_mac (
    input  wire               clk,
    input  wire               clr,
    input  wire               add,
    input  wire               swap,
    input  wire signed [7:0]  k,
    input  wire signed [10:0] v,
    output reg  signed [25:0] hold
);
    reg signed [18:0] kvp;
    reg signed [25:0] acc;
    always @(posedge clk) begin
        kvp <= k * v;
        if (clr)      acc <= 26'sd0;
        else if (add) acc <= acc + kvp;
        if (swap) hold <= acc;
    end
endmodule
