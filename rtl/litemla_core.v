`timescale 1ns/1ps

// ReLU linear attention, one head of DIM channels.
//   Phase 1 : ctx[r][c] += relu(k[r]) * vpad[c]          one token per cycle, vpad = [v, 1<<FRAC_BITS]
//   Phase 2 : out[c] = (relu(q) . ctx[:,c]) / (relu(q) . ctx[:,DIM])   one token per cycle, pipelined
//
// Arithmetic is identical to the 2026-05 version (48-bit context, 64-bit dot products, numerator << 16,
// 64-step restoring division with floor semantics, round-half-up to FRAC_BITS, saturate). The 2026-10-10
// change is only structural: phase 2 used to take ~70 cycles per token (one token in flight); it is now a
// pipeline that accepts one q per cycle (MAC -> PREP -> 32 x 2 division steps -> OUT), ~70 cycles latency.
module litemla_core #(
    parameter DIM = 16,
    parameter N_TOKENS = 256,
    parameter FRAC_BITS = 5
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    input  wire [DIM*16-1:0]        q_in,
    input  wire [DIM*16-1:0]        k_in,
    input  wire [(DIM+1)*16-1:0]    v_in,
    input  wire                     in_valid,
    output wire                     in_ready,
    output reg  [DIM*16-1:0]        out_data,
    output reg                      out_valid,
    input  wire                     out_ready       // tied to 1 by accelerator_top; the pipeline never stalls
);
    localparam ST_IDLE   = 2'd0;
    localparam ST_PHASE1 = 2'd1;
    localparam ST_PHASE2 = 2'd2;
    localparam DIV_STAGES = 32;                     // 2 restoring steps per stage, bits 63..0
    localparam signed [64:0] ROUND_ADD = 65'sd1 <<< (16 - FRAC_BITS - 1);

    reg [1:0]  state;
    reg [15:0] token_cnt;                           // phase 1 tokens accumulated
    reg [15:0] out_cnt;                             // phase 2 tokens emitted

    reg signed [47:0] context_mem [0:DIM-1][0:DIM];

    // ---------------------------------------------------------------- input unpack + ReLU
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

    assign in_ready = (state == ST_PHASE1) || (state == ST_PHASE2);

    // ---------------------------------------------------------------- phase 1 + control
    integer r, c, i;
    wire p2_fire = (state == ST_PHASE2) && in_valid;
    wire last_out = out_valid && (out_cnt == N_TOKENS - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE; token_cnt <= 0; out_cnt <= 0; done <= 0;
            for (r = 0; r < DIM; r = r + 1)
                for (c = 0; c <= DIM; c = c + 1) context_mem[r][c] <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= ST_PHASE1; token_cnt <= 0; out_cnt <= 0;
                        for (r = 0; r < DIM; r = r + 1)
                            for (c = 0; c <= DIM; c = c + 1) context_mem[r][c] <= 0;
                    end
                end
                ST_PHASE1: begin
                    if (in_valid) begin
                        for (r = 0; r < DIM; r = r + 1)
                            for (c = 0; c <= DIM; c = c + 1)
                                context_mem[r][c] <= context_mem[r][c] +
                                    ($signed({{32{k_vec[r][15]}}, k_vec[r]}) * $signed({{32{v_vec[c][15]}}, v_vec[c]}));
                        if (token_cnt == N_TOKENS - 1) begin state <= ST_PHASE2; token_cnt <= 0; end
                        else token_cnt <= token_cnt + 1;
                    end
                end
                ST_PHASE2: begin
                    if (out_valid) out_cnt <= out_cnt + 1;
                    if (last_out) begin state <= ST_IDLE; done <= 1'b1; end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------- phase 2 pipeline
    // stage A: q capture
    reg                a_v;
    reg signed [15:0]  a_q [0:DIM-1];
    // stage B: q . ctx (64-bit)
    reg                b_v;
    reg signed [63:0]  b_dot [0:DIM];
    // stage C: division operands
    reg                c_v;
    reg        [63:0]  c_num  [0:DIM-1];        // |dot << 16|, bits 63..0 are what the divider consumes
    reg                c_sign [0:DIM-1];
    reg        [64:0]  c_den;
    // division stages
    reg                d_v    [0:DIV_STAGES];
    reg        [64:0]  d_rem  [0:DIV_STAGES][0:DIM-1];
    reg        [63:0]  d_quo  [0:DIV_STAGES][0:DIM-1];
    reg        [63:0]  d_num  [0:DIV_STAGES][0:DIM-1];
    reg                d_sign [0:DIV_STAGES][0:DIM-1];
    reg        [64:0]  d_den  [0:DIV_STAGES];

    reg signed [63:0] mac_sum [0:DIM];
    reg signed [63:0] norm_val;
    reg signed [63:0] shifted [0:DIM-1];
    reg        [64:0] nr1, nr2;
    reg        [64:0] rem_t;
    reg        [63:0] quo_t;
    reg signed [64:0] final_val;
    reg signed [64:0] rounded_val;
    integer s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_v <= 0; b_v <= 0; c_v <= 0; out_valid <= 0;
            for (s = 0; s <= DIV_STAGES; s = s + 1) d_v[s] <= 0;
        end else begin
            // ---- A
            a_v <= p2_fire;
            for (i = 0; i < DIM; i = i + 1) a_q[i] <= q_vec[i];

            // ---- B: dot products (same expression as before, one cycle)
            b_v <= a_v;
            for (c = 0; c <= DIM; c = c + 1) begin
                mac_sum[c] = 64'd0;
                for (r = 0; r < DIM; r = r + 1)
                    mac_sum[c] = mac_sum[c] + ($signed({{48{a_q[r][15]}}, a_q[r]}) * $signed({{16{context_mem[r][c][47]}}, context_mem[r][c]}));
                b_dot[c] <= mac_sum[c];
            end

            // ---- C: |num|, sign, den (den <= 0 -> 1)
            c_v <= b_v;
            norm_val = (b_dot[DIM] <= 0) ? 64'sd1 : b_dot[DIM];
            c_den <= (norm_val < 0) ? -norm_val : norm_val;
            for (i = 0; i < DIM; i = i + 1) begin
                shifted[i] = b_dot[i] <<< 16;
                c_num[i]  <= (shifted[i] < 0) ? -shifted[i] : shifted[i];
                c_sign[i] <= shifted[i][63] ^ norm_val[63];
            end

            // ---- D0: load divider
            d_v[0] <= c_v;
            d_den[0] <= c_den;
            for (i = 0; i < DIM; i = i + 1) begin
                d_rem[0][i] <= 65'd0; d_quo[0][i] <= 64'd0; d_num[0][i] <= c_num[i]; d_sign[0][i] <= c_sign[i];
            end
            // ---- D1..D32: two restoring steps each, bit (63 - 2s) then (62 - 2s)
            for (s = 0; s < DIV_STAGES; s = s + 1) begin
                d_v[s+1]   <= d_v[s];
                d_den[s+1] <= d_den[s];
                for (i = 0; i < DIM; i = i + 1) begin
                    quo_t = d_quo[s][i];
                    nr1 = {d_rem[s][i][63:0], d_num[s][i][63 - 2*s]};
                    if (nr1 >= d_den[s]) begin rem_t = nr1 - d_den[s]; quo_t[63 - 2*s] = 1'b1; end
                    else begin rem_t = nr1; quo_t[63 - 2*s] = 1'b0; end
                    nr2 = {rem_t[63:0], d_num[s][i][62 - 2*s]};
                    if (nr2 >= d_den[s]) begin rem_t = nr2 - d_den[s]; quo_t[62 - 2*s] = 1'b1; end
                    else begin rem_t = nr2; quo_t[62 - 2*s] = 1'b0; end
                    d_rem[s+1][i]  <= rem_t;
                    d_quo[s+1][i]  <= quo_t;
                    d_num[s+1][i]  <= d_num[s][i];
                    d_sign[s+1][i] <= d_sign[s][i];
                end
            end

            // ---- E: floor correction, round, saturate
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
