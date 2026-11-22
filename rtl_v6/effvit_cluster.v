`timescale 1ns/1ps

// ============================================================================
// conv_cluster (v6, INT8): K weight-stationary 16x16 systolic arrays. Every PE
// is one DSP48E2 that multiplies one int8 activation by TWO int8 weights at
// once, so an array computes 32 output channels (two 16-channel chunks) of a
// 1x1 convolution, and the cluster 2K chunks per step.
//
// Packing. A PE holds the weight pair as the 27-bit number  hi * 2^19 + lo.
// The weight store keeps it as 16 bits {hi', lo} with hi' = hi - (lo < 0), so
// the 27 bits are just {hi', 11 x lo[7], lo}: no logic between the weight
// register and the multiplier. With weights in [-127, 127] and activations in
// [-128, 127] the sum of the 16 "lo" products of a column stays below 2^18, so
// it fits the low 19 bits and the two sums separate again as
//     lo sum = signed P[18:0],    hi sum = signed P[47:19] + P[18].
//
// After the last input chunk each output channel is requantised with its own
// parameters {shift[4:0], fsel[1:0], bias'[24:0]} (bias' already contains the
// rounding constant 2^(shift-1)):  v = (sum + bias') >>> shift, then
//     mode 0: saturate to int8          (QKV)
//     mode 1: saturate to int16         (projection, point-wise: residual branches)
//     mode 2: hardswish table -> int8   (expand)
// Results leave in 16-bit lanes whatever the mode.
//
// Control (shadow weights, tiles, timing windows) is the v5 control with
// "output chunk" replaced by "pair of output chunks".
// ============================================================================

// ---------------------------------------------------------------- PE: one activation x two weights
module pe_sw8 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               load_wt,          // shift wt_in into the shadow chain
    input  wire signed [7:0]  act_in,
    input  wire               flag_in,          // 1 on the first token of a tile: use + adopt the shadow weight
    input  wire [15:0]        wt_in,            // {hi', lo}
    input  wire signed [47:0] psum_in,
    output reg  signed [7:0]  act_out,
    output reg                flag_out,
    output wire [15:0]        wt_out,
    output reg  signed [47:0] psum_out
);
    // As in v5 the product is taken from wt_reg and act_out (the weight a token must use is what wt_reg holds one cycle
    // later, act_out is act_in one cycle later): no multiplexer in front of the multiplier.
    reg [15:0] wt_reg, wt_sh;
    wire signed [26:0] wt_pair = {wt_reg[15:8], {11{wt_reg[7]}}, wt_reg[7:0]};
    reg signed [34:0] mult_r;
    assign wt_out = wt_sh;
    always @(posedge clk) begin
        if (load_wt) wt_sh <= wt_in;
        if (flag_in) wt_reg <= wt_sh;
        act_out  <= act_in;
        mult_r   <= act_out * wt_pair;
        psum_out <= psum_in + mult_r;
    end
    always @(posedge clk) begin
        if (!rst_n) flag_out <= 1'b0;
        else        flag_out <= flag_in;
    end
endmodule

// ---------------------------------------------------------------- skew / deskew delay lines (no reset: they map to SRLs)
module skew_buffer #(parameter ROWS = 16, parameter WIDTH = 16) (
    input  wire                  clk,
    input  wire [ROWS*WIDTH-1:0] flat_in,
    output wire [ROWS*WIDTH-1:0] skewed_out
);
    genvar r;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_skew
            if (r == 0) begin : gen_row_0
                assign skewed_out[WIDTH-1:0] = flat_in[WIDTH-1:0];
            end else begin : gen_row_delay
                reg [WIDTH-1:0] shift_reg [0:r-1];
                integer i;
                always @(posedge clk) begin
                    shift_reg[0] <= flat_in[r*WIDTH +: WIDTH];
                    for (i = 1; i < r; i = i + 1) shift_reg[i] <= shift_reg[i-1];
                end
                assign skewed_out[r*WIDTH +: WIDTH] = shift_reg[r-1];
            end
        end
    endgenerate
endmodule

module deskew_buffer #(parameter COLS = 16, parameter WIDTH = 48) (
    input  wire                  clk,
    input  wire [COLS*WIDTH-1:0] skewed_in,
    output wire [COLS*WIDTH-1:0] flat_out
);
    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : col_deskew
            localparam DELAY = COLS - 1 - c;
            if (DELAY == 0) begin : gen_col_0
                assign flat_out[c*WIDTH +: WIDTH] = skewed_in[c*WIDTH +: WIDTH];
            end else begin : gen_col_delay
                reg [WIDTH-1:0] shift_reg [0:DELAY-1];
                integer i;
                always @(posedge clk) begin
                    shift_reg[0] <= skewed_in[c*WIDTH +: WIDTH];
                    for (i = 1; i < DELAY; i = i + 1) shift_reg[i] <= shift_reg[i-1];
                end
                assign flat_out[c*WIDTH +: WIDTH] = shift_reg[DELAY-1];
            end
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- array
module systolic_array_sw8 #(parameter ROWS = 16, parameter COLS = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 load_wt,
    input  wire [ROWS*9-1:0]    act_in_row,     // {flag, act[7:0]} per row
    input  wire [COLS*16-1:0]   wt_in_col,
    output wire [COLS*48-1:0]   psum_out_col
);
    wire signed [7:0]  act  [0:ROWS-1][0:COLS];
    wire               flg  [0:ROWS-1][0:COLS];
    wire [15:0]        wt   [0:ROWS][0:COLS-1];
    wire signed [47:0] psum [0:ROWS][0:COLS-1];
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : bound_row
            assign act[r][0] = act_in_row[r*9 +: 8];
            assign flg[r][0] = act_in_row[r*9 + 8];
        end
        for (c = 0; c < COLS; c = c + 1) begin : bound_col
            assign wt[0][c]   = wt_in_col[c*16 +: 16];
            assign psum[0][c] = 48'd0;
            assign psum_out_col[c*48 +: 48] = psum[ROWS][c];
        end
        for (r = 0; r < ROWS; r = r + 1) begin : row
            for (c = 0; c < COLS; c = c + 1) begin : col
                pe_sw8 u_pe (.clk(clk), .rst_n(rst_n), .load_wt(load_wt),
                             .act_in(act[r][c]), .flag_in(flg[r][c]), .wt_in(wt[r][c]), .psum_in(psum[r][c]),
                             .act_out(act[r][c+1]), .flag_out(flg[r][c+1]), .wt_out(wt[r+1][c]), .psum_out(psum[r+1][c]));
            end
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- engine: input regs -> skew -> array -> deskew
// out_valid and flat_psum_out come ROWS + COLS + 2 cycles after act_valid (as in v5).
module conv_engine_sw8 #(parameter ROWS = 16, parameter COLS = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 load_wt,
    input  wire                 act_valid,
    input  wire                 act_flag,
    input  wire [ROWS*8-1:0]    flat_act_in,
    input  wire [COLS*16-1:0]   flat_wt_in,
    output wire [COLS*48-1:0]   flat_psum_out,
    output wire                 out_valid
);
    reg                load_r, valid_r, flag_r;
    reg [ROWS*8-1:0]   act_r;
    reg [COLS*16-1:0]  wt_r;
    always @(posedge clk) begin act_r <= flat_act_in; wt_r <= flat_wt_in; end
    always @(posedge clk) begin
        if (!rst_n) begin load_r <= 1'b0; valid_r <= 1'b0; flag_r <= 1'b0; end
        else        begin load_r <= load_wt; valid_r <= act_valid; flag_r <= act_flag; end
    end

    wire [ROWS*9-1:0] act_f, skewed_act;
    genvar r;
    generate for (r = 0; r < ROWS; r = r + 1) begin : pack
        assign act_f[r*9 +: 9] = {flag_r, act_r[r*8 +: 8]};
    end endgenerate
    skew_buffer #(.ROWS(ROWS), .WIDTH(9)) u_skew (.clk(clk), .flat_in(act_f), .skewed_out(skewed_act));
    wire [COLS*48-1:0] skewed_psum;
    systolic_array_sw8 #(.ROWS(ROWS), .COLS(COLS)) u_arr (.clk(clk), .rst_n(rst_n), .load_wt(load_r),
                                                        .act_in_row(skewed_act), .wt_in_col(wt_r), .psum_out_col(skewed_psum));
    deskew_buffer #(.COLS(COLS), .WIDTH(48)) u_deskew (.clk(clk), .skewed_in(skewed_psum), .flat_out(flat_psum_out));
    localparam TOTAL_DELAY = ROWS + COLS + 1;      // after valid_r
    reg [TOTAL_DELAY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (!rst_n) valid_pipe <= 0;
        else        valid_pipe <= {valid_pipe[TOTAL_DELAY-2:0], valid_r};
    end
    assign out_valid = valid_pipe[TOTAL_DELAY-1];
endmodule

// ---------------------------------------------------------------- integer hardswish table, two lookups per memory
// y = round(x * clamp(x + 3, 0, 6) / 6) with F = F_LO + fsel fractional bits on both sides (fsel = 0..3), saturated to int8.
//   p >= HI(F)      : min(p, 127)      (x >= 3 passes through; for F >= 6, HI is where the result reaches 127)
//   p <= -3 * 2^F   : 0
//   otherwise       : rom[base(fsel) + p + 3 * 2^F] = floor((p * (p + 3 * 2^F) + 3 * 2^F) / (6 * 2^F))
// Latency 3.
module hswish_rom2 #(parameter F_LO = 3, parameter AW = 10) (
    input  wire               clk,
    input  wire signed [11:0] p_a,
    input  wire [1:0]         f_a,
    output reg  signed [7:0]  y_a,
    input  wire signed [11:0] p_b,
    input  wire [1:0]         f_b,
    output reg  signed [7:0]  y_b
);
    function integer hi_of(input integer f);
        begin
            case (f)
                3: hi_of = 24;
                4: hi_of = 48;
                5: hi_of = 96;
                6: hi_of = 145;
                7: hi_of = 175;
                default: hi_of = 201;
            endcase
        end
    endfunction
    function integer base_of(input integer k);
        integer j;
        begin
            base_of = 0;
            for (j = 0; j < k; j = j + 1) base_of = base_of + 3 * (1 << (F_LO + j)) + hi_of(F_LO + j);
        end
    endfunction

    // the table starts as constants: base_of loops, so it may only be called with a constant
    localparam integer B1 = base_of(1), B2 = base_of(2), B3 = base_of(3);
    (* rom_style = "block" *) reg [7:0] rom [0:(1 << AW) - 1];
    integer ik, ip, it, id, iy, iz;
    initial begin
        // two nested loops: the synthesis tool stops a single loop at 2000 iterations
        for (iz = 0; iz < (1 << (AW - 6)); iz = iz + 1)
            for (ip = 0; ip < 64; ip = ip + 1) rom[iz * 64 + ip] = 8'd0;
        for (ik = 0; ik < 4; ik = ik + 1) begin
            id = 6 * (1 << (F_LO + ik));
            for (ip = -3 * (1 << (F_LO + ik)); ip < hi_of(F_LO + ik); ip = ip + 1) begin
                it = ip * (ip + 3 * (1 << (F_LO + ik))) + 3 * (1 << (F_LO + ik));
                iy = (it >= 0) ? (it / id) : -((-it + id - 1) / id);
                if (iy > 127) iy = 127;
                if (iy < -128) iy = -128;
                rom[base_of(ik) + ip + 3 * (1 << (F_LO + ik))] = iy[7:0];
            end
        end
    end

    // stage 1
    reg signed [11:0] pa1, pb1;
    reg [1:0]         fa1, fb1;
    always @(posedge clk) begin pa1 <= p_a; fa1 <= f_a; pb1 <= p_b; fb1 <= f_b; end
    function [AW+13:0] look(input signed [11:0] p, input [1:0] f);   // {hi, lo, min(p,127)[7:0] .. , addr}
        integer three, hi, base;
        reg signed [12:0] idx;
        reg [7:0] pm;
        begin
            three = 3 * (1 << (F_LO + f));
            hi    = hi_of(F_LO + f);
            base  = (f == 2'd0) ? 0 : (f == 2'd1) ? B1 : (f == 2'd2) ? B2 : B3;
            idx   = p + three;
            pm    = (p > 127) ? 8'd127 : p[7:0];
            look  = {(p >= hi), (p <= -three), pm, {(AW+4){1'b0}}};
            look[AW-1:0] = base + idx;
        end
    endfunction
    wire [AW+13:0] la = look(pa1, fa1);
    wire [AW+13:0] lb = look(pb1, fb1);
    // stage 2: memory
    reg [7:0] qa, qb, pma, pmb;
    reg       hia, loa, hib, lob;
    always @(posedge clk) begin
        qa <= rom[la[AW-1:0]];
        qb <= rom[lb[AW-1:0]];
        {hia, loa, pma} <= la[AW+13:AW+4];
        {hib, lob, pmb} <= lb[AW+13:AW+4];
    end
    // stage 3
    always @(posedge clk) begin
        y_a <= loa ? 8'sd0 : hia ? pma : qa;
        y_b <= lob ? 8'sd0 : hib ? pmb : qb;
    end
endmodule

// ---------------------------------------------------------------- per-channel requantisation: (acc + bias') >>> shift
// Two registers: sum, then shift. prm = {shift[4:0], fsel[1:0], bias'[24:0]}.
module requant_lane #(parameter AW = 26) (
    input  wire                 clk,
    input  wire signed [AW-1:0] acc,
    input  wire [31:0]          prm,
    output reg  signed [AW:0]   v,
    output reg  [1:0]           fsel
);
    reg signed [AW:0] s;
    reg [4:0]         sh;
    reg [1:0]         f;
    wire signed [24:0] bias = prm[24:0];
    always @(posedge clk) begin
        s    <= acc + bias;
        sh   <= prm[31:27];
        f    <= prm[26:25];
        v    <= s >>> sh;
        fsel <= f;
    end
endmodule

// ---------------------------------------------------------------- psum accumulate across in-chunks, then requantise
// 32 lanes: lanes 0..15 are the "lo" sums of the 16 columns (first chunk of the pair), 16..31 the "hi" sums.
// out_valid comes 11 cycles after the in_valid of the last in-chunk.
module psum_buffer8 #(parameter COLS = 16, parameter N_TOKENS = 256) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clear_psum,
    input  wire [7:0]             chunks,
    input  wire [1:0]             mode,
    input  wire                   in_valid,
    input  wire [COLS*48-1:0]     flat_psum_in,
    input  wire [2*COLS*32-1:0]   flat_prm_in,
    output reg                    out_valid,
    output reg  [2*COLS*16-1:0]   flat_out_data
);
    localparam L  = 2 * COLS;
    localparam SW = 26;                           // running sum: 64 chunks x 260,096 < 2^24
    (* ram_style = "block" *) reg [L*SW-1:0] ram [0:N_TOKENS-1];
    reg [L*SW-1:0]    ram_q0, ram_q, sum_r;
    reg [COLS*48-1:0] psum_d1;
    reg [L*20-1:0]    u_d2;
    reg v_d1, v_d2, sum_v;
    reg [6:0] vp;
    reg [15:0] rd_addr, a_d1, a_d2, chunk_cnt;

    always @(posedge clk) begin
        if (!rst_n) rd_addr <= 0;
        else if (clear_psum) rd_addr <= 0;
        else if (in_valid) rd_addr <= (rd_addr == N_TOKENS-1) ? 16'd0 : rd_addr + 16'd1;
    end
    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : unpack
            wire signed [18:0] lo = psum_d1[c*48 +: 19];
            wire signed [28:0] hi = psum_d1[c*48 + 19 +: 29];
            wire signed [19:0] hi_fix = hi[19:0] + {19'd0, psum_d1[c*48 + 18]};
            always @(posedge clk) begin
                u_d2[c*20 +: 20]          <= {lo[18], lo};
                u_d2[(COLS + c)*20 +: 20] <= hi_fix;
            end
        end
    endgenerate
    // the running sum of a token is read two cycles before it is used and written back in the cycle it is used
    always @(posedge clk) begin
        ram_q0  <= ram[rd_addr];
        ram_q   <= ram_q0;
        psum_d1 <= flat_psum_in;
        a_d1    <= rd_addr;
        a_d2    <= a_d1;
    end
    always @(posedge clk) begin
        if (!rst_n) begin v_d1 <= 1'b0; v_d2 <= 1'b0; end
        else        begin v_d1 <= in_valid; v_d2 <= v_d1; end
    end
    always @(posedge clk) begin
        if (!rst_n) chunk_cnt <= 0;
        else if (clear_psum) chunk_cnt <= 0;
        else if (v_d2 && a_d2 == N_TOKENS-1) chunk_cnt <= (chunk_cnt == chunks-1) ? 16'd0 : chunk_cnt + 16'd1;
    end
    wire [L*SW-1:0] sum_out;
    genvar l;
    generate
        for (l = 0; l < L; l = l + 1) begin : adder
            wire signed [19:0]   in_val  = u_d2[l*20 +: 20];
            wire signed [SW-1:0] ram_val = (chunk_cnt == 0) ? {SW{1'b0}} : ram_q[l*SW +: SW];
            assign sum_out[l*SW +: SW] = in_val + ram_val;
        end
    endgenerate
    always @(posedge clk) begin
        if (v_d2) ram[a_d2] <= sum_out;
        sum_r <= sum_out;
    end
    // sum_v | bias | shift | saturate | table 1 | table 2 | table 3 | output mux
    always @(posedge clk) begin
        if (!rst_n) begin sum_v <= 1'b0; vp <= 7'd0; out_valid <= 1'b0; end
        else begin sum_v <= v_d2 && (chunk_cnt == chunks-1); vp <= {vp[5:0], sum_v}; out_valid <= vp[5]; end
    end
    wire [L*8-1:0] hs_y;
    generate
        for (l = 0; l < L; l = l + 1) begin : rq
            wire signed [SW:0] v;
            wire [1:0]         fsel;
            requant_lane #(.AW(SW)) u_rq (.clk(clk), .acc(sum_r[l*SW +: SW]), .prm(flat_prm_in[l*32 +: 32]), .v(v), .fsel(fsel));
            wire ov16 = (v[SW:15] != {(SW-14){v[SW]}});
            wire ov12 = (v[SW:11] != {(SW-10){v[SW]}});
            reg signed [15:0] s16, s16_d1, s16_d2, s16_d3;
            reg signed [11:0] p12;
            reg [1:0]         f3;
            always @(posedge clk) begin
                s16 <= ov16 ? (v[SW] ? 16'h8000 : 16'h7FFF) : v[15:0];
                p12 <= ov12 ? (v[SW] ? 12'h800 : 12'h7FF) : v[11:0];
                f3  <= fsel;
                s16_d1 <= s16; s16_d2 <= s16_d1; s16_d3 <= s16_d2;
            end
            wire signed [7:0] y8 = hs_y[l*8 +: 8];
            wire ov8 = (s16_d3[15:7] != {9{s16_d3[15]}});
            wire signed [7:0] s8 = ov8 ? (s16_d3[15] ? 8'h80 : 8'h7F) : s16_d3[7:0];
            always @(posedge clk)
                flat_out_data[l*16 +: 16] <= (mode == 2'd2) ? {{8{y8[7]}}, y8} : (mode == 2'd0) ? {{8{s8[7]}}, s8} : s16_d3;
        end
        for (l = 0; l < L; l = l + 2) begin : hs
            hswish_rom2 #(.F_LO(3), .AW(10)) u_hs (.clk(clk),
                .p_a(rq[l].p12),   .f_a(rq[l].f3),   .y_a(hs_y[l*8 +: 8]),
                .p_b(rq[l+1].p12), .f_b(rq[l+1].f3), .y_b(hs_y[(l+1)*8 +: 8]));
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- the cluster
module conv_cluster8 #(
    parameter K = 8,                 // arrays; 2K output chunks per step
    parameter N_TOKENS = 256,
    parameter ACT_LAT = 5,           // act_data arrives ACT_LAT cycles after (act_token, act_ic)
    parameter W_LAT = 6,             // w_rdata arrives W_LAT cycles after w_addr
    parameter B_LAT = 2              // b_rdata arrives B_LAT cycles after b_addr
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [7:0]           cfg_oc,          // output chunks (multiple of 2K)
    input  wire [7:0]           cfg_ic,          // input chunks
    input  wire                 cfg_ic_major,    // 0: tile = pair*IC + ic (M1/M3/P2); 1: tile = ic*PAIRS + pair (P3)
    input  wire [1:0]           cfg_mode,        // 0 int8, 1 int16, 2 hardswish
    input  wire [17:0]          cfg_w_base,      // weight word address of tile 0
    input  wire [9:0]           cfg_b_base,      // parameter entry (512 bits) of out-chunk 0
    output reg                  busy,
    output reg                  done,            // 1-cycle pulse

    output wire [15:0]          act_token,       // activation request: (token, ic) this cycle
    output wire [7:0]           act_ic,
    output wire                 act_fire,
    input  wire [127:0]         act_data,        // 16 x int8, requested ACT_LAT cycles earlier

    output reg  [17:0]          w_addr,          // weight word address (registered)
    input  wire [255:0]         w_rdata,
    output wire [9:0]           b_addr,
    input  wire [511:0]         b_rdata,         // parameters of one out-chunk: 16 x 32 bits

    output wire                 out_valid,       // 2K output chunks of one token
    output wire [15:0]          out_token,
    output wire [7:0]           out_oc_base,
    output wire [2*K*256-1:0]   out_data         // chunk c at [c*256 +: 256], 16-bit lanes
);
    localparam S_IDLE = 3'd0, S_BIAS = 3'd1, S_LOAD0 = 3'd2, S_LOADW = 3'd3, S_CALC = 3'd4, S_DRAIN = 3'd5;
    localparam LOAD_START = 32;
    localparam WL = W_LAT + 2;                   // + the tile register and the w_addr register

    reg [2:0]  state;
    wire       out_done;                         // every output group delivered
    reg [7:0]  oc_n, ic_n, pairs, groups;        // layer config latched at start
    reg        ic_major;
    reg [1:0]  mode;
    reg [17:0] w_base;
    reg [9:0]  b_base;

    // ---- parameter ram (one 512-bit entry per output chunk; even and odd chunks apart), filled at layer start
    reg [511:0] prm_ram_e [0:31];
    reg [511:0] prm_ram_o [0:31];
    reg [7:0]   bias_cnt;
    reg [B_LAT-1:0] bias_wr_sr;
    reg [7:0]   bias_idx_sr [0:B_LAT-1];
    assign b_addr = b_base + {2'd0, bias_cnt};

    // ---- compute stream
    reg [15:0] tok;
    reg [7:0]  ic, grp;
    reg        act_fire_r;
    assign act_token = tok;
    assign act_ic    = ic;
    assign act_fire  = act_fire_r;
    reg [ACT_LAT-1:0] act_v_sr, act_f_sr;        // aligned with act_data at [ACT_LAT-1]

    wire last_tok  = (tok == N_TOKENS - 1);
    wire last_ic   = (ic == ic_n - 1);
    wire last_grp  = (grp == groups - 1);
    wire last_step = last_ic && last_grp;

    // ---- weight loader: shadow-loads K tiles of step (ld_grp, ld_ic)
    reg        ld_active;
    reg [7:0]  ld_k, ld_j;                       // array index, row counter 0..15
    reg [7:0]  ld_grp, ld_ic;
    wire [15:0] ld_pair = ld_grp * K + ld_k;
    wire [15:0] ld_tile = ic_major ? (ld_ic * pairs + ld_pair) : (ld_pair * ic_n + ld_ic);
    reg [WL-1:0] ld_act_sr;                      // aligned with w_rdata at [WL-1]
    reg [7:0]    ld_k_sr [0:WL-1];
    wire       ld_last = ld_active && (ld_j == 15) && (ld_k == K - 1);
    reg [7:0]  l0_cnt;
    reg [15:0] tile_q;
    reg [7:0]  j_q;

    // next step after (grp, ic)
    wire [7:0] nx_ic  = last_ic ? 8'd0 : ic + 8'd1;
    wire [7:0] nx_grp = last_ic ? grp + 8'd1 : grp;

    integer si;
    always @(posedge clk) begin
        tile_q <= ld_tile;
        j_q    <= ld_j;
        w_addr <= w_base + {tile_q[13:0], 4'd0} + (18'd15 - {10'd0, j_q});
        ld_k_sr[0] <= ld_k;
        for (si = 1; si < WL; si = si + 1) ld_k_sr[si] <= ld_k_sr[si-1];
        bias_idx_sr[0] <= bias_cnt;
        for (si = 1; si < B_LAT; si = si + 1) bias_idx_sr[si] <= bias_idx_sr[si-1];
        if (bias_wr_sr[B_LAT-1] && !bias_idx_sr[B_LAT-1][0]) prm_ram_e[bias_idx_sr[B_LAT-1][5:1]] <= b_rdata;
        if (bias_wr_sr[B_LAT-1] &&  bias_idx_sr[B_LAT-1][0]) prm_ram_o[bias_idx_sr[B_LAT-1][5:1]] <= b_rdata;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 0; done <= 0;
            oc_n <= 0; ic_n <= 0; pairs <= 0; groups <= 0; ic_major <= 0; mode <= 0; w_base <= 0; b_base <= 0;
            bias_cnt <= 0; bias_wr_sr <= 0;
            tok <= 0; ic <= 0; grp <= 0; act_fire_r <= 0; act_v_sr <= 0; act_f_sr <= 0;
            ld_active <= 0; ld_k <= 0; ld_j <= 0; ld_grp <= 0; ld_ic <= 0; ld_act_sr <= 0; l0_cnt <= 0;
        end else begin
            done <= 0;
            act_v_sr   <= {act_v_sr, act_fire_r};
            act_f_sr   <= {act_f_sr, act_fire_r && (tok == 0)};
            ld_act_sr  <= {ld_act_sr, ld_active};
            bias_wr_sr <= {bias_wr_sr, (state == S_BIAS)};

            // weight loader runs independently of the main state
            if (ld_active) begin
                if (ld_j == 15) begin
                    ld_j <= 0;
                    if (ld_k == K - 1) begin ld_k <= 0; ld_active <= 0; end
                    else ld_k <= ld_k + 8'd1;
                end else ld_j <= ld_j + 8'd1;
            end

            case (state)
                S_IDLE: begin
                    act_fire_r <= 0;
                    if (start) begin
                        oc_n <= cfg_oc; ic_n <= cfg_ic; pairs <= cfg_oc / 2; groups <= cfg_oc / (2 * K); ic_major <= cfg_ic_major;
                        mode <= cfg_mode; w_base <= cfg_w_base; b_base <= cfg_b_base;
                        busy <= 1; bias_cnt <= 0; state <= S_BIAS;
                    end
                end
                S_BIAS: begin                                   // read oc_n parameter entries
                    if (bias_cnt == oc_n - 1) begin
                        bias_cnt <= 0;
                        ld_active <= 1; ld_k <= 0; ld_j <= 0; ld_grp <= 0; ld_ic <= 0;   // initial shadow load, step 0
                        state <= S_LOAD0;
                    end else bias_cnt <= bias_cnt + 8'd1;
                end
                S_LOAD0: begin
                    l0_cnt <= 0;
                    if (ld_last) state <= S_LOADW;
                end
                S_LOADW: begin                                  // let the last weight word reach the shadow registers
                    if (l0_cnt == WL + 2) begin
                        tok <= 0; ic <= 0; grp <= 0; act_fire_r <= 1; state <= S_CALC;
                    end else l0_cnt <= l0_cnt + 8'd1;
                end
                S_CALC: begin
                    // kick the loader for the next step once the swap wavefront has passed
                    if (tok == LOAD_START && !last_step) begin
                        ld_active <= 1; ld_k <= 0; ld_j <= 0; ld_grp <= nx_grp; ld_ic <= nx_ic;
                    end
                    if (last_tok) begin
                        tok <= 0;
                        if (last_step) begin act_fire_r <= 0; state <= S_DRAIN; end
                        else begin ic <= nx_ic; grp <= nx_grp; end
                    end else tok <= tok + 16'd1;
                end
                S_DRAIN: begin
                    if (out_done) begin busy <= 0; done <= 1; state <= S_IDLE; end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ---- arrays + psum buffers
    wire clear = (state == S_BIAS);
    wire eng_valid;
    wire [K*768-1:0] psum_flat;
    reg  [15:0] out_tok;
    reg  [7:0]  out_grp;
    wire [K-1:0] pb_valid;
    genvar k;
    generate
        for (k = 0; k < K; k = k + 1) begin : arr
            wire ov;
            conv_engine_sw8 u_eng (
                .clk(clk), .rst_n(rst_n), .load_wt(ld_act_sr[WL-1] && (ld_k_sr[WL-1] == k)),
                .act_valid(act_v_sr[ACT_LAT-1]), .act_flag(act_f_sr[ACT_LAT-1]), .flat_act_in(act_data), .flat_wt_in(w_rdata),
                .flat_psum_out(psum_flat[k*768 +: 768]), .out_valid(ov)
            );
            if (k == 0) begin : v0
                assign eng_valid = ov;
            end
            reg [4:0]    prm_ra;                 // the parameters of a group are needed (IC - 1) * N_TOKENS cycles after out_grp changes
            reg [1023:0] prm_word;
            always @(posedge clk) begin
                prm_ra   <= out_grp[1:0] * K + k;
                prm_word <= {prm_ram_o[prm_ra], prm_ram_e[prm_ra]};
            end
            psum_buffer8 #(.COLS(16), .N_TOKENS(N_TOKENS)) u_pb (
                .clk(clk), .rst_n(rst_n), .clear_psum(clear), .chunks(ic_n), .mode(mode), .in_valid(eng_valid),
                .flat_psum_in(psum_flat[k*768 +: 768]), .flat_prm_in(prm_word),
                .out_valid(pb_valid[k]), .flat_out_data(out_data[k*512 +: 512])
            );
        end
    endgenerate

    assign out_valid   = pb_valid[0];
    assign out_token   = out_tok;
    assign out_oc_base = out_grp * (2 * K);
    assign out_done = (out_grp == groups);
    always @(posedge clk) begin
        if (!rst_n) begin out_tok <= 0; out_grp <= 0; end
        else if (clear) begin out_tok <= 0; out_grp <= 0; end
        else if (out_valid) begin
            if (out_tok == N_TOKENS - 1) begin out_tok <= 0; out_grp <= out_grp + 8'd1; end
            else out_tok <= out_tok + 16'd1;
        end
    end
endmodule
