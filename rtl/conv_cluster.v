`timescale 1ns/1ps

// ============================================================================
// conv_cluster: K weight-stationary 16x16 systolic arrays computing K output
// chunks of a 1x1 convolution at once, with shadow weight registers so that the
// next tile's weights stream in while the current tile computes. Tiles run back
// to back (no flush); one activation word per cycle feeds all K arrays.
//
// Arithmetic is identical to conv_engine_top + psum_buffer (48-bit psums,
// round-half-up >> FRAC_BITS, + bias, saturate to int16).
//
// Constraints: K*16 <= N_TOKENS - 34 (next-step shadow load must fit in a step after LOAD_START),
//              cfg_oc % K == 0.
// ============================================================================

// ---------------------------------------------------------------- PE with shadow weight
module pe_sw (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               load_wt,          // shift wt_in into the shadow chain
    input  wire signed [15:0] act_in,
    input  wire               flag_in,          // 1 on the first token of a tile: use + adopt the shadow weight
    input  wire signed [15:0] wt_in,
    input  wire signed [47:0] psum_in,
    output reg  signed [15:0] act_out,
    output reg                flag_out,
    output wire signed [15:0] wt_out,
    output reg  signed [47:0] psum_out
);
    reg signed [15:0] wt_reg, wt_sh;
    assign wt_out = wt_sh;
    wire signed [15:0] w_use = flag_in ? wt_sh : wt_reg;
    wire signed [47:0] mult_res = act_in * w_use;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_out <= 0; flag_out <= 0; psum_out <= 0; wt_reg <= 0; wt_sh <= 0;
        end else begin
            if (load_wt) wt_sh <= wt_in;
            if (flag_in) wt_reg <= wt_sh;
            act_out  <= act_in;
            flag_out <= flag_in;
            psum_out <= psum_in + mult_res;
        end
    end
endmodule

// ---------------------------------------------------------------- array
module systolic_array_sw #(parameter ROWS = 16, parameter COLS = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 load_wt,
    input  wire [ROWS*17-1:0]   act_in_row,     // {flag, act[15:0]} per row
    input  wire [COLS*16-1:0]   wt_in_col,
    output wire [COLS*48-1:0]   psum_out_col
);
    wire signed [15:0] act  [0:ROWS-1][0:COLS];
    wire               flg  [0:ROWS-1][0:COLS];
    wire signed [15:0] wt   [0:ROWS][0:COLS-1];
    wire signed [47:0] psum [0:ROWS][0:COLS-1];
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : bound_row
            assign act[r][0] = act_in_row[r*17 +: 16];
            assign flg[r][0] = act_in_row[r*17 + 16];
        end
        for (c = 0; c < COLS; c = c + 1) begin : bound_col
            assign wt[0][c]   = wt_in_col[c*16 +: 16];
            assign psum[0][c] = 48'd0;
            assign psum_out_col[c*48 +: 48] = psum[ROWS][c];
        end
        for (r = 0; r < ROWS; r = r + 1) begin : row
            for (c = 0; c < COLS; c = c + 1) begin : col
                pe_sw u_pe (.clk(clk), .rst_n(rst_n), .load_wt(load_wt),
                            .act_in(act[r][c]), .flag_in(flg[r][c]), .wt_in(wt[r][c]), .psum_in(psum[r][c]),
                            .act_out(act[r][c+1]), .flag_out(flg[r][c+1]), .wt_out(wt[r+1][c]), .psum_out(psum[r+1][c]));
            end
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- engine: skew -> array -> deskew, 31-cycle latency
module conv_engine_sw #(parameter ROWS = 16, parameter COLS = 16) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 load_wt,
    input  wire                 act_valid,
    input  wire                 act_flag,
    input  wire [ROWS*16-1:0]   flat_act_in,
    input  wire [COLS*16-1:0]   flat_wt_in,
    output wire [COLS*48-1:0]   flat_psum_out,
    output wire                 out_valid
);
    wire [ROWS*17-1:0] act_f, skewed_act;
    genvar r;
    generate for (r = 0; r < ROWS; r = r + 1) begin : pack
        assign act_f[r*17 +: 17] = {act_flag, flat_act_in[r*16 +: 16]};
    end endgenerate
    skew_buffer #(.ROWS(ROWS), .WIDTH(17)) u_skew (.clk(clk), .rst_n(rst_n), .flat_in(act_f), .skewed_out(skewed_act));
    wire [COLS*48-1:0] skewed_psum;
    systolic_array_sw #(.ROWS(ROWS), .COLS(COLS)) u_arr (.clk(clk), .rst_n(rst_n), .load_wt(load_wt),
                                                       .act_in_row(skewed_act), .wt_in_col(flat_wt_in), .psum_out_col(skewed_psum));
    deskew_buffer #(.COLS(COLS), .WIDTH(48)) u_deskew (.clk(clk), .rst_n(rst_n), .skewed_in(skewed_psum), .flat_out(flat_psum_out));
    localparam TOTAL_DELAY = ROWS + COLS - 1;
    reg [TOTAL_DELAY-1:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= 0;
        else valid_pipe <= {valid_pipe[TOTAL_DELAY-2:0], act_valid};
    end
    assign out_valid = valid_pipe[TOTAL_DELAY-1];
endmodule

// ---------------------------------------------------------------- psum accumulate across in-chunks (runtime chunk count)
module psum_buffer_dyn #(parameter COLS = 16, parameter N_TOKENS = 256, parameter FRAC_BITS = 11) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear_psum,
    input  wire [7:0]           chunks,
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
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) chunk_cnt <= 0;
        else if (clear_psum) chunk_cnt <= 0;
        else if (in_valid_d1 && rd_addr_d1 == N_TOKENS-1) chunk_cnt <= (chunk_cnt == chunks-1) ? 0 : chunk_cnt + 1;
    end
    wire [COLS*48-1:0] sum_out;
    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : adder
            wire signed [47:0] in_val  = psum_in_d1[c*48 +: 48];
            wire signed [47:0] ram_val = (chunk_cnt == 0) ? 48'd0 : ram_q_sync[c*48 +: 48];
            assign sum_out[c*48 +: 48] = in_val + ram_val;
        end
    endgenerate
    always @(posedge clk) begin
        if (in_valid_d1) ram[rd_addr_d1] <= sum_out;
    end
    generate
        for (c = 0; c < COLS; c = c + 1) begin : clip
            wire signed [47:0] val = sum_out[c*48 +: 48];
            wire signed [47:0] rounding_adder = (FRAC_BITS > 0) ? (48'sh1 << (FRAC_BITS - 1)) : 48'sh0;
            wire signed [47:0] shifted = (val + rounding_adder) >>> FRAC_BITS;
            wire signed [15:0] bias_val = flat_bias_in[c*16 +: 16];
            wire signed [47:0] final_sum = shifted + {{32{bias_val[15]}}, bias_val};
            assign flat_out_data[c*16 +: 16] = (final_sum > 32767) ? 16'd32767 : (final_sum < -32768) ? -16'd32768 : final_sum[15:0];
        end
    endgenerate
    assign out_valid = in_valid_d1 && (chunk_cnt == chunks-1);
endmodule

// ---------------------------------------------------------------- the cluster
module conv_cluster #(
    parameter K = 8,
    parameter N_TOKENS = 256,
    parameter FRAC_BITS = 11
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [7:0]           cfg_oc,          // output chunks (multiple of K)
    input  wire [7:0]           cfg_ic,          // input chunks
    input  wire                 cfg_ic_major,    // 0: tile = oc*IC + ic (M1/M3/P2); 1: tile = ic*OC + oc (P3)
    input  wire [31:0]          cfg_w_base,      // weight word address of tile 0
    input  wire [31:0]          cfg_b_base,      // bias word address of out-chunk 0
    output reg                  busy,
    output reg                  done,            // 1-cycle pulse

    output wire [15:0]          act_token,       // activation request: (token, ic) this cycle
    output wire [7:0]           act_ic,
    output wire                 act_fire,
    input  wire [255:0]         act_data,        // the word requested one cycle earlier

    output wire [31:0]          w_addr,          // weight BRAM word address; data one cycle later
    input  wire [255:0]         w_rdata,
    output wire [31:0]          b_addr,
    input  wire [255:0]         b_rdata,

    output wire                 out_valid,       // K output chunks of one token
    output wire [15:0]          out_token,
    output wire [7:0]           out_oc_base,
    output wire [K*256-1:0]     out_data
);
    localparam S_IDLE = 3'd0, S_BIAS = 3'd1, S_LOAD0 = 3'd2, S_CALC = 3'd3, S_DRAIN = 3'd4;
    localparam LOAD_START = 32;                  // next-step shadow load starts after the swap wavefront has crossed all 16x16 PEs (row+col <= 30)
    localparam KB = (K <= 1) ? 1 : $clog2(K);

    reg [2:0]  state;
    wire       out_done;                         // every output group delivered
    reg [7:0]  oc_n, ic_n, groups;               // layer config latched at start
    reg        ic_major;
    reg [31:0] w_base, b_base;

    // ---- bias ram (one word per output chunk), filled at layer start
    reg [255:0] bias_ram [0:63];
    reg [7:0]   bias_cnt;
    reg         bias_wr_d1;
    reg [7:0]   bias_idx_d1;
    assign b_addr = b_base + bias_cnt;

    // ---- compute stream
    reg [15:0] tok;
    reg [7:0]  ic, grp;
    reg        act_fire_r;
    assign act_token = tok;
    assign act_ic    = ic;
    assign act_fire  = act_fire_r;
    reg        act_valid_d1, act_flag_d1;        // aligned with act_data (BRAM latency 1)

    wire last_tok  = (tok == N_TOKENS - 1);
    wire last_ic   = (ic == ic_n - 1);
    wire last_grp  = (grp == groups - 1);
    wire last_step = last_ic && last_grp;

    // ---- weight loader: shadow-loads K tiles of step (ld_grp, ld_ic)
    reg        ld_active;
    reg [7:0]  ld_k, ld_j;                       // array index, row counter 0..15
    reg [7:0]  ld_grp, ld_ic;
    wire [15:0] ld_oc   = ld_grp * K + ld_k;
    wire [15:0] ld_tile = ic_major ? (ld_ic * oc_n + ld_oc) : (ld_oc * ic_n + ld_ic);
    assign w_addr = w_base + ld_tile * 16 + (15 - ld_j);
    reg        ld_active_d1;
    reg [7:0]  ld_k_d1;
    wire       ld_last = ld_active && (ld_j == 15) && (ld_k == K - 1);

    // next step after (grp, ic)
    wire [7:0] nx_ic  = last_ic ? 8'd0 : ic + 1;
    wire [7:0] nx_grp = last_ic ? grp + 1 : grp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 0; done <= 0;
            oc_n <= 0; ic_n <= 0; groups <= 0; ic_major <= 0; w_base <= 0; b_base <= 0;
            bias_cnt <= 0; bias_wr_d1 <= 0; bias_idx_d1 <= 0;
            tok <= 0; ic <= 0; grp <= 0; act_fire_r <= 0; act_valid_d1 <= 0; act_flag_d1 <= 0;
            ld_active <= 0; ld_k <= 0; ld_j <= 0; ld_grp <= 0; ld_ic <= 0; ld_active_d1 <= 0; ld_k_d1 <= 0;
        end else begin
            done <= 0;
            act_valid_d1 <= act_fire_r;
            act_flag_d1  <= act_fire_r && (tok == 0);
            ld_active_d1 <= ld_active;
            ld_k_d1      <= ld_k;
            bias_wr_d1   <= (state == S_BIAS);
            bias_idx_d1  <= bias_cnt;
            if (bias_wr_d1) bias_ram[bias_idx_d1] <= b_rdata;

            // weight loader runs independently of the main state
            if (ld_active) begin
                if (ld_j == 15) begin
                    ld_j <= 0;
                    if (ld_k == K - 1) begin ld_k <= 0; ld_active <= 0; end
                    else ld_k <= ld_k + 1;
                end else ld_j <= ld_j + 1;
            end

            case (state)
                S_IDLE: begin
                    act_fire_r <= 0;
                    if (start) begin
                        oc_n <= cfg_oc; ic_n <= cfg_ic; groups <= cfg_oc / K; ic_major <= cfg_ic_major;
                        w_base <= cfg_w_base; b_base <= cfg_b_base;
                        busy <= 1; bias_cnt <= 0; state <= S_BIAS;
                    end
                end
                S_BIAS: begin                                   // read oc_n bias words
                    if (bias_cnt == oc_n - 1) begin
                        bias_cnt <= 0;
                        ld_active <= 1; ld_k <= 0; ld_j <= 0; ld_grp <= 0; ld_ic <= 0;   // initial shadow load, step 0
                        state <= S_LOAD0;
                    end else bias_cnt <= bias_cnt + 1;
                end
                S_LOAD0: begin
                    if (ld_last) begin                          // shadow of step 0 complete next cycle
                        tok <= 0; ic <= 0; grp <= 0; act_fire_r <= 1; state <= S_CALC;
                    end
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
                    end else tok <= tok + 1;
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
            conv_engine_sw u_eng (
                .clk(clk), .rst_n(rst_n), .load_wt(ld_active_d1 && (ld_k_d1 == k)),
                .act_valid(act_valid_d1), .act_flag(act_flag_d1), .flat_act_in(act_data), .flat_wt_in(w_rdata),
                .flat_psum_out(psum_flat[k*768 +: 768]), .out_valid(ov)
            );
            if (k == 0) begin : v0
                assign eng_valid = ov;
            end
            wire [255:0] bias_word = bias_ram[out_grp * K + k];
            psum_buffer_dyn #(.COLS(16), .N_TOKENS(N_TOKENS), .FRAC_BITS(FRAC_BITS)) u_pb (
                .clk(clk), .rst_n(rst_n), .clear_psum(clear), .chunks(ic_n), .in_valid(eng_valid),
                .flat_psum_in(psum_flat[k*768 +: 768]), .flat_bias_in(bias_word),
                .out_valid(pb_valid[k]), .flat_out_data(out_data[k*256 +: 256])
            );
        end
    endgenerate

    assign out_valid   = pb_valid[0];
    assign out_token   = out_tok;
    assign out_oc_base = out_grp * K;
    assign out_done = (out_grp == groups);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin out_tok <= 0; out_grp <= 0; end
        else if (clear) begin out_tok <= 0; out_grp <= 0; end
        else if (out_valid) begin
            if (out_tok == N_TOKENS - 1) begin out_tok <= 0; out_grp <= out_grp + 1; end
            else out_tok <= out_tok + 1;
        end
    end
endmodule

// ---------------------------------------------------------------- SRAM with a K-word write port and a 1-word read port
// word address = token*STRIDE + chunk; a write stores K consecutive chunks (waddr multiple of K).
module banked_sram #(
    parameter K = 8,
    parameter DEPTH = 4096,          // total 256-bit words
    parameter AW = 12,
    parameter ULTRA = 0
)(
    input  wire             clk,
    input  wire             we,
    input  wire [AW-1:0]    waddr,
    input  wire [K*256-1:0] wdata,
    input  wire [AW-1:0]    raddr,
    output wire [255:0]     rdata,
    input  wire [AW-1:0]    raddr2,          // second independent read port
    output wire [255:0]     rdata2
);
    localparam KB = $clog2(K);
    localparam RD = DEPTH / K;
    reg [KB-1:0] sel_d1, sel2_d1;
    wire [K*256-1:0] rq, rq2;
    genvar k;
    generate
        for (k = 0; k < K; k = k + 1) begin : bank
            if (ULTRA) begin : u
                (* ram_style = "ultra" *) reg [255:0] mem [0:RD-1];
                reg [255:0] q, q2;
                always @(posedge clk) begin
                    if (we) mem[waddr >> KB] <= wdata[k*256 +: 256];
                    q  <= mem[raddr >> KB];
                    q2 <= mem[raddr2 >> KB];
                end
                assign rq[k*256 +: 256]  = q;
                assign rq2[k*256 +: 256] = q2;
            end else begin : b
                (* ram_style = "block" *) reg [255:0] mem [0:RD-1];
                reg [255:0] q, q2;
                always @(posedge clk) begin
                    if (we) mem[waddr >> KB] <= wdata[k*256 +: 256];
                    q  <= mem[raddr >> KB];
                    q2 <= mem[raddr2 >> KB];
                end
                assign rq[k*256 +: 256]  = q;
                assign rq2[k*256 +: 256] = q2;
            end
        end
    endgenerate
    always @(posedge clk) begin sel_d1 <= raddr[KB-1:0]; sel2_d1 <= raddr2[KB-1:0]; end
    assign rdata  = rq[sel_d1*256 +: 256];
    assign rdata2 = rq2[sel2_d1*256 +: 256];
endmodule
