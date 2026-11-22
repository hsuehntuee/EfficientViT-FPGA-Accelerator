`timescale 1ns/1ps

// ============================================================================
// accel_core (v6, INT8): EfficientViT-B1 Stage 4 (backbone.stages[3]), up to four
// blocks back to back from one start. The arithmetic is int8/evit_int8.py.
//
// Per block (the passes of v5):
//   PRM     read the block's per-channel input shifts and attention parameters
//   M1      QKV 1x1 (256->768)      conv_cluster8, input = int8(x)                -> sram_q/k/v
//           MS branch (5x5 DW + group 1x1) trails M1 group by group               -> sram_q2/k2/v2
//   ATT_Q   16 heads on q/k/v                                                      -> sram_h1
//   ATT_MS  same on q2/k2/v2 (starts when the MS branch is far enough ahead)       -> sram_h2
//   M3      proj 1x1 (512->256)     conv_cluster8, int16 out                       -> sram_proj   (mid = sat(x + proj) at read time)
//   P2      expand 1x1 (256->1024) + hswish, input = int8(mid)                     -> sram_exp
//           DW pass (5x5 depthwise + hswish) trails P2 group by group              -> sram_dw
//   P3      point 1x1 (1024->256)   conv_cluster8, int16 out; starts while the DW pass still runs  -> sram_pw_raw
//   P4      x <= sat(sat(x + proj) + pw), written back into the x memory
//
// What differs from v5:
//   * x, proj and pw stay int16 (Q10.5); every other tensor is int8 with a per-channel scale, so those
//     memories are 128 bits wide (16 channels x 8 bits);
//   * the cluster produces 16 chunks per step instead of 8 (two per array);
//   * x and mid are converted to int8 on the way into the cluster (per-channel shift 0..2);
//   * every convolution output is requantised with per-channel parameters (512 bits per 16-channel chunk);
//   * hardswish is a table inside the cluster (expand) and behind the depthwise engine.
// ============================================================================
module accel_core #(
    parameter N_TOKENS = 256,
    parameter IMG_SIZE = 16,
    parameter C_IN_CHUNKS = 16,
    parameter C_OUT_CHUNKS = 64,
    parameter CONV_PAR = 8,          // arrays K (2K*8 + W_LAT + 35 < N_TOKENS + ACT_LAT)
    parameter WBIG_LAT = 6           // read latency of the big weight store
)(
    input  wire                 clk,
    input  wire                 rst_n,
    // control
    input  wire                 reg_start,       // level: 1 starts (from idle), 0 acknowledges done
    input  wire [1:0]           cfg_first,       // first block to run
    input  wire [1:0]           cfg_nm1,         // number of blocks - 1
    output reg                  reg_done,
    output wire                 mem_idle,        // host may access the x memory and the weight store
    output reg  [31:0]          reg_cycle_cnt,   // cycles from start to done
    output reg  [127:0]         blk_cycles,      // reg_cycle_cnt at the end of the 1st .. 4th block of this run
    output wire [1:0]           cur_blk,

    // weight store, read side
    output wire [17:0]          wbig_raddr,      input wire [255:0] wbig_rdata,     // latency WBIG_LAT
    output wire [9:0]           clb_raddr,       input wire [511:0] clb_rdata,      // latency 1, 512-bit entries
    output reg  [13:0]          dww_raddr,       input wire [127:0] dww_rdata,      // latency 3; the other small memories: latency 1
    output reg  [8:0]           dwb_raddr,       input wire [511:0] dwb_rdata,
    output reg  [11:0]          mspww_raddr,     input wire [127:0] mspww_rdata,
    output reg  [7:0]           mspwb_raddr,     input wire [511:0] mspwb_rdata,

    // host access to the x memory (only while mem_idle), read latency 1
    input  wire                 host_x_we,
    input  wire [11:0]          host_x_addr,
    input  wire [255:0]         host_x_wdata,
    output wire [255:0]         host_x_rdata
);
    localparam K         = CONV_PAR;
    localparam KC        = 2 * CONV_PAR;         // output chunks per cluster step
    localparam ACT_WORDS = N_TOKENS * C_IN_CHUNKS;
    localparam AW        = $clog2(ACT_WORDS);
    localparam EXP_WORDS = N_TOKENS * C_OUT_CHUNKS;
    localparam EAW       = $clog2(EXP_WORDS);
    localparam DW_WORDS  = N_TOKENS * C_OUT_CHUNKS;
    localparam DAW       = $clog2(DW_WORDS);
    localparam N_HEADS   = 16;
    localparam ACT_LAT   = 5;                    // (token, ic) -> address register -> SRAM -> stage A -> int8 conversion -> layer mux -> cluster
    localparam P4_LAT    = 4;                    // address -> SRAM -> stage -> mid -> y
    localparam N_PRM     = 34;                   // parameter entries read at the start of a block

    // ------------------------------------------------------------------ main FSM
    localparam S_IDLE = 5'd0, S_PRM = 5'd1, S_M1_START = 5'd2, S_M1_RUN = 5'd3,
               S_ATTQ_START = 5'd4, S_ATTQ_RUN = 5'd5, S_WAIT_MS = 5'd6, S_ATTM_START = 5'd7, S_ATTM_RUN = 5'd8,
               S_M3_START = 5'd9, S_M3_RUN = 5'd10, S_P2_START = 5'd11, S_P2_RUN = 5'd12, S_WAIT_DW = 5'd13,
               S_P3_START = 5'd14, S_P3_RUN = 5'd15, S_P4_RES_ADD = 5'd16, S_P4_FLUSH = 5'd17, S_BLK_NEXT = 5'd18, S_DONE = 5'd19,
               S_WAIT_MS2 = 5'd20, S_DW_HEAD = 5'd21;
    // The two background branches are slower per chunk than the pass that reads their output (about 330 cycles a chunk
    // against 264 per attention head and 256 per point-wise input chunk), so the reader may start once the branch is far
    // enough ahead that it is never caught: MS chunk 32 + h must be complete before head h, which holds from MS_HEAD = 38
    // chunks on; depthwise chunk ic before point-wise step ic, which holds from DW_HEAD = 24 chunks on.
    localparam MS_HEAD = 38, DW_HEAD = 24;
    localparam L_M1 = 2'd0, L_M3 = 2'd1, L_P2 = 2'd2, L_P3 = 2'd3;

    reg [4:0]  state, next_state;
    reg [1:0]  layer;
    reg [15:0] addr_cnt;
    reg [3:0]  flush_cnt;
    reg [5:0]  prm_cnt;
    reg [1:0]  blk, blk_left, run_idx;
    wire cl_busy, cl_done, att_done, ms_done, dw_done, ms_ahead, dw_ahead;
    wire cl_layer_start = (state == S_M1_START || state == S_M3_START || state == S_P2_START || state == S_P3_START);
    assign mem_idle = (state == S_IDLE) || (state == S_DONE);
    assign cur_blk  = blk;

    always @(posedge clk) begin
        if (!rst_n) state <= S_IDLE; else state <= next_state;
    end
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:        if (reg_start) next_state = S_PRM;
            S_PRM:         if (prm_cnt == N_PRM + 5) next_state = S_M1_START;
            S_M1_START:    next_state = S_M1_RUN;
            S_M1_RUN:      if (cl_done) next_state = S_ATTQ_START;
            S_ATTQ_START:  next_state = S_ATTQ_RUN;
            S_ATTQ_RUN:    if (att_done) next_state = S_WAIT_MS;
            S_WAIT_MS:     if (ms_done || ms_ahead) next_state = S_ATTM_START;
            S_ATTM_START:  next_state = S_ATTM_RUN;
            S_ATTM_RUN:    if (att_done) next_state = S_WAIT_MS2;
            S_WAIT_MS2:    if (ms_done) next_state = S_M3_START;
            S_M3_START:    next_state = S_M3_RUN;
            S_M3_RUN:      if (cl_done) next_state = S_P2_START;
            S_P2_START:    next_state = S_P2_RUN;
            S_P2_RUN:      if (cl_done) next_state = S_DW_HEAD;
            S_DW_HEAD:     if (dw_done || dw_ahead) next_state = S_P3_START;
            S_P3_START:    next_state = S_P3_RUN;
            S_P3_RUN:      if (cl_done) next_state = S_WAIT_DW;
            S_WAIT_DW:     if (dw_done) next_state = S_P4_RES_ADD;
            S_P4_RES_ADD:  if (addr_cnt == ACT_WORDS-1) next_state = S_P4_FLUSH;
            S_P4_FLUSH:    if (flush_cnt == P4_LAT + 2) next_state = S_BLK_NEXT;
            S_BLK_NEXT:    next_state = (blk_left == 2'd0) ? S_DONE : S_PRM;
            S_DONE:        if (!reg_start) next_state = S_IDLE;
            default:       next_state = S_IDLE;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            layer <= 0; addr_cnt <= 0; flush_cnt <= 0; prm_cnt <= 0; reg_cycle_cnt <= 0; blk_cycles <= 0;
            blk <= 0; blk_left <= 0; run_idx <= 0; reg_done <= 0;
        end else begin
            reg_done <= (state == S_DONE);
            if (state == S_IDLE) begin
                if (reg_start) begin reg_cycle_cnt <= 0; blk <= cfg_first; blk_left <= cfg_nm1; run_idx <= 0; end
            end else if (state != S_DONE) reg_cycle_cnt <= reg_cycle_cnt + 32'd1;
            if (state == S_P4_RES_ADD) addr_cnt <= addr_cnt + 16'd1; else addr_cnt <= 0;
            if (state == S_P4_FLUSH) flush_cnt <= flush_cnt + 4'd1; else flush_cnt <= 0;
            if (state == S_PRM) prm_cnt <= prm_cnt + 6'd1; else prm_cnt <= 0;
            if (state == S_BLK_NEXT) begin
                blk_cycles[run_idx*32 +: 32] <= reg_cycle_cnt;
                run_idx <= run_idx + 2'd1;
                if (blk_left != 2'd0) begin blk <= blk + 2'd1; blk_left <= blk_left - 2'd1; end
            end
            case (state)
                S_M1_START: layer <= L_M1;
                S_M3_START: layer <= L_M3;
                S_P2_START: layer <= L_P2;
                S_P3_START: layer <= L_P3;
                default: ;
            endcase
        end
    end

    // weight store bases of the current block: 7 UltraRAM pages (28,672 words) of big weights,
    // 256 entries of cluster parameters, 4096 words of depthwise weights, 128 entries of depthwise
    // parameters, 1024 words of MS 1x1 weights, 64 entries of MS 1x1 parameters.
    wire [5:0]  blk_page  = {2'd0, blk, 2'd0} + {3'd0, blk, 1'd0} + {4'd0, blk};      // blk * 7
    wire [17:0] blk_wbase = {blk_page, 12'd0};
    wire [9:0]  blk_bbase = {blk, 8'd0};

    // ------------------------------------------------------------------ block parameters (S_PRM)
    // entry 144: x -> int8 shifts (16 chunks x 16 lanes x 2 bits), 145: the same for mid,
    // 160 + src*16 + head: attention {shift[15][3:0] .. shift[0][3:0], f1[3:0]}
    reg [511:0] clb_q;
    reg [511:0] x8_sh_r, mid8_sh_r;
    reg [67:0]  att_prm [0:31];
    // The first read waits two cycles: the weight store hands its read ports from the host to the accelerator one cycle after idle ends.
    wire [5:0]  prm_idx  = prm_cnt - 6'd2;
    wire        prm_rd   = (state == S_PRM) && (prm_cnt >= 2) && (prm_idx < N_PRM);
    wire [9:0]  prm_addr = blk_bbase + ((prm_idx < 2) ? (10'd144 + {4'd0, prm_idx}) : (10'd158 + {4'd0, prm_idx}));
    reg         prm_v1, prm_v2;
    reg [5:0]   prm_i1, prm_i2;
    wire [9:0]  cl_b_addr;
    assign clb_raddr = (state == S_PRM) ? prm_addr : cl_b_addr;
    always @(posedge clk) begin
        clb_q  <= clb_rdata;
        prm_v1 <= prm_rd; prm_v2 <= prm_v1;
        prm_i1 <= prm_idx; prm_i2 <= prm_i1;
        if (prm_v2) begin
            if (prm_i2 == 6'd0)      x8_sh_r   <= clb_q;
            else if (prm_i2 == 6'd1) mid8_sh_r <= clb_q;
            else                     att_prm[prm_i2[4:0] - 5'd2] <= clb_q[67:0];
        end
    end

    // ------------------------------------------------------------------ conv cluster
    wire [15:0] cl_act_token;  wire [7:0] cl_act_ic;  wire cl_act_fire;
    wire        cl_out_valid;  wire [15:0] cl_out_token;  wire [7:0] cl_out_oc_base;
    wire [KC*256-1:0] cl_out_data;
    wire [KC*128-1:0] cl_out8;                    // the int8 results: low byte of every lane
    genvar oc, ol;
    generate
        for (oc = 0; oc < KC; oc = oc + 1) begin : o8c
            for (ol = 0; ol < 16; ol = ol + 1) begin : o8l
                assign cl_out8[oc*128 + ol*8 +: 8] = cl_out_data[oc*256 + ol*16 +: 8];
            end
        end
    endgenerate
    reg         cl_start;
    reg [7:0]   cfg_oc, cfg_ic;
    reg         cfg_ic_major;
    reg [1:0]   cfg_mode;
    reg [17:0]  cfg_w_base;
    reg [9:0]   cfg_b_base;
    always @(posedge clk) begin
        if (!rst_n) begin cl_start <= 0; cfg_oc <= 0; cfg_ic <= 0; cfg_ic_major <= 0; cfg_mode <= 0; cfg_w_base <= 0; cfg_b_base <= 0; end
        else begin
            cl_start <= cl_layer_start;
            case (state)   // big weights of a block: QKV at 0, proj at 6144, expand at 10240, point at 18432; parameter entries: QKV 0, proj 48, expand 64, point 128
                S_M1_START: begin cfg_oc <= 48; cfg_ic <= 16; cfg_ic_major <= 0; cfg_mode <= 2'd0; cfg_w_base <= blk_wbase;            cfg_b_base <= blk_bbase;          end
                S_M3_START: begin cfg_oc <= 16; cfg_ic <= 32; cfg_ic_major <= 0; cfg_mode <= 2'd1; cfg_w_base <= blk_wbase + 18'd6144;  cfg_b_base <= blk_bbase + 10'd48;  end
                S_P2_START: begin cfg_oc <= 64; cfg_ic <= 16; cfg_ic_major <= 0; cfg_mode <= 2'd2; cfg_w_base <= blk_wbase + 18'd10240; cfg_b_base <= blk_bbase + 10'd64;  end
                S_P3_START: begin cfg_oc <= 16; cfg_ic <= 64; cfg_ic_major <= 1; cfg_mode <= 2'd1; cfg_w_base <= blk_wbase + 18'd18432; cfg_b_base <= blk_bbase + 10'd128; end
                default: ;
            endcase
        end
    end

    wire [255:0] sram_x_rdata_A, sram_x_rdata_B, sram_proj_rdata_A, sram_proj_rdata_B, sram_pw_raw_rdata;
    wire [127:0] sram_h1_rdata, sram_h2_rdata, sram_dw_rdata;

    // activation path into the cluster:
    // address registers -> SRAM outputs -> stage A -> int8 conversion of x and of (x + proj) -> layer mux
    reg  [AW-1:0]  xa_addr, projA_raddr, h_raddr;        // registered addresses (loaded in the x memory section below)
    reg  [DAW-1:0] dw_raddr;
    reg            h_sel_d1, h_sel_d2, xb_host;
    reg  [3:0]     ic_d1, ic_d2;
    reg  [31:0]    x8_sh_w, mid8_sh_w;
    reg  [255:0]   xA_r, projA_r;
    reg  [127:0]   h_r, dwA_r, h_r2, dwA_r2, x8_r, mid8_r, cl_act_data;
    wire [255:0]   mid_A;
    wire [127:0]   x8_w, mid8_w;
    adder_array   u_add_mid_A (.a(xA_r), .b(projA_r), .c(mid_A));
    to_int8_array u_x8   (.x(xA_r),  .sh(x8_sh_w),   .y(x8_w));
    to_int8_array u_mid8 (.x(mid_A), .sh(mid8_sh_w), .y(mid8_w));
    always @(posedge clk) begin
        ic_d1     <= cl_act_ic[3:0];
        ic_d2     <= ic_d1;
        x8_sh_w   <= x8_sh_r[ic_d2*32 +: 32];
        mid8_sh_w <= mid8_sh_r[ic_d2*32 +: 32];
        xA_r      <= sram_x_rdata_A;
        projA_r   <= sram_proj_rdata_A;
        h_r       <= h_sel_d2 ? sram_h1_rdata : sram_h2_rdata;
        dwA_r     <= sram_dw_rdata;
        x8_r      <= x8_w;
        mid8_r    <= mid8_w;
        h_r2      <= h_r;
        dwA_r2    <= dwA_r;
        cl_act_data <= (layer == L_M1) ? x8_r : (layer == L_M3) ? h_r2 : (layer == L_P2) ? mid8_r : dwA_r2;
    end

    // data of the small weight memories, registered once more here
    reg [127:0] dww_q, mspww_q;
    reg [511:0] dwb_q, mspwb_q;
    always @(posedge clk) begin
        dww_q <= dww_rdata; dwb_q <= dwb_rdata; mspww_q <= mspww_rdata; mspwb_q <= mspwb_rdata;
    end

    conv_cluster8 #(.K(K), .N_TOKENS(N_TOKENS), .ACT_LAT(ACT_LAT), .W_LAT(WBIG_LAT), .B_LAT(2)) u_cluster (
        .clk(clk), .rst_n(rst_n), .start(cl_start),
        .cfg_oc(cfg_oc), .cfg_ic(cfg_ic), .cfg_ic_major(cfg_ic_major), .cfg_mode(cfg_mode), .cfg_w_base(cfg_w_base), .cfg_b_base(cfg_b_base),
        .busy(cl_busy), .done(cl_done),
        .act_token(cl_act_token), .act_ic(cl_act_ic), .act_fire(cl_act_fire), .act_data(cl_act_data),
        .w_addr(wbig_raddr), .w_rdata(wbig_rdata), .b_addr(cl_b_addr), .b_rdata(clb_q),
        .out_valid(cl_out_valid), .out_token(cl_out_token), .out_oc_base(cl_out_oc_base), .out_data(cl_out_data)
    );
    // groups completed by the cluster in the current layer. Cleared in the *_START state (the edge on
    // which the MS FSM leaves MS_IDLE), never later: a stale count starts MS chunk 0 early (2026-10-10 board bug).
    reg [7:0] cl_groups_done;
    always @(posedge clk) begin
        if (!rst_n) cl_groups_done <= 0;
        else if (cl_layer_start) cl_groups_done <= 0;
        else if (cl_out_valid && cl_out_token == N_TOKENS-1) cl_groups_done <= cl_groups_done + 8'd1;
    end

    // ------------------------------------------------------------------ x memory (block input, replaced by the block output in P4)
    // port A: cluster reads (M1, P2) and the P4 write-back; port B: P4 reads and the host.
    // Every address into an activation memory on the cluster side is registered here (first stage of ACT_LAT).
    // xa_addr carries the P4 write address when the word in y_r is due, else the cluster read address.
    // xb_host follows mem_idle by one cycle: the P4 reads and the first host access are far from that edge.
    wire [AW-1:0] x_raddr_A = (cl_act_token * C_IN_CHUNKS) + cl_act_ic;
    reg  [P4_LAT:1]   p4_v_sr;
    reg  [AW-1:0]     p4_a_sr [1:P4_LAT];
    reg  [255:0]      y_r;
    wire              p4_rd = (state == S_P4_RES_ADD);
    always @(posedge clk) begin
        xa_addr     <= p4_v_sr[P4_LAT-1] ? p4_a_sr[P4_LAT-1] : x_raddr_A;
        projA_raddr <= x_raddr_A;
        h_raddr     <= (cl_act_token * 16) + (cl_act_ic % 16);
        dw_raddr    <= (cl_act_ic * N_TOKENS) + cl_act_token;
        h_sel_d1    <= (cl_act_ic < 16);
        h_sel_d2    <= h_sel_d1;
        xb_host     <= mem_idle;
    end
    tdp_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_x (
        .clk(clk),
        .we_a(p4_v_sr[P4_LAT]), .addr_a(xa_addr), .din_a(y_r), .dout_a(sram_x_rdata_A),
        .we_b(host_x_we && xb_host), .addr_b(xb_host ? host_x_addr : addr_cnt[AW-1:0]), .din_b(host_x_wdata), .dout_b(sram_x_rdata_B)
    );
    assign host_x_rdata = sram_x_rdata_B;

    // attention sequencer signals (defined below) that address q/k/v
    wire [AW-1:0] att_q_raddr, att_kv_raddr;
    // MS branch signals
    reg  [15:0] ms_chunk, ms_act_cnt, ms_wt_cnt, ms_out_token;
    wire [AW-1:0] ms_raddr = (ms_act_cnt * 16) + (ms_chunk % 16);

    wire [AW-1:0] cl_out_addr16 = (cl_out_token * 16);
    wire m1_we = cl_out_valid && (layer == L_M1);
    wire [127:0] q_att, k_att, v_att, q_ms, k_ms, v_ms;                 // port 1: attention, port 2: MS branch
    banked_sram #(.K(KC), .W(128), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_q (.clk(clk), .we(m1_we && cl_out_oc_base == 8'd0),
        .waddr(cl_out_addr16), .wdata(cl_out8), .raddr(att_q_raddr), .rdata(q_att), .raddr2(ms_raddr), .rdata2(q_ms));
    banked_sram #(.K(KC), .W(128), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_k (.clk(clk), .we(m1_we && cl_out_oc_base == 8'd16),
        .waddr(cl_out_addr16), .wdata(cl_out8), .raddr(att_kv_raddr), .rdata(k_att), .raddr2(ms_raddr), .rdata2(k_ms));
    banked_sram #(.K(KC), .W(128), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_v (.clk(clk), .we(m1_we && cl_out_oc_base == 8'd32),
        .waddr(cl_out_addr16), .wdata(cl_out8), .raddr(att_kv_raddr), .rdata(v_att), .raddr2(ms_raddr), .rdata2(v_ms));

    wire ms_pw_buf_valid;  wire [127:0] ms_pw_buf_data;
    wire [AW-1:0] ms_out_addr = (ms_out_token * 16) + (ms_chunk % 16);
    wire [127:0] q2_att, k2_att, v2_att;
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(ACT_WORDS), .ADDR_WIDTH(AW), .WREG(1)) inst_sram_q2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk < 16),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .raddr(att_q_raddr), .rdata(q2_att));
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(ACT_WORDS), .ADDR_WIDTH(AW), .WREG(1)) inst_sram_k2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk >= 16 && ms_chunk < 32),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .raddr(att_kv_raddr), .rdata(k2_att));
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(ACT_WORDS), .ADDR_WIDTH(AW), .WREG(1)) inst_sram_v2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk >= 32),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .raddr(att_kv_raddr), .rdata(v2_att));

    // ------------------------------------------------------------------ attention sequencer
    // step s = 0..16, ATT_STEP cycles each. With c = att_cyc:
    //   c = 0..255  address k/v of head s (s < 16) and q of head s-1 (s >= 1); data and valid one cycle later
    //   k/v token c is in the accumulator from c + 5 (last one: 260)
    //   q token c reads the holding registers at c + 3 (last one: 258)
    //   c = ATT_STEP-1 = 263: ctx_swap (holding <= accumulator of head s, accumulator <= 0), s < 16
    localparam ATT_STEP = N_TOKENS + 8;
    reg        att_run, att_src;              // att_src: 0 = q/k/v (h1), 1 = q2/k2/v2 (h2)
    reg [4:0]  att_step;
    reg [15:0] att_cyc;
    reg        ctx_clr, ctx_swap;
    reg        kv_valid_d1, q_valid_d1;
    reg [15:0] att_out_tok;
    reg [4:0]  att_out_head;
    wire       att_stream = att_run && (att_cyc < N_TOKENS);
    wire       att_kv_on  = att_stream && (att_step < N_HEADS);
    wire       att_q_on   = att_stream && (att_step >= 1);
    assign att_kv_raddr = (att_cyc * 16) + att_step[3:0];
    assign att_q_raddr  = (att_cyc * 16) + (att_step - 1);
    wire [127:0] mla_out_data;
    wire         mla_out_valid;
    assign att_done = att_run && (att_out_head == N_HEADS);

    always @(posedge clk) begin
        if (!rst_n) begin
            att_run <= 0; att_src <= 0; att_step <= 0; att_cyc <= 0; ctx_clr <= 0; ctx_swap <= 0;
            kv_valid_d1 <= 0; q_valid_d1 <= 0; att_out_tok <= 0; att_out_head <= 0;
        end else begin
            kv_valid_d1 <= att_kv_on;
            q_valid_d1  <= att_q_on;
            ctx_clr     <= 0;
            ctx_swap    <= 0;
            if (state == S_ATTQ_START || state == S_ATTM_START) begin
                att_run <= 1; att_src <= (state == S_ATTM_START); att_step <= 0; att_cyc <= 0;
                att_out_tok <= 0; att_out_head <= 0;
                ctx_clr <= 1;                                       // streams start next cycle (cyc 0)
            end else if (att_run) begin
                if (att_cyc == ATT_STEP - 2 && att_step < N_HEADS) ctx_swap <= 1;
                if (att_cyc == ATT_STEP - 1) begin
                    if (att_step < N_HEADS) begin att_cyc <= 0; att_step <= att_step + 5'd1; end   // step 16 (q of head 15) then holds until drained
                end else att_cyc <= att_cyc + 16'd1;
                if (mla_out_valid) begin
                    if (att_out_tok == N_TOKENS-1) begin att_out_tok <= 0; att_out_head <= att_out_head + 5'd1; end
                    else att_out_tok <= att_out_tok + 16'd1;
                end
                if (att_done) att_run <= 0;
            end
        end
    end
    wire [127:0] att_q_in = att_src ? q2_att : q_att;
    wire [127:0] att_k_in = att_src ? k2_att : k_att;
    wire [127:0] att_v_in = att_src ? v2_att : v_att;
    // parameters: the "1" of the head whose k / v stream, the output shifts of the head whose results leave
    reg [10:0] att_one_q;
    reg [63:0] att_sh_q;
    wire [67:0] att_prm_kv  = att_prm[{att_src, att_step[3:0]}];
    wire [67:0] att_prm_out = att_prm[{att_src, att_out_head[3:0]}];
    always @(posedge clk) begin
        att_one_q <= 11'd1 << att_prm_kv[3:0];
        att_sh_q  <= att_prm_out[67:4];
    end
    litemla_core8 #(.DIM(16), .N_TOKENS(N_TOKENS), .QBITS(12)) u_litemla (
        .clk(clk), .rst_n(rst_n),
        .ctx_clr(ctx_clr), .ctx_swap(ctx_swap), .kv_valid(kv_valid_d1), .k_in(att_k_in), .v_in(att_v_in), .one_in(att_one_q),
        .q_valid(q_valid_d1), .q_in(att_q_in), .sh_in(att_sh_q),
        .out_data(mla_out_data), .out_valid(mla_out_valid)
    );
    wire [AW-1:0] h_waddr = (att_out_tok * 16) + att_out_head[3:0];
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(ACT_WORDS), .ADDR_WIDTH(AW), .WREG(1)) inst_sram_h1 (.clk(clk), .we(mla_out_valid && !att_src),
        .waddr(h_waddr), .wdata(mla_out_data), .raddr(h_raddr), .rdata(sram_h1_rdata));
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(ACT_WORDS), .ADDR_WIDTH(AW), .WREG(1)) inst_sram_h2 (.clk(clk), .we(mla_out_valid && att_src),
        .waddr(h_waddr), .wdata(mla_out_data), .raddr(h_raddr), .rdata(sram_h2_rdata));

    // ------------------------------------------------------------------ MS branch (background, trails M1 group by group)
    localparam MS_IDLE = 3'd0, MS_WAIT = 3'd1, MS_LOAD = 3'd2, MS_CALC = 3'd3, MS_FLUSH = 3'd4, MS_DONE = 3'd5;
    reg [2:0] ms_state;
    wire ms_chunk_ready = (layer == L_M1) && (cl_groups_done > (ms_chunk / KC));    // its q/k/v group is in SRAM
    assign ms_done  = (ms_state == MS_DONE);
    assign ms_ahead = (ms_chunk >= MS_HEAD);
    always @(posedge clk) begin
        if (!rst_n) begin ms_state <= MS_IDLE; ms_chunk <= 0; ms_act_cnt <= 0; ms_wt_cnt <= 0; end
        else begin
            case (ms_state)
                MS_IDLE:  begin ms_chunk <= 0; ms_act_cnt <= 0; ms_wt_cnt <= 0; if (state == S_M1_START) ms_state <= MS_WAIT; end
                MS_WAIT:  if (ms_chunk_ready) ms_state <= MS_LOAD;
                MS_LOAD:  begin ms_act_cnt <= 0; if (ms_wt_cnt == 24) begin ms_wt_cnt <= 0; ms_state <= MS_CALC; end else ms_wt_cnt <= ms_wt_cnt + 16'd1; end
                MS_CALC:  begin if (ms_act_cnt == N_TOKENS-1) ms_state <= MS_FLUSH; else ms_act_cnt <= ms_act_cnt + 16'd1; end
                MS_FLUSH: if (ms_out_token == N_TOKENS) begin
                              if (ms_chunk == 47) ms_state <= MS_DONE; else begin ms_chunk <= ms_chunk + 16'd1; ms_state <= MS_WAIT; end
                          end
                MS_DONE:  if (state == S_M3_START || state == S_IDLE) ms_state <= MS_IDLE;   // S_WAIT_MS2 has consumed ms_done by then
                default:  ms_state <= MS_IDLE;
            endcase
        end
    end
    // Weight / parameter addresses and data are registered. The depthwise weights arrive five cycles after the FSM state
    // (address register, memory with two output registers, dww_q), the 1x1 weights three cycles after it; the
    // parameters do not change during a chunk. So the depthwise load strobe is delayed five times, the 1x1 strobe three times.
    reg load_dw_ms_d1, load_dw_ms_d2, load_dw_ms_d3, load_dw_ms_d4, load_dw_ms_d5, act_valid_ms_d1, load_pw_ms_d1, load_pw_ms_d2, load_pw_ms_d3;
    always @(posedge clk) begin
        if (!rst_n) begin
            load_dw_ms_d1 <= 0; load_dw_ms_d2 <= 0; load_dw_ms_d3 <= 0; load_dw_ms_d4 <= 0; load_dw_ms_d5 <= 0; act_valid_ms_d1 <= 0;
            load_pw_ms_d1 <= 0; load_pw_ms_d2 <= 0; load_pw_ms_d3 <= 0;
        end else begin
            load_dw_ms_d1   <= (ms_state == MS_LOAD);
            load_dw_ms_d2   <= load_dw_ms_d1;
            load_dw_ms_d3   <= load_dw_ms_d2;
            load_dw_ms_d4   <= load_dw_ms_d3;
            load_dw_ms_d5   <= load_dw_ms_d4;
            act_valid_ms_d1 <= (ms_state == MS_CALC);
            load_pw_ms_d1   <= (ms_state == MS_LOAD) && (ms_wt_cnt < 16);
            load_pw_ms_d2   <= load_pw_ms_d1;
            load_pw_ms_d3   <= load_pw_ms_d2;
        end
    end
    always @(posedge clk) begin
        if (!rst_n) ms_out_token <= 0;
        else if (ms_state == MS_LOAD) ms_out_token <= 0;
        else if (ms_pw_buf_valid) ms_out_token <= ms_out_token + 16'd1;
    end
    wire ms_active = (ms_state == MS_LOAD || ms_state == MS_CALC || ms_state == MS_FLUSH);
    wire [15:0] rev_pw_wt_cnt = (ms_wt_cnt < 16) ? (16'd15 - ms_wt_cnt) : 16'd0;

    wire [127:0] ms_sram_rdata = (ms_chunk < 16) ? q_ms : (ms_chunk < 32) ? k_ms : v_ms;
    wire ms_lb_valid; wire [25*128-1:0] ms_window_data;
    dw_line_buffer #(.IMG_SIZE(IMG_SIZE), .CHANNELS(16), .LW(8)) u_ms_line_buf (
        .clk(clk), .rst_n(rst_n), .in_valid(act_valid_ms_d1), .in_data(ms_sram_rdata), .out_valid(ms_lb_valid), .window_data(ms_window_data)
    );
    wire [16*12-1:0] ms_dw_p; wire [31:0] ms_dw_fsel_unused; wire ms_dw_valid;
    dw_conv_engine8 #(.CHANNELS(16)) u_ms_dw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_dw_ms_d5), .flat_wt_in(dww_q), .flat_prm_in(dwb_q),
        .in_valid(ms_lb_valid), .window_data(ms_window_data), .out_valid(ms_dw_valid), .flat_out_p(ms_dw_p), .flat_out_fsel(ms_dw_fsel_unused)
    );
    // the MS depthwise output is int8: saturate the 12-bit value
    reg [127:0] ms_dw8;
    reg         ms_dw8_valid;
    genvar ml;
    generate
        for (ml = 0; ml < 16; ml = ml + 1) begin : msat
            wire signed [11:0] p = ms_dw_p[ml*12 +: 12];
            wire ov = (p[11:7] != {5{p[11]}});
            always @(posedge clk) ms_dw8[ml*8 +: 8] <= ov ? (p[11] ? 8'h80 : 8'h7F) : p[7:0];
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n) ms_dw8_valid <= 0; else ms_dw8_valid <= ms_dw_valid;
    end
    ms_pw_engine8 u_ms_pw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_pw_ms_d3), .act_valid(ms_dw8_valid),
        .flat_act_in(ms_dw8), .flat_wt_in(mspww_q), .flat_prm_in(mspwb_q),
        .flat_out_data(ms_pw_buf_data), .out_valid(ms_pw_buf_valid)
    );

    // ------------------------------------------------------------------ proj / expand / DW pass / pw
    wire proj_we = cl_out_valid && (layer == L_M3);
    banked_sram #(.K(KC), .W(256), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_proj_A (.clk(clk), .we(proj_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(projA_raddr), .rdata(sram_proj_rdata_A), .raddr2({AW{1'b0}}), .rdata2());
    banked_sram #(.K(KC), .W(256), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_proj_B (.clk(clk), .we(proj_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(addr_cnt[AW-1:0]), .rdata(sram_proj_rdata_B), .raddr2({AW{1'b0}}), .rdata2());

    // expand: the cluster output is already through the hardswish table
    wire exp_we = cl_out_valid && (layer == L_P2);
    reg  [7:0] exp_groups_done;                                  // expand groups fully written to sram_exp
    always @(posedge clk) begin
        if (!rst_n) exp_groups_done <= 0;
        else if (state == S_P2_START) exp_groups_done <= 0;
        else if (exp_we && cl_out_token == N_TOKENS-1) exp_groups_done <= exp_groups_done + 8'd1;
    end

    // DW pass FSM (background during P2)
    localparam DW_IDLE = 3'd0, DW_WAIT = 3'd1, DW_LOAD = 3'd2, DW_RUN = 3'd3, DW_DRAIN = 3'd4, DW_DONE = 3'd5;
    reg [2:0]  dw_state;
    reg [15:0] dw_chunk, dw_wt_cnt, dw_in_tok, dw_token_cnt;
    wire dw_valid;
    wire dw_chunk_ready = (exp_groups_done > (dw_chunk / KC));
    assign dw_done  = (dw_state == DW_DONE);
    assign dw_ahead = (dw_chunk >= DW_HEAD);
    always @(posedge clk) begin
        if (!rst_n) begin dw_state <= DW_IDLE; dw_chunk <= 0; dw_wt_cnt <= 0; dw_in_tok <= 0; end
        else begin
            case (dw_state)
                DW_IDLE:  begin dw_chunk <= 0; dw_wt_cnt <= 0; dw_in_tok <= 0; if (state == S_P2_START) dw_state <= DW_WAIT; end
                DW_WAIT:  if (dw_chunk_ready) dw_state <= DW_LOAD;
                DW_LOAD:  begin dw_in_tok <= 0; if (dw_wt_cnt == 24) begin dw_wt_cnt <= 0; dw_state <= DW_RUN; end else dw_wt_cnt <= dw_wt_cnt + 16'd1; end
                DW_RUN:   begin if (dw_in_tok == N_TOKENS-1) dw_state <= DW_DRAIN; else dw_in_tok <= dw_in_tok + 16'd1; end
                DW_DRAIN: if (dw_token_cnt == N_TOKENS) begin
                              if (dw_chunk == C_OUT_CHUNKS-1) dw_state <= DW_DONE; else begin dw_chunk <= dw_chunk + 16'd1; dw_state <= DW_WAIT; end
                          end
                DW_DONE:  if (state == S_P4_RES_ADD || state == S_IDLE) dw_state <= DW_IDLE; // S_WAIT_DW has consumed dw_done by then
                default:  dw_state <= DW_IDLE;
            endcase
        end
    end
    reg dw_run_d1, load_dw_d1, load_dw_d2, load_dw_d3, load_dw_d4, load_dw_d5;
    always @(posedge clk) begin
        if (!rst_n) begin dw_run_d1 <= 0; load_dw_d1 <= 0; load_dw_d2 <= 0; load_dw_d3 <= 0; load_dw_d4 <= 0; load_dw_d5 <= 0; end
        else begin
            dw_run_d1 <= (dw_state == DW_RUN);
            load_dw_d1 <= (dw_state == DW_LOAD); load_dw_d2 <= load_dw_d1; load_dw_d3 <= load_dw_d2; load_dw_d4 <= load_dw_d3; load_dw_d5 <= load_dw_d4;
        end
    end
    wire [EAW-1:0] exp_waddr = (cl_out_token * C_OUT_CHUNKS) + cl_out_oc_base;
    wire [EAW-1:0] exp_raddr = (dw_in_tok * C_OUT_CHUNKS) + dw_chunk;
    wire [127:0] sram_exp_rdata;
    banked_sram #(.K(KC), .W(128), .DEPTH(EXP_WORDS), .AW(EAW)) inst_sram_exp (.clk(clk), .we(exp_we),
        .waddr(exp_waddr), .wdata(cl_out8), .raddr(exp_raddr), .rdata(sram_exp_rdata), .raddr2({EAW{1'b0}}), .rdata2());

    wire lb_valid; wire [25*128-1:0] window_data;
    dw_line_buffer #(.IMG_SIZE(IMG_SIZE), .CHANNELS(16), .LW(8)) u_line_buf (
        .clk(clk), .rst_n(rst_n), .in_valid(dw_run_d1), .in_data(sram_exp_rdata), .out_valid(lb_valid), .window_data(window_data)
    );
    wire [16*12-1:0] dw_p; wire [31:0] dw_fsel; wire dw_valid_raw;
    dw_conv_engine8 #(.CHANNELS(16)) u_dw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_dw_d5), .flat_wt_in(dww_q), .flat_prm_in(dwb_q),
        .in_valid(lb_valid), .window_data(window_data), .out_valid(dw_valid_raw), .flat_out_p(dw_p), .flat_out_fsel(dw_fsel)
    );
    // hardswish table (F = 5 + fsel), 3 cycles
    wire [127:0] dw_out_act;
    genvar hl;
    generate
        for (hl = 0; hl < 16; hl = hl + 2) begin : dwhs
            hswish_rom2 #(.F_LO(5), .AW(12)) u_hs (.clk(clk),
                .p_a(dw_p[hl*12 +: 12]),     .f_a(dw_fsel[hl*2 +: 2]),     .y_a(dw_out_act[hl*8 +: 8]),
                .p_b(dw_p[(hl+1)*12 +: 12]), .f_b(dw_fsel[(hl+1)*2 +: 2]), .y_b(dw_out_act[(hl+1)*8 +: 8]));
        end
    endgenerate
    reg [2:0] dw_hs_v;
    always @(posedge clk) begin
        if (!rst_n) dw_hs_v <= 0; else dw_hs_v <= {dw_hs_v[1:0], dw_valid_raw};
    end
    assign dw_valid = dw_hs_v[2];
    always @(posedge clk) begin
        if (!rst_n) dw_token_cnt <= 0;
        else if (dw_state == DW_LOAD || dw_state == DW_IDLE) dw_token_cnt <= 0;
        else if (dw_valid) dw_token_cnt <= dw_token_cnt + 16'd1;
    end
    wire [DAW-1:0] dw_waddr = (dw_chunk * N_TOKENS) + dw_token_cnt;
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(DW_WORDS), .ADDR_WIDTH(DAW), .WREG(1)) inst_sram_dw (.clk(clk), .we(dw_valid && dw_token_cnt < N_TOKENS),
        .waddr(dw_waddr), .wdata(dw_out_act), .raddr(dw_raddr), .rdata(sram_dw_rdata));

    wire pw_we = cl_out_valid && (layer == L_P3);
    banked_sram #(.K(KC), .W(256), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_pw_raw (.clk(clk), .we(pw_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(addr_cnt[AW-1:0]), .rdata(sram_pw_raw_rdata), .raddr2({AW{1'b0}}), .rdata2());

    // small weight memories: MS branch while it is active, DW pass otherwise (never both). Registered addresses.
    always @(posedge clk) begin
        dww_raddr   <= {blk, 12'd0} + (ms_active ? (14'd2048 + ms_chunk[5:0] * 14'd25 + ms_wt_cnt[4:0]) : (dw_chunk[5:0] * 14'd25 + dw_wt_cnt[4:0]));
        dwb_raddr   <= {blk, 7'd0}  + (ms_active ? (9'd64 + ms_chunk[5:0]) : {3'd0, dw_chunk[5:0]});
        mspww_raddr <= {blk, 10'd0} + {ms_chunk[5:0], 4'd0} + rev_pw_wt_cnt[3:0];
        mspwb_raddr <= {blk, 6'd0}  + ms_chunk[5:0];
    end

    // ------------------------------------------------------------------ P4: x <= sat(sat(x + proj) + pw), in place
    reg  [255:0] xB_r, projB_r, pwB_r, midB_r, pwB_r2;
    wire [255:0] mid_B, y_w;
    adder_array u_add_mid_B  (.a(xB_r), .b(projB_r), .c(mid_B));
    adder_array u_add_pw_res (.a(midB_r), .b(pwB_r2), .c(y_w));
    integer pi;
    always @(posedge clk) begin
        xB_r    <= sram_x_rdata_B;
        projB_r <= sram_proj_rdata_B;
        pwB_r   <= sram_pw_raw_rdata;
        midB_r  <= mid_B;
        pwB_r2  <= pwB_r;
        y_r     <= y_w;
        p4_a_sr[1] <= addr_cnt[AW-1:0];
        for (pi = 2; pi <= P4_LAT; pi = pi + 1) p4_a_sr[pi] <= p4_a_sr[pi-1];
    end
    always @(posedge clk) begin
        if (!rst_n) p4_v_sr <= 0;
        else        p4_v_sr <= {p4_v_sr[P4_LAT-1:1], p4_rd};
    end
endmodule
