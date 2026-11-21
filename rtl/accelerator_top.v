`timescale 1ns/1ps

// ============================================================================
// EfficientViT-B1 Stage 4 block accelerator, v4.1 (2026-10-10): v3 datapath + overlapped scheduling
//   v4.1: cl_groups_done cleared at *_START (cross-run MS chunk-0 stale read, see below)
//
//   LOAD_X  host x -> sram_x_in
//   M1      QKV 1x1 (256->768)      conv_cluster (K out-chunks at once)        -> sram_q/k/v
//           MS branch (5x5 DW + group 1x1, own engines) trails M1 group by group -> sram_q2/k2/v2
//   ATT_Q   16 heads on q/k/v, litemla_core2: phase 1 of head h+1 overlaps phase 2 of head h   -> sram_h1
//   ATT_MS  same on q2/k2/v2 (waits for the MS branch)                                        -> sram_h2
//   M3      proj 1x1 (512->256)     conv_cluster                              -> sram_proj   (mid = sat(x + proj) at read time)
//   P2      expand 1x1 (256->1024) + hswish   conv_cluster                    -> sram_exp
//           DW pass (5x5 depthwise + hswish) trails P2 group by group         -> sram_dw
//   P3      point 1x1 (1024->256)   conv_cluster                              -> sram_pw_raw
//   P4      y = sat(sat(x + proj) + pw)                                       -> BRAM_Y
//
// Arithmetic identical to the verified RTL (model/rtl_model.py); only scheduling changed.
// ============================================================================
module accelerator_top #(
    parameter ROWS = 16,
    parameter COLS = 16,
    parameter N_TOKENS = 256,
    parameter IMG_SIZE = 16,
    parameter C_IN_CHUNKS = 16,
    parameter C_OUT_CHUNKS = 64,
    parameter FRAC_BITS = 11,
    parameter A_FRAC_BITS = 5,
    parameter EXPAND_ACT = 1,
    parameter DW_ACT = 1,
    parameter CONV_PAR = 8       // K (K*16 <= N_TOKENS-34)
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 reg_start,
    output reg                  reg_done,
    output reg  [31:0]          reg_cycle_cnt,

    output wire                 bram_x_clk,    output wire                 bram_x_en,
    output wire                 bram_w_clk,    output wire                 bram_w_en,
    output wire                 bram_b_clk,    output wire                 bram_b_en,
    output wire                 bram_dw_w_clk, output wire                 bram_dw_w_en,
    output wire                 bram_dw_b_clk, output wire                 bram_dw_b_en,
    output wire                 bram_pw_w_clk, output wire                 bram_pw_w_en,
    output wire                 bram_pw_b_clk, output wire                 bram_pw_b_en,
    output wire                 bram_ms_pw_w_clk, output wire              bram_ms_pw_w_en,
    output wire                 bram_ms_pw_b_clk, output wire              bram_ms_pw_b_en,
    output wire [31:0]          bram_ms_pw_w_addr, input wire [255:0]      bram_ms_pw_w_rdata,
    output wire [31:0]          bram_ms_pw_b_addr, input wire [255:0]      bram_ms_pw_b_rdata,
    output wire                 bram_y_clk,    output wire                 bram_y_en,
    output wire                 bram_mla_w_clk, output wire                bram_mla_w_en,
    output wire                 bram_mla_b_clk, output wire                bram_mla_b_en,
    output wire [31:0]          bram_x_addr,    input  wire [255:0]        bram_x_rdata,
    output wire [31:0]          bram_w_addr,    input  wire [255:0]        bram_w_rdata,
    output wire [31:0]          bram_b_addr,    input  wire [255:0]        bram_b_rdata,
    output wire [31:0]          bram_dw_w_addr, input  wire [255:0]        bram_dw_w_rdata,
    output wire [31:0]          bram_dw_b_addr, input  wire [255:0]        bram_dw_b_rdata,
    output wire [31:0]          bram_pw_w_addr, input  wire [255:0]        bram_pw_w_rdata,
    output wire [31:0]          bram_pw_b_addr, input  wire [255:0]        bram_pw_b_rdata,
    output wire [31:0]          bram_y_addr,    output wire [255:0]        bram_y_wdata,
    output wire [31:0]          bram_y_we,
    output wire [31:0]          bram_mla_w_addr, input wire [255:0]        bram_mla_w_rdata,
    output wire [31:0]          bram_mla_b_addr, input wire [255:0]        bram_mla_b_rdata
);
    localparam K         = CONV_PAR;
    localparam ACT_WORDS = N_TOKENS * C_IN_CHUNKS;
    localparam AW        = $clog2(ACT_WORDS);
    localparam EXP_WORDS = N_TOKENS * C_OUT_CHUNKS;
    localparam EAW       = $clog2(EXP_WORDS);
    localparam DW_WORDS  = N_TOKENS * C_OUT_CHUNKS;
    localparam DAW       = $clog2(DW_WORDS);
    localparam N_HEADS   = 16;

    assign bram_x_clk = clk;       assign bram_x_en = 1'b1;
    assign bram_w_clk = clk;       assign bram_w_en = 1'b1;
    assign bram_b_clk = clk;       assign bram_b_en = 1'b1;
    assign bram_dw_w_clk = clk;    assign bram_dw_w_en = 1'b1;
    assign bram_dw_b_clk = clk;    assign bram_dw_b_en = 1'b1;
    assign bram_pw_w_clk = clk;    assign bram_pw_w_en = 1'b1;
    assign bram_pw_b_clk = clk;    assign bram_pw_b_en = 1'b1;
    assign bram_ms_pw_w_clk = clk; assign bram_ms_pw_w_en = 1'b1;
    assign bram_ms_pw_b_clk = clk; assign bram_ms_pw_b_en = 1'b1;
    assign bram_y_clk = clk;       assign bram_y_en = 1'b1;
    assign bram_mla_w_clk = clk;   assign bram_mla_w_en = 1'b1;
    assign bram_mla_b_clk = clk;   assign bram_mla_b_en = 1'b1;

    // ------------------------------------------------------------------ main FSM
    localparam S_IDLE = 5'd0, S_LOAD_X = 5'd1, S_M1_START = 5'd2, S_M1_RUN = 5'd3,
               S_ATTQ_START = 5'd4, S_ATTQ_RUN = 5'd5, S_WAIT_MS = 5'd6, S_ATTM_START = 5'd7, S_ATTM_RUN = 5'd8,
               S_M3_START = 5'd9, S_M3_RUN = 5'd10, S_P2_START = 5'd11, S_P2_RUN = 5'd12, S_WAIT_DW = 5'd13,
               S_P3_START = 5'd14, S_P3_RUN = 5'd15, S_P4_RES_ADD = 5'd16, S_DONE = 5'd17;
    localparam L_M1 = 2'd0, L_M3 = 2'd1, L_P2 = 2'd2, L_P3 = 2'd3;

    reg [4:0]  state, next_state;
    reg [1:0]  layer;
    reg [15:0] addr_cnt;
    wire cl_busy, cl_done, att_done, ms_done, dw_done;
    wire cl_layer_start = (state == S_M1_START || state == S_M3_START || state == S_P2_START || state == S_P3_START);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE; else state <= next_state;
    end
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:        if (reg_start) next_state = S_LOAD_X;
            S_LOAD_X:      if (addr_cnt == ACT_WORDS-1) next_state = S_M1_START;
            S_M1_START:    next_state = S_M1_RUN;
            S_M1_RUN:      if (cl_done) next_state = S_ATTQ_START;
            S_ATTQ_START:  next_state = S_ATTQ_RUN;
            S_ATTQ_RUN:    if (att_done) next_state = S_WAIT_MS;
            S_WAIT_MS:     if (ms_done) next_state = S_ATTM_START;
            S_ATTM_START:  next_state = S_ATTM_RUN;
            S_ATTM_RUN:    if (att_done) next_state = S_M3_START;
            S_M3_START:    next_state = S_M3_RUN;
            S_M3_RUN:      if (cl_done) next_state = S_P2_START;
            S_P2_START:    next_state = S_P2_RUN;
            S_P2_RUN:      if (cl_done) next_state = S_WAIT_DW;
            S_WAIT_DW:     if (dw_done) next_state = S_P3_START;
            S_P3_START:    next_state = S_P3_RUN;
            S_P3_RUN:      if (cl_done) next_state = S_P4_RES_ADD;
            S_P4_RES_ADD:  if (addr_cnt == ACT_WORDS-1) next_state = S_DONE;
            S_DONE:        if (!reg_start) next_state = S_IDLE;
            default:       next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin layer <= 0; addr_cnt <= 0; reg_cycle_cnt <= 0; end
        else begin
            if (state == S_IDLE) begin if (reg_start) reg_cycle_cnt <= 0; end
            else if (state != S_DONE) reg_cycle_cnt <= reg_cycle_cnt + 1;
            if (state == S_LOAD_X || state == S_P4_RES_ADD) addr_cnt <= addr_cnt + 1; else addr_cnt <= 0;
            case (state)
                S_M1_START: layer <= L_M1;
                S_M3_START: layer <= L_M3;
                S_P2_START: layer <= L_P2;
                S_P3_START: layer <= L_P3;
                default: ;
            endcase
        end
    end
    always @(posedge clk or negedge rst_n) if (!rst_n) reg_done <= 0; else reg_done <= (state == S_DONE);

    reg [15:0] addr_cnt_d1;
    reg state_load_x_d1, state_p4_res_add_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin addr_cnt_d1 <= 0; state_load_x_d1 <= 0; state_p4_res_add_d1 <= 0; end
        else begin addr_cnt_d1 <= addr_cnt; state_load_x_d1 <= (state == S_LOAD_X); state_p4_res_add_d1 <= (state == S_P4_RES_ADD); end
    end

    // ------------------------------------------------------------------ conv cluster
    wire [15:0] cl_act_token;  wire [7:0] cl_act_ic;  wire cl_act_fire;
    wire [31:0] cl_w_addr, cl_b_addr;
    wire        cl_out_valid;  wire [15:0] cl_out_token;  wire [7:0] cl_out_oc_base;
    wire [K*256-1:0] cl_out_data;
    reg         cl_start;
    reg [7:0]   cfg_oc, cfg_ic;
    reg         cfg_ic_major;
    reg [31:0]  cfg_w_base, cfg_b_base;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin cl_start <= 0; cfg_oc <= 0; cfg_ic <= 0; cfg_ic_major <= 0; cfg_w_base <= 0; cfg_b_base <= 0; end
        else begin
            cl_start <= cl_layer_start;
            case (state)
                S_M1_START: begin cfg_oc <= 48; cfg_ic <= 16; cfg_ic_major <= 0; cfg_w_base <= 0;     cfg_b_base <= 0;  end
                S_M3_START: begin cfg_oc <= 16; cfg_ic <= 32; cfg_ic_major <= 0; cfg_w_base <= 12288; cfg_b_base <= 48; end
                S_P2_START: begin cfg_oc <= 64; cfg_ic <= 16; cfg_ic_major <= 0; cfg_w_base <= 0;     cfg_b_base <= 0;  end
                S_P3_START: begin cfg_oc <= 16; cfg_ic <= 64; cfg_ic_major <= 1; cfg_w_base <= 0;     cfg_b_base <= 0;  end
                default: ;
            endcase
        end
    end
    wire [255:0] cl_w_rdata = (layer == L_M1 || layer == L_M3) ? bram_mla_w_rdata : (layer == L_P2) ? bram_w_rdata : bram_pw_w_rdata;
    wire [255:0] cl_b_rdata = (layer == L_M1 || layer == L_M3) ? bram_mla_b_rdata : (layer == L_P2) ? bram_b_rdata : bram_pw_b_rdata;
    assign bram_mla_w_addr = cl_w_addr << 5;
    assign bram_mla_b_addr = cl_b_addr << 5;
    assign bram_w_addr     = cl_w_addr << 5;
    assign bram_b_addr     = cl_b_addr << 5;
    assign bram_pw_w_addr  = cl_w_addr << 5;
    assign bram_pw_b_addr  = cl_b_addr << 5;

    reg [7:0] cl_ic_d1;  reg [1:0] layer_d1;
    always @(posedge clk) begin cl_ic_d1 <= cl_act_ic; layer_d1 <= layer; end

    wire [255:0] sram_x_in_rdata_A, sram_x_in_rdata_B, sram_proj_rdata_A, sram_proj_rdata_B;
    wire [255:0] sram_h1_rdata, sram_h2_rdata, sram_dw_rdata, sram_pw_raw_rdata;
    wire [255:0] mid_A, mid_B;
    adder_array u_add_mid_A (.a(sram_x_in_rdata_A), .b(sram_proj_rdata_A), .c(mid_A));
    adder_array u_add_mid_B (.a(sram_x_in_rdata_B), .b(sram_proj_rdata_B), .c(mid_B));
    wire [255:0] cl_act_data =
        (layer_d1 == L_M1) ? sram_x_in_rdata_A :
        (layer_d1 == L_M3) ? ((cl_ic_d1 < 16) ? sram_h1_rdata : sram_h2_rdata) :
        (layer_d1 == L_P2) ? mid_A : sram_dw_rdata;

    conv_cluster #(.K(K), .N_TOKENS(N_TOKENS), .FRAC_BITS(FRAC_BITS)) u_cluster (
        .clk(clk), .rst_n(rst_n), .start(cl_start),
        .cfg_oc(cfg_oc), .cfg_ic(cfg_ic), .cfg_ic_major(cfg_ic_major), .cfg_w_base(cfg_w_base), .cfg_b_base(cfg_b_base),
        .busy(cl_busy), .done(cl_done),
        .act_token(cl_act_token), .act_ic(cl_act_ic), .act_fire(cl_act_fire), .act_data(cl_act_data),
        .w_addr(cl_w_addr), .w_rdata(cl_w_rdata), .b_addr(cl_b_addr), .b_rdata(cl_b_rdata),
        .out_valid(cl_out_valid), .out_token(cl_out_token), .out_oc_base(cl_out_oc_base), .out_data(cl_out_data)
    );
    // groups completed by the cluster in the current layer (last token of a group written).
    // Cleared in the *_START state (same edge the MS FSM leaves MS_IDLE), not on cl_start one cycle
    // later: with the late clear, run 2+ saw the previous run's P3 count (2) for one cycle, so
    // ms_chunk_ready fired at once and MS chunk 0 read the PREVIOUS block's q chunk 0 (2026-10-10 board bug).
    reg [7:0] cl_groups_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cl_groups_done <= 0;
        else if (cl_layer_start) cl_groups_done <= 0;
        else if (cl_out_valid && cl_out_token == N_TOKENS-1) cl_groups_done <= cl_groups_done + 1;
    end

    // ------------------------------------------------------------------ activation SRAMs
    wire [AW-1:0] x_raddr_A = (cl_act_token * C_IN_CHUNKS) + cl_act_ic;
    true_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_x_in (
        .clk(clk),
        .we_a(state_load_x_d1 && addr_cnt_d1 < ACT_WORDS), .addr_a(state_load_x_d1 ? addr_cnt_d1[AW-1:0] : x_raddr_A),
        .din_a(bram_x_rdata), .dout_a(sram_x_in_rdata_A),
        .we_b(1'b0), .addr_b(addr_cnt[AW-1:0]), .din_b(256'd0), .dout_b(sram_x_in_rdata_B)
    );

    // attention sequencer signals (defined below) that address q/k/v
    wire [AW-1:0] att_q_raddr, att_kv_raddr;
    // MS branch signals
    reg  [15:0] ms_chunk, ms_act_cnt, ms_wt_cnt, ms_out_token;
    wire [AW-1:0] ms_raddr = (ms_act_cnt * 16) + (ms_chunk % 16);

    wire [AW-1:0] cl_out_addr16 = (cl_out_token * 16) + (cl_out_oc_base % 16);
    wire m1_we = cl_out_valid && (layer == L_M1);
    wire [255:0] q_att, k_att, v_att, q_ms, k_ms, v_ms;                 // port 1: attention, port 2: MS branch
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_q (.clk(clk), .we(m1_we && cl_out_oc_base < 16),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(att_q_raddr), .rdata(q_att), .raddr2(ms_raddr), .rdata2(q_ms));
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_k (.clk(clk), .we(m1_we && cl_out_oc_base >= 16 && cl_out_oc_base < 32),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(att_kv_raddr), .rdata(k_att), .raddr2(ms_raddr), .rdata2(k_ms));
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_v (.clk(clk), .we(m1_we && cl_out_oc_base >= 32),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(att_kv_raddr), .rdata(v_att), .raddr2(ms_raddr), .rdata2(v_ms));

    wire ms_pw_buf_valid;  wire [255:0] ms_pw_buf_data;
    wire [AW-1:0] ms_out_addr = (ms_out_token * 16) + (ms_chunk % 16);
    wire [255:0] q2_att, k2_att, v2_att;
    simple_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_q2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk < 16),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .re(1'b1), .raddr(att_q_raddr), .rdata(q2_att));
    simple_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_k2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk >= 16 && ms_chunk < 32),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .re(1'b1), .raddr(att_kv_raddr), .rdata(k2_att));
    simple_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_v2 (.clk(clk), .we(ms_pw_buf_valid && ms_chunk >= 32),
        .waddr(ms_out_addr), .wdata(ms_pw_buf_data), .re(1'b1), .raddr(att_kv_raddr), .rdata(v2_att));

    // ------------------------------------------------------------------ attention sequencer
    // step s = 0..16, 260 cycles each: cycles 0..255 stream kv of head s (s<16) and q of head s-1 (s>=1);
    // cycle 258 clears the bank of head s+1 (its q user, head s-1, left the MAC stage by then).
    localparam ATT_STEP = N_TOKENS + 4;
    reg        att_run, att_src;              // att_src: 0 = q/k/v (h1), 1 = q2/k2/v2 (h2)
    reg [4:0]  att_step;
    reg [15:0] att_cyc;
    reg        kv_fire, q_fire, q_fire_bank, p1_begin, p1_bank;
    reg        kv_valid_d1, q_valid_d1, q_bank_d1;
    reg [15:0] att_out_tok;
    reg [4:0]  att_out_head;
    wire       att_stream = att_run && (att_cyc < N_TOKENS);
    wire       att_kv_on  = att_stream && (att_step < N_HEADS);
    wire       att_q_on   = att_stream && (att_step >= 1);
    assign att_kv_raddr = (att_cyc * 16) + att_step[3:0];
    assign att_q_raddr  = (att_cyc * 16) + (att_step - 1);
    wire [255:0] mla_out_data;
    wire         mla_out_valid;
    assign att_done = att_run && (att_out_head == N_HEADS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            att_run <= 0; att_src <= 0; att_step <= 0; att_cyc <= 0; p1_begin <= 0; p1_bank <= 0;
            kv_valid_d1 <= 0; q_valid_d1 <= 0; q_bank_d1 <= 0; att_out_tok <= 0; att_out_head <= 0;
        end else begin
            kv_valid_d1 <= att_kv_on;
            q_valid_d1  <= att_q_on;
            q_bank_d1   <= (att_step - 1) & 1;
            p1_begin    <= 0;
            if (state == S_ATTQ_START || state == S_ATTM_START) begin
                att_run <= 1; att_src <= (state == S_ATTM_START); att_step <= 0; att_cyc <= 0;
                att_out_tok <= 0; att_out_head <= 0;
                p1_begin <= 1; p1_bank <= 0;                        // bank 0 for head 0; streams start next cycle (cyc 0)
            end else if (att_run) begin
                if (att_cyc == ATT_STEP - 2 && att_step < N_HEADS - 1) begin p1_begin <= 1; p1_bank <= ~att_step[0]; end   // head s+1 -> bank (s+1)%2
                if (att_cyc == ATT_STEP - 1) begin
                    if (att_step < N_HEADS) begin att_cyc <= 0; att_step <= att_step + 1; end   // step 16 (q of head 15) then holds until drained
                end else att_cyc <= att_cyc + 1;
                if (mla_out_valid) begin
                    if (att_out_tok == N_TOKENS-1) begin att_out_tok <= 0; att_out_head <= att_out_head + 1; end
                    else att_out_tok <= att_out_tok + 1;
                end
                if (att_done) att_run <= 0;
            end
        end
    end
    wire [255:0] att_q_in = att_src ? q2_att : q_att;
    wire [255:0] att_k_in = att_src ? k2_att : k_att;
    wire [255:0] att_v_in = att_src ? v2_att : v_att;
    litemla_core2 #(.DIM(16), .N_TOKENS(N_TOKENS), .FRAC_BITS(A_FRAC_BITS)) u_litemla (
        .clk(clk), .rst_n(rst_n),
        .p1_begin(p1_begin), .p1_bank(p1_bank), .kv_valid(kv_valid_d1), .k_in(att_k_in), .v_in({16'd1 << A_FRAC_BITS, att_v_in}),
        .q_valid(q_valid_d1), .q_bank(q_bank_d1), .q_in(att_q_in),
        .out_data(mla_out_data), .out_valid(mla_out_valid)
    );
    wire [AW-1:0] h_raddr = (cl_act_token * 16) + (cl_act_ic % 16);
    wire [AW-1:0] h_waddr = (att_out_tok * 16) + att_out_head[3:0];
    simple_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_h1 (.clk(clk), .we(mla_out_valid && !att_src),
        .waddr(h_waddr), .wdata(mla_out_data), .re(1'b1), .raddr(h_raddr), .rdata(sram_h1_rdata));
    simple_dual_port_bram #(.DEPTH(ACT_WORDS), .ADDR_WIDTH(AW)) inst_sram_h2 (.clk(clk), .we(mla_out_valid && att_src),
        .waddr(h_waddr), .wdata(mla_out_data), .re(1'b1), .raddr(h_raddr), .rdata(sram_h2_rdata));

    // ------------------------------------------------------------------ MS branch (background, trails M1 group by group)
    localparam MS_IDLE = 3'd0, MS_WAIT = 3'd1, MS_LOAD = 3'd2, MS_CALC = 3'd3, MS_FLUSH = 3'd4, MS_DONE = 3'd5;
    reg [2:0] ms_state;
    wire ms_chunk_ready = (layer == L_M1) && (cl_groups_done > (ms_chunk / K));    // its q/k/v group is in SRAM
    assign ms_done = (ms_state == MS_DONE);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ms_state <= MS_IDLE; ms_chunk <= 0; ms_act_cnt <= 0; ms_wt_cnt <= 0; end
        else begin
            case (ms_state)
                MS_IDLE:  begin ms_chunk <= 0; ms_act_cnt <= 0; ms_wt_cnt <= 0; if (state == S_M1_START) ms_state <= MS_WAIT; end
                MS_WAIT:  if (ms_chunk_ready) ms_state <= MS_LOAD;
                MS_LOAD:  begin ms_act_cnt <= 0; if (ms_wt_cnt == 24) begin ms_wt_cnt <= 0; ms_state <= MS_CALC; end else ms_wt_cnt <= ms_wt_cnt + 1; end
                MS_CALC:  begin if (ms_act_cnt == N_TOKENS-1) ms_state <= MS_FLUSH; else ms_act_cnt <= ms_act_cnt + 1; end
                MS_FLUSH: if (ms_out_token == N_TOKENS) begin
                              if (ms_chunk == 47) ms_state <= MS_DONE; else begin ms_chunk <= ms_chunk + 1; ms_state <= MS_WAIT; end
                          end
                MS_DONE:  if (state == S_IDLE) ms_state <= MS_IDLE;
                default:  ms_state <= MS_IDLE;
            endcase
        end
    end
    reg load_dw_ms_d1, act_valid_ms_d1, load_pw_ms_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin load_dw_ms_d1 <= 0; act_valid_ms_d1 <= 0; load_pw_ms_d1 <= 0; end
        else begin
            load_dw_ms_d1   <= (ms_state == MS_LOAD);
            act_valid_ms_d1 <= (ms_state == MS_CALC);
            load_pw_ms_d1   <= (ms_state == MS_LOAD) && (ms_wt_cnt < 16);
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) ms_out_token <= 0;
        else if (ms_state == MS_LOAD) ms_out_token <= 0;
        else if (ms_pw_buf_valid) ms_out_token <= ms_out_token + 1;
    end
    wire ms_active = (ms_state == MS_LOAD || ms_state == MS_CALC || ms_state == MS_FLUSH);
    wire clear_ms  = (ms_state == MS_LOAD) && (ms_chunk == 0);
    wire [15:0] rev_pw_wt_cnt = (ms_wt_cnt < 16) ? (16'd15 - ms_wt_cnt) : 16'd0;
    assign bram_ms_pw_w_addr = (ms_state == MS_LOAD && ms_wt_cnt < 16) ? ({16'd0, (ms_chunk * 16) + rev_pw_wt_cnt} << 5) : 32'd0;
    assign bram_ms_pw_b_addr = {16'd0, ms_chunk} << 5;

    wire [255:0] ms_sram_rdata = (ms_chunk < 16) ? q_ms : (ms_chunk < 32) ? k_ms : v_ms;
    wire ms_lb_valid; wire [6399:0] ms_window_data;
    dw_line_buffer #(.IMG_SIZE(IMG_SIZE), .CHANNELS(16)) u_ms_line_buf (
        .clk(clk), .rst_n(rst_n), .in_valid(act_valid_ms_d1), .in_data(ms_sram_rdata), .out_valid(ms_lb_valid), .window_data(ms_window_data)
    );
    wire [255:0] ms_dw_out_data; wire ms_dw_valid;
    dw_conv_engine #(.CHANNELS(16), .FRAC_BITS(FRAC_BITS)) u_ms_dw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_dw_ms_d1), .flat_wt_in(bram_dw_w_rdata), .flat_bias_in(bram_dw_b_rdata),
        .in_valid(ms_lb_valid), .window_data(ms_window_data), .out_valid(ms_dw_valid), .flat_out_data(ms_dw_out_data)
    );
    wire [767:0] ms_pw_raw_psum; wire ms_pw_raw_valid;
    conv_engine_top u_ms_pw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_pw_ms_d1), .act_valid(ms_dw_valid),
        .flat_act_in(ms_dw_out_data), .flat_wt_in(bram_ms_pw_w_rdata), .flat_psum_in(768'd0),
        .flat_psum_out(ms_pw_raw_psum), .out_valid(ms_pw_raw_valid)
    );
    psum_buffer #(.COLS(16), .N_TOKENS(N_TOKENS), .CHUNKS(1), .FRAC_BITS(FRAC_BITS), .A_FRAC_BITS(A_FRAC_BITS), .USE_RELU6(0)) u_ms_pw_buf (
        .clk(clk), .rst_n(rst_n), .clear_psum(clear_ms), .in_valid(ms_pw_raw_valid), .flat_psum_in(ms_pw_raw_psum),
        .flat_bias_in(bram_ms_pw_b_rdata), .out_valid(ms_pw_buf_valid), .flat_out_data(ms_pw_buf_data)
    );

    // ------------------------------------------------------------------ proj / expand / DW pass / pw
    wire proj_we = cl_out_valid && (layer == L_M3);
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_proj_A (.clk(clk), .we(proj_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(x_raddr_A), .rdata(sram_proj_rdata_A), .raddr2({AW{1'b0}}), .rdata2());
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_proj_B (.clk(clk), .we(proj_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(addr_cnt[AW-1:0]), .rdata(sram_proj_rdata_B), .raddr2({AW{1'b0}}), .rdata2());

    wire [K*256-1:0] exp_act_data;
    wire exp_act_valid;
    generate
        if (EXPAND_ACT == 1) begin : g_exp_hswish
            wire [K-1:0] hv;
            genvar hk;
            for (hk = 0; hk < K; hk = hk + 1) begin : hs
                hswish_array #(.A_FRAC_BITS(A_FRAC_BITS)) u_hs (.clk(clk), .rst_n(rst_n),
                    .in_valid(cl_out_valid && (layer == L_P2)), .in_data(cl_out_data[hk*256 +: 256]),
                    .out_valid(hv[hk]), .out_data(exp_act_data[hk*256 +: 256]));
            end
            assign exp_act_valid = hv[0];
        end else begin : g_exp_relu6
            genvar hk, hc;
            for (hk = 0; hk < K; hk = hk + 1) begin : rl
                for (hc = 0; hc < 16; hc = hc + 1) begin : ch
                    wire signed [15:0] v = cl_out_data[hk*256 + hc*16 +: 16];
                    assign exp_act_data[hk*256 + hc*16 +: 16] = (v < 0) ? 16'd0 : (v > (16'sd6 <<< A_FRAC_BITS)) ? (16'sd6 <<< A_FRAC_BITS) : v;
                end
            end
            assign exp_act_valid = cl_out_valid && (layer == L_P2);
        end
    endgenerate
    localparam EXP_LAT = (EXPAND_ACT == 1) ? 3 : 0;
    reg [15:0] exp_tok_pipe [0:3];
    reg [7:0]  exp_oc_pipe  [0:3];
    integer ep;
    always @(posedge clk) begin
        exp_tok_pipe[0] <= cl_out_token; exp_oc_pipe[0] <= cl_out_oc_base;
        for (ep = 1; ep < 4; ep = ep + 1) begin exp_tok_pipe[ep] <= exp_tok_pipe[ep-1]; exp_oc_pipe[ep] <= exp_oc_pipe[ep-1]; end
    end
    wire [15:0] exp_tok = (EXP_LAT == 0) ? cl_out_token   : exp_tok_pipe[EXP_LAT-1];
    wire [7:0]  exp_oc  = (EXP_LAT == 0) ? cl_out_oc_base : exp_oc_pipe[EXP_LAT-1];
    reg  [7:0]  exp_groups_done;                                 // expand groups fully written to sram_exp (after hswish)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) exp_groups_done <= 0;
        else if (state == S_P2_START) exp_groups_done <= 0;
        else if (exp_act_valid && exp_tok == N_TOKENS-1) exp_groups_done <= exp_groups_done + 1;
    end

    // DW pass FSM (background during P2)
    localparam DW_IDLE = 3'd0, DW_WAIT = 3'd1, DW_LOAD = 3'd2, DW_RUN = 3'd3, DW_DRAIN = 3'd4, DW_DONE = 3'd5;
    reg [2:0]  dw_state;
    reg [15:0] dw_chunk, dw_wt_cnt, dw_in_tok, dw_token_cnt;
    wire dw_valid;
    wire dw_chunk_ready = (exp_groups_done > (dw_chunk / K));
    assign dw_done = (dw_state == DW_DONE);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin dw_state <= DW_IDLE; dw_chunk <= 0; dw_wt_cnt <= 0; dw_in_tok <= 0; end
        else begin
            case (dw_state)
                DW_IDLE:  begin dw_chunk <= 0; dw_wt_cnt <= 0; dw_in_tok <= 0; if (state == S_P2_START) dw_state <= DW_WAIT; end
                DW_WAIT:  if (dw_chunk_ready) dw_state <= DW_LOAD;
                DW_LOAD:  begin dw_in_tok <= 0; if (dw_wt_cnt == 24) begin dw_wt_cnt <= 0; dw_state <= DW_RUN; end else dw_wt_cnt <= dw_wt_cnt + 1; end
                DW_RUN:   begin if (dw_in_tok == N_TOKENS-1) dw_state <= DW_DRAIN; else dw_in_tok <= dw_in_tok + 1; end
                DW_DRAIN: if (dw_token_cnt == N_TOKENS) begin
                              if (dw_chunk == C_OUT_CHUNKS-1) dw_state <= DW_DONE; else begin dw_chunk <= dw_chunk + 1; dw_state <= DW_WAIT; end
                          end
                DW_DONE:  if (state == S_IDLE) dw_state <= DW_IDLE;
                default:  dw_state <= DW_IDLE;
            endcase
        end
    end
    reg dw_run_d1, load_dw_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin dw_run_d1 <= 0; load_dw_d1 <= 0; end
        else begin dw_run_d1 <= (dw_state == DW_RUN); load_dw_d1 <= (dw_state == DW_LOAD); end
    end
    wire [EAW-1:0] exp_waddr = (exp_tok * C_OUT_CHUNKS) + exp_oc;
    wire [EAW-1:0] exp_raddr = (dw_in_tok * C_OUT_CHUNKS) + dw_chunk;
    wire [255:0] sram_exp_rdata;
    banked_sram #(.K(K), .DEPTH(EXP_WORDS), .AW(EAW), .ULTRA(1)) inst_sram_exp (.clk(clk), .we(exp_act_valid),
        .waddr(exp_waddr), .wdata(exp_act_data), .raddr(exp_raddr), .rdata(sram_exp_rdata), .raddr2({EAW{1'b0}}), .rdata2());

    wire lb_valid; wire [6399:0] window_data;
    dw_line_buffer #(.IMG_SIZE(IMG_SIZE), .CHANNELS(16)) u_line_buf (
        .clk(clk), .rst_n(rst_n), .in_valid(dw_run_d1), .in_data(sram_exp_rdata), .out_valid(lb_valid), .window_data(window_data)
    );
    wire [255:0] dw_out_data; wire dw_valid_raw;
    dw_conv_engine #(.CHANNELS(16), .FRAC_BITS(FRAC_BITS)) u_dw_engine (
        .clk(clk), .rst_n(rst_n), .load_wt(load_dw_d1), .flat_wt_in(bram_dw_w_rdata), .flat_bias_in(bram_dw_b_rdata),
        .in_valid(lb_valid), .window_data(window_data), .out_valid(dw_valid_raw), .flat_out_data(dw_out_data)
    );
    wire [255:0] dw_out_relu6;
    generate
        if (DW_ACT == 1) begin : g_dw_hswish
            hswish_array #(.A_FRAC_BITS(A_FRAC_BITS)) u_dw_hswish (.clk(clk), .rst_n(rst_n),
                .in_valid(dw_valid_raw), .in_data(dw_out_data), .out_valid(dw_valid), .out_data(dw_out_relu6));
        end else begin : g_dw_noact
            assign dw_valid = dw_valid_raw;
            assign dw_out_relu6 = dw_out_data;
        end
    endgenerate
    always @(posedge clk) begin
        if (dw_state == DW_LOAD || dw_state == DW_IDLE) dw_token_cnt <= 0;
        else if (dw_valid) dw_token_cnt <= dw_token_cnt + 1;
    end
    wire [DAW-1:0] dw_waddr = (dw_chunk * N_TOKENS) + dw_token_cnt;
    wire [DAW-1:0] dw_raddr = (cl_act_ic * N_TOKENS) + cl_act_token;
    simple_dual_port_bram #(.DEPTH(DW_WORDS), .ADDR_WIDTH(DAW)) inst_sram_dw (.clk(clk), .we(dw_valid && dw_waddr < DW_WORDS),
        .waddr(dw_waddr), .wdata(dw_out_relu6), .re(1'b1), .raddr(dw_raddr), .rdata(sram_dw_rdata));

    wire pw_we = cl_out_valid && (layer == L_P3);
    banked_sram #(.K(K), .DEPTH(ACT_WORDS), .AW(AW)) inst_sram_pw_raw (.clk(clk), .we(pw_we),
        .waddr(cl_out_addr16), .wdata(cl_out_data), .raddr(addr_cnt[AW-1:0]), .rdata(sram_pw_raw_rdata), .raddr2({AW{1'b0}}), .rdata2());

    // external DW weight BRAM: MS branch while it is active, DW pass otherwise (never both)
    assign bram_x_addr = (state == S_LOAD_X) ? ({16'd0, addr_cnt} << 5) : 32'd0;
    assign bram_dw_w_addr = ms_active ? ({16'd0, 16384 + (ms_chunk * 25) + ms_wt_cnt} << 5) : ({16'd0, (dw_chunk * 25) + dw_wt_cnt} << 5);
    assign bram_dw_b_addr = ms_active ? ({16'd0, 1024 + ms_chunk} << 5) : ({16'd0, dw_chunk} << 5);

    // ------------------------------------------------------------------ P4
    wire [255:0] add_res_pw;
    adder_array u_add_pw_res (.a(mid_B), .b(sram_pw_raw_rdata), .c(add_res_pw));
    wire heartbeat_pulse = (state == S_LOAD_X && addr_cnt == 16'd0);
    assign bram_y_addr  = heartbeat_pulse ? ((ACT_WORDS-1) << 5) : ({16'd0, addr_cnt_d1} << 5);
    assign bram_y_wdata = heartbeat_pulse ? {16{16'hFACE}}  : add_res_pw;
    assign bram_y_we    = heartbeat_pulse ? 32'hFFFFFFFF    : {32{state_p4_res_add_d1}};

endmodule

// ============================================================================
// helpers
// ============================================================================
module adder_array (input [255:0] a, input [255:0] b, output [255:0] c);
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : add_blk
        wire signed [15:0] va = a[i*16 +: 16];
        wire signed [15:0] vb = b[i*16 +: 16];
        wire signed [16:0] s  = va + vb;
        assign c[i*16 +: 16] = (s > 32767) ? 16'd32767 : (s < -32768) ? -16'd32768 : s[15:0];
    end endgenerate
endmodule

module hswish_array #(parameter A_FRAC_BITS = 5) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    input  wire [255:0] in_data,
    output wire         out_valid,
    output wire [255:0] out_data
);
    localparam PW = 2 * A_FRAC_BITS + 6;
    localparam signed [16:0] THREE = 3 << A_FRAC_BITS;
    localparam signed [16:0] SIX   = 6 << A_FRAC_BITS;
    localparam signed [19:0] KC    = (1 << 24) / (6 << A_FRAC_BITS);
    localparam signed [47:0] ROUND = 48'sd1 <<< 23;
    reg [2:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= 0;
        else valid_pipe <= {valid_pipe[1:0], in_valid};
    end
    assign out_valid = valid_pipe[2];
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : hs_blk
        reg  signed [15:0]   x0, x1, y2;
        reg  signed [PW-1:0] p1;
        reg                  pass1;
        wire signed [16:0] xp3  = x0 + THREE;
        wire signed [16:0] t    = (xp3 < 0) ? 17'sd0 : (xp3 > SIX) ? SIX : xp3;
        wire signed [32:0] xt   = x0 * t;
        wire signed [47:0] prod = p1 * KC;
        wire signed [47:0] rnd  = prod + ROUND;
        wire signed [47:0] quo  = rnd >>> 24;
        always @(posedge clk) begin
            x0    <= in_data[i*16 +: 16];
            x1    <= x0;
            p1    <= xt[PW-1:0];
            pass1 <= (xp3 >= SIX);
            y2    <= pass1 ? x1 : quo[15:0];
        end
        assign out_data[i*16 +: 16] = y2;
    end endgenerate
endmodule

module simple_dual_port_bram #(parameter DATA_WIDTH = 256, parameter DEPTH = 1024, parameter ADDR_WIDTH = 10) (
    input  wire                  clk,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [DATA_WIDTH-1:0] wdata,
    input  wire                  re,
    input  wire [ADDR_WIDTH-1:0] raddr,
    output reg  [DATA_WIDTH-1:0] rdata
);
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
    integer j;
    initial begin
        for (j = 0; j < DEPTH; j = j + 1) ram[j] = 0;
    end
    always @(posedge clk) begin if (we) ram[waddr] <= wdata; end
    always @(posedge clk) begin if (re) rdata <= ram[raddr]; end
endmodule

module true_dual_port_bram #(parameter DATA_WIDTH = 256, parameter DEPTH = 1024, parameter ADDR_WIDTH = 10) (
    input  wire                  clk,
    input  wire                  we_a,
    input  wire [ADDR_WIDTH-1:0] addr_a,
    input  wire [DATA_WIDTH-1:0] din_a,
    output reg  [DATA_WIDTH-1:0] dout_a,
    input  wire                  we_b,
    input  wire [ADDR_WIDTH-1:0] addr_b,
    input  wire [DATA_WIDTH-1:0] din_b,
    output reg  [DATA_WIDTH-1:0] dout_b
);
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
    integer j;
    initial begin
        for (j = 0; j < DEPTH; j = j + 1) ram[j] = 0;
    end
    always @(posedge clk) begin
        if (we_a) ram[addr_a] <= din_a;
        dout_a <= ram[addr_a];
    end
    always @(posedge clk) begin
        if (we_b) ram[addr_b] <= din_b;
        dout_b <= ram[addr_b];
    end
endmodule
