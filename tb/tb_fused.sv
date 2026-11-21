`timescale 1ns/1ps

module tb_accelerator_top();

    // ==========================================
    // 參數定義 (對齊你的 Top Module)
    // ==========================================
    parameter ROWS          = 16;
    parameter COLS          = 16;
    parameter N_TOKENS      = 256;    // 512x512 input -> Stage 4 = 16x16 tokens
    parameter IMG_SIZE      = 16;
    parameter C_IN_CHUNKS   = 16;  
    parameter C_OUT_CHUNKS  = 64; 
    parameter FRAC_BITS     = 11;
    parameter A_FRAC_BITS   = 5;
    parameter EXPAND_ACT    = 1;      // 1 = hswish, 0 = ReLU6（舊行為）
    parameter DW_ACT        = 1;      // 1 = hswish, 0 = 無（舊行為；5/24 的 y_golden 是這個）
    parameter CONV_PAR      = 8;      // K output chunks in parallel (K*16 <= N_TOKENS-16)

    // ==========================================
    // 1. Clock & Reset Generation
    // ==========================================
    reg clk;
    reg rst_n;
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz Clock (10ns period)
    end
    
    // ==========================================
    // 2. DUT 訊號宣告
    // ==========================================
    reg  reg_start;
    wire reg_done;
    wire [31:0] reg_cycle_cnt;

    wire bram_x_clk, bram_x_en;       wire [31:0] bram_x_addr;       wire [255:0] bram_x_rdata;
    wire bram_w_clk, bram_w_en;       wire [31:0] bram_w_addr;       wire [255:0] bram_w_rdata;
    wire bram_b_clk, bram_b_en;       wire [31:0] bram_b_addr;       wire [255:0] bram_b_rdata;
    wire bram_dw_w_clk, bram_dw_w_en; wire [31:0] bram_dw_w_addr;    wire [255:0] bram_dw_w_rdata;
    wire bram_dw_b_clk, bram_dw_b_en; wire [31:0] bram_dw_b_addr;    wire [255:0] bram_dw_b_rdata;
    wire bram_pw_w_clk, bram_pw_w_en; wire [31:0] bram_pw_w_addr;    wire [255:0] bram_pw_w_rdata;
    wire bram_pw_b_clk, bram_pw_b_en; wire [31:0] bram_pw_b_addr;    wire [255:0] bram_pw_b_rdata;
    
    // 🔥 [新增] MS PW BRAM 端口
    wire bram_ms_pw_w_clk, bram_ms_pw_w_en; wire [31:0] bram_ms_pw_w_addr; wire [255:0] bram_ms_pw_w_rdata;
    wire bram_ms_pw_b_clk, bram_ms_pw_b_en; wire [31:0] bram_ms_pw_b_addr; wire [255:0] bram_ms_pw_b_rdata;

    wire bram_mla_w_clk, bram_mla_w_en; wire [31:0] bram_mla_w_addr; wire [255:0] bram_mla_w_rdata;
    wire bram_mla_b_clk, bram_mla_b_en; wire [31:0] bram_mla_b_addr; wire [255:0] bram_mla_b_rdata;
    
    wire bram_y_clk, bram_y_en;       wire [31:0] bram_y_addr;       wire [255:0] bram_y_wdata; wire [31:0] bram_y_we;

    // ==========================================
    // 3. DUT Instantiation
    // ==========================================
    accelerator_top #(
        .ROWS(ROWS), .COLS(COLS), .N_TOKENS(N_TOKENS), .IMG_SIZE(IMG_SIZE),
        .C_IN_CHUNKS(C_IN_CHUNKS), .C_OUT_CHUNKS(C_OUT_CHUNKS),
        .FRAC_BITS(FRAC_BITS), .A_FRAC_BITS(A_FRAC_BITS), .EXPAND_ACT(EXPAND_ACT), .DW_ACT(DW_ACT), .CONV_PAR(CONV_PAR)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .reg_start(reg_start), .reg_done(reg_done), .reg_cycle_cnt(reg_cycle_cnt),
        
        .bram_x_clk(bram_x_clk), .bram_x_en(bram_x_en), .bram_x_addr(bram_x_addr), .bram_x_rdata(bram_x_rdata),
        .bram_w_clk(bram_w_clk), .bram_w_en(bram_w_en), .bram_w_addr(bram_w_addr), .bram_w_rdata(bram_w_rdata),
        .bram_b_clk(bram_b_clk), .bram_b_en(bram_b_en), .bram_b_addr(bram_b_addr), .bram_b_rdata(bram_b_rdata),
        .bram_dw_w_clk(bram_dw_w_clk), .bram_dw_w_en(bram_dw_w_en), .bram_dw_w_addr(bram_dw_w_addr), .bram_dw_w_rdata(bram_dw_w_rdata),
        .bram_dw_b_clk(bram_dw_b_clk), .bram_dw_b_en(bram_dw_b_en), .bram_dw_b_addr(bram_dw_b_addr), .bram_dw_b_rdata(bram_dw_b_rdata),
        .bram_pw_w_clk(bram_pw_w_clk), .bram_pw_w_en(bram_pw_w_en), .bram_pw_w_addr(bram_pw_w_addr), .bram_pw_w_rdata(bram_pw_w_rdata),
        .bram_pw_b_clk(bram_pw_b_clk), .bram_pw_b_en(bram_pw_b_en), .bram_pw_b_addr(bram_pw_b_addr), .bram_pw_b_rdata(bram_pw_b_rdata),
        
        // 🔥 接上 MS PW 端口
        .bram_ms_pw_w_clk(bram_ms_pw_w_clk), .bram_ms_pw_w_en(bram_ms_pw_w_en), .bram_ms_pw_w_addr(bram_ms_pw_w_addr), .bram_ms_pw_w_rdata(bram_ms_pw_w_rdata),
        .bram_ms_pw_b_clk(bram_ms_pw_b_clk), .bram_ms_pw_b_en(bram_ms_pw_b_en), .bram_ms_pw_b_addr(bram_ms_pw_b_addr), .bram_ms_pw_b_rdata(bram_ms_pw_b_rdata),

        .bram_mla_w_clk(bram_mla_w_clk), .bram_mla_w_en(bram_mla_w_en), .bram_mla_w_addr(bram_mla_w_addr), .bram_mla_w_rdata(bram_mla_w_rdata),
        .bram_mla_b_clk(bram_mla_b_clk), .bram_mla_b_en(bram_mla_b_en), .bram_mla_b_addr(bram_mla_b_addr), .bram_mla_b_rdata(bram_mla_b_rdata),
        
        .bram_y_clk(bram_y_clk), .bram_y_en(bram_y_en), .bram_y_addr(bram_y_addr), .bram_y_wdata(bram_y_wdata), .bram_y_we(bram_y_we)
    );

    // ==========================================
    // 4. Mock BRAM Instantiations (🔥 嚴格鎖死 block_0 🔥)
    // ==========================================
    mock_external_bram #(.DEPTH(4096),  .INIT_FILE("block_0_x.hex"))     mem_x     (.clk(clk), .en(bram_x_en),      .we(32'b0),     .addr(bram_x_addr),     .wdata(256'd0),     .rdata(bram_x_rdata));
    mock_external_bram #(.DEPTH(16384), .INIT_FILE("block_0_w.hex"))     mem_w     (.clk(clk), .en(bram_w_en),      .we(32'b0),     .addr(bram_w_addr),     .wdata(256'd0),     .rdata(bram_w_rdata));
    mock_external_bram #(.DEPTH(2048),  .INIT_FILE("block_0_b.hex"))     mem_b     (.clk(clk), .en(bram_b_en),      .we(32'b0),     .addr(bram_b_addr),     .wdata(256'd0),     .rdata(bram_b_rdata));
    
    // MS DW 共用原本 DW 端口，但我們在 Python 有 padding
    mock_external_bram #(.DEPTH(32768), .INIT_FILE("block_0_dw_w.hex"))  mem_dw_w  (.clk(clk), .en(bram_dw_w_en),   .we(32'b0),     .addr(bram_dw_w_addr),  .wdata(256'd0),     .rdata(bram_dw_w_rdata));
    mock_external_bram #(.DEPTH(2048),  .INIT_FILE("block_0_dw_b.hex"))  mem_dw_b  (.clk(clk), .en(bram_dw_b_en),   .we(32'b0),     .addr(bram_dw_b_addr),  .wdata(256'd0),     .rdata(bram_dw_b_rdata));
    
    mock_external_bram #(.DEPTH(16384), .INIT_FILE("block_0_pw_w.hex"))  mem_pw_w  (.clk(clk), .en(bram_pw_w_en),   .we(32'b0),     .addr(bram_pw_w_addr),  .wdata(256'd0),     .rdata(bram_pw_w_rdata));
    mock_external_bram #(.DEPTH(2048),  .INIT_FILE("block_0_pw_b.hex"))  mem_pw_b  (.clk(clk), .en(bram_pw_b_en),   .we(32'b0),     .addr(bram_pw_b_addr),  .wdata(256'd0),     .rdata(bram_pw_b_rdata));
    
    // 🔥 [新增] MS PW 的獨立 BRAM
    mock_external_bram #(.DEPTH(1024),  .INIT_FILE("block_0_ms_pw_w.hex")) mem_ms_pw_w (.clk(clk), .en(bram_ms_pw_w_en), .we(32'b0),     .addr(bram_ms_pw_w_addr), .wdata(256'd0),     .rdata(bram_ms_pw_w_rdata));
    mock_external_bram #(.DEPTH(1024),  .INIT_FILE("block_0_ms_pw_b.hex")) mem_ms_pw_b (.clk(clk), .en(bram_ms_pw_b_en), .we(32'b0),     .addr(bram_ms_pw_b_addr), .wdata(256'd0),     .rdata(bram_ms_pw_b_rdata));

    mock_external_bram #(.DEPTH(32768), .INIT_FILE("block_0_mla_w.hex")) mem_mla_w (.clk(clk), .en(bram_mla_w_en),  .we(32'b0),     .addr(bram_mla_w_addr), .wdata(256'd0),     .rdata(bram_mla_w_rdata));
    mock_external_bram #(.DEPTH(1024),  .INIT_FILE("block_0_mla_b.hex")) mem_mla_b (.clk(clk), .en(bram_mla_b_en),  .we(32'b0),     .addr(bram_mla_b_addr), .wdata(256'd0),     .rdata(bram_mla_b_rdata));
    
    // 注意：這裡將 32-bit 的 bram_y_we 完整接進去，讓 Mock BRAM 能做位元組遮罩寫入
    mock_external_bram #(.DEPTH(4096),  .INIT_FILE("")) mem_y (.clk(clk), .en(bram_y_en), .we(bram_y_we), .addr(bram_y_addr), .wdata(bram_y_wdata), .rdata());
    
    // BRAM_Y depth follows the token count
    localparam Y_DEPTH = N_TOKENS * C_IN_CHUNKS;   // 256 tokens x 16 chunks = 4096 words
    reg [255:0] golden_y [0:Y_DEPTH-1];

    // ==========================================
    // 5. Test sequence: +SEQ=<block digits>, e.g. +SEQ=01 runs block 0 then block 1 with NO reset in between
    //    (what the board does). Default "0". The mock BRAMs are reloaded from block_<n>_*.hex before each run.
    // ==========================================
    integer i, ch, run, nruns;
    integer err_cnt;
    integer d, abs_max;
    longint abs_sum;
    reg signed [15:0] hw_v, gd_v;
    string seq;
    reg dump_en;                                   // spy monitors write files only while this is 1
    integer blk;

    task load_block(input integer b);
        $readmemh($sformatf("block_%0d_x.hex", b),        mem_x.mem);
        $readmemh($sformatf("block_%0d_w.hex", b),        mem_w.mem);
        $readmemh($sformatf("block_%0d_b.hex", b),        mem_b.mem);
        $readmemh($sformatf("block_%0d_dw_w.hex", b),     mem_dw_w.mem);
        $readmemh($sformatf("block_%0d_dw_b.hex", b),     mem_dw_b.mem);
        $readmemh($sformatf("block_%0d_pw_w.hex", b),     mem_pw_w.mem);
        $readmemh($sformatf("block_%0d_pw_b.hex", b),     mem_pw_b.mem);
        $readmemh($sformatf("block_%0d_ms_pw_w.hex", b),  mem_ms_pw_w.mem);
        $readmemh($sformatf("block_%0d_ms_pw_b.hex", b),  mem_ms_pw_b.mem);
        $readmemh($sformatf("block_%0d_mla_w.hex", b),    mem_mla_w.mem);
        $readmemh($sformatf("block_%0d_mla_b.hex", b),    mem_mla_b.mem);
        $readmemh($sformatf("block_%0d_y_golden.hex", b), golden_y);
        for (i = 0; i < 4096; i = i + 1) mem_y.mem[i] = 256'd0;
    endtask

    initial begin
        rst_n = 0; reg_start = 0; dump_en = 0;
        if (!$value$plusargs("SEQ=%s", seq)) seq = "0";
        nruns = seq.len();
        #100; rst_n = 1; #50;

        for (run = 0; run < nruns; run = run + 1) begin
            blk = seq[run] - "0";
            load_block(blk);
            dump_en = (run == nruns - 1);
            $display("=================================================");
            $display(" [TB] RUN %0d / %0d : block %0d   (BRAM X[0] = %64x)", run + 1, nruns, blk, mem_x.mem[0]);
            $display("=================================================");
            reg_start = 1; #10; reg_start = 0;
            wait(reg_done == 1'b1);
            $display(" [TB] block %0d finished in %0d cycles.", blk, reg_cycle_cnt);

            err_cnt = 0; abs_sum = 0; abs_max = 0;
            for (i = 0; i < Y_DEPTH; i = i + 1) begin
                if (mem_y.mem[i] !== golden_y[i]) begin
                    err_cnt = err_cnt + 1;
                    if (err_cnt <= 4) $display(" [DIFF] Addr: %04d | HW: %64h | Golden: %64h", i, mem_y.mem[i], golden_y[i]);
                end
                for (ch = 0; ch < 16; ch = ch + 1) begin
                    hw_v = mem_y.mem[i][ch*16 +: 16];
                    gd_v = golden_y[i][ch*16 +: 16];
                    d = hw_v - gd_v;
                    if (d < 0) d = -d;
                    abs_sum = abs_sum + d;
                    if (d > abs_max) abs_max = d;
                end
            end
            $display(" [TB] block %0d: words differing %0d / %0d, mean |HW - golden| = %0d.%02d LSB, max = %0d LSB",
                     blk, err_cnt, Y_DEPTH, abs_sum / (Y_DEPTH*16), (abs_sum * 100 / (Y_DEPTH*16)) % 100, abs_max);
            // wait until the FSM is back in IDLE (host clears start, then waits before the next run)
            #200;
        end
        $display("=================================================");
        $display(" [TB] SEQUENCE %s DONE", seq);
        $display("=================================================");
        #100;
        $finish;
    end

    // ==========================================
    // Spy monitors. Cluster outputs (M1 q/k/v, M3 proj, P3 pw) are written as K lines per token:
    // file order [group][token][k], chunk = group*K + k  ("cluster" layout in rtl_model.py).
    // MS / attention / DW dumps stay [chunk][token].
    // ==========================================
    integer f_q, f_k, f_v, f_h1, f_h2, f_ms_pw, f_proj, f_p2, f_p3, kk;
    localparam KPAR = CONV_PAR;

    initial begin
        f_q     = $fopen("hw_m1_q.txt", "w");
        f_k     = $fopen("hw_m1_k.txt", "w");
        f_v     = $fopen("hw_m1_v.txt", "w");
        f_h1    = $fopen("hw_m2_h1.txt", "w");
        f_h2    = $fopen("hw_m2_h2.txt", "w");
        f_ms_pw = $fopen("hw_ms_pw_mid.txt", "w");
        f_proj  = $fopen("hw_m3_proj.txt", "w");
        f_p2    = $fopen("hw_p2_dw.txt", "w");
        f_p3    = $fopen("hw_p3_pw.txt", "w");

        forever @(posedge clk) if (dump_en) begin
            if (u_dut.cl_out_valid) begin
                for (kk = 0; kk < KPAR; kk = kk + 1) begin
                    case (u_dut.layer)
                        2'd0: begin
                            if (u_dut.cl_out_oc_base < 16)       $fdisplay(f_q, "%064x", u_dut.cl_out_data[kk*256 +: 256]);
                            else if (u_dut.cl_out_oc_base < 32)  $fdisplay(f_k, "%064x", u_dut.cl_out_data[kk*256 +: 256]);
                            else                                 $fdisplay(f_v, "%064x", u_dut.cl_out_data[kk*256 +: 256]);
                        end
                        2'd1: $fdisplay(f_proj, "%064x", u_dut.cl_out_data[kk*256 +: 256]);
                        2'd3: $fdisplay(f_p3,   "%064x", u_dut.cl_out_data[kk*256 +: 256]);
                        default: ;
                    endcase
                end
            end
            if (u_dut.ms_pw_buf_valid) $fdisplay(f_ms_pw, "%064x", u_dut.ms_pw_buf_data);
            if (u_dut.mla_out_valid) begin
                if (!u_dut.att_src) $fdisplay(f_h1, "%064x", u_dut.mla_out_data);
                else                   $fdisplay(f_h2, "%064x", u_dut.mla_out_data);
            end
            if (u_dut.dw_valid) $fdisplay(f_p2, "%064x", u_dut.dw_out_relu6);
        end
    end
endmodule

// ==========================================
// 6. Mock 外部 BRAM 模型 (支援 32-bit Byte Enable)
// ==========================================
module mock_external_bram #(
    parameter DEPTH = 1024,
    parameter INIT_FILE = ""
)(
    input  wire         clk,
    input  wire         en,
    input  wire [31:0]  we,     // 改為 32-bit 支援 Byte Write Enable
    input  wire [31:0]  addr,
    input  wire [255:0] wdata,
    output reg  [255:0] rdata
);
    reg [255:0] mem [0:DEPTH-1];
    integer j;
    
    initial begin
        for (j = 0; j < DEPTH; j = j + 1) mem[j] = 256'd0;
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end
    
    wire [31:0] word_idx = addr >> 5;
    integer byte_idx;

    always @(posedge clk) begin
        if (en) begin
            // 處理寫入：支援 Byte Enable
            for (byte_idx = 0; byte_idx < 32; byte_idx = byte_idx + 1) begin
                if (we[byte_idx]) begin
                    mem[word_idx][byte_idx*8 +: 8] <= wdata[byte_idx*8 +: 8];
                end
            end
            // 處理讀取 (1-cycle latency)
            rdata <= mem[word_idx];
        end
    end
endmodule