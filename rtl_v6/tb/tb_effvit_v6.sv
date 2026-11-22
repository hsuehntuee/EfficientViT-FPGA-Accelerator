`timescale 1ns/1ps

// ============================================================================
// Testbench of effvit_top (v6, INT8). It drives the two ports the board has, and only
// those: the AXI4-Lite registers and the host memory port (what axi_bram_ctrl
// drives). Nothing is loaded through the back door.
//
//   1. write the weights and parameters of blocks 0..3 through the host memory port
//      (hex files of int8/evit_int8.py export, mapped as in effvit_sys.v)
//   2. read part of them back (every small memory in full, 64 words of every big array)
//   3. run the sequence +SEQ=<chars>, no reset in between:
//        0..3  one block: x = block_N_x, result against block_N_y_golden
//        c     blocks 0..3 from one start: x = block_0_x, result against block_3_y_golden
//        p     blocks 1..2 from one start: x = block_1_x, result against block_2_y_golden
//      default "c011332021cp"
//   Every block end is also checked inside the design (x memory against that block's golden).
//
//   +DUMP=1 writes what every stage produced during the last run (hw_cl.txt, hw_ms.txt, hw_att.txt, hw_dw.txt)
//   for int8/check_dumps.py. +HEX=<dir> selects the vector directory (default "."). +PROG=<n> prints the cycle
//   counter every n status polls.
// Last line: TB_PASS or TB_FAIL.
// ============================================================================
module tb_effvit_v6;
    localparam HRL = 8;                       // host read latency
    localparam CLK_HALF = 2.5;                // 200 MHz

    reg clk = 0, rst_n = 0;
    always #(CLK_HALF) clk = ~clk;

    // ---------------------------------------------------------------- DUT
    reg          hm_en = 0;
    reg  [31:0]  hm_we = 0;
    reg  [23:0]  hm_addr = 0;
    reg  [255:0] hm_wdata = 0;
    wire [255:0] hm_rdata;
    reg  [5:0]   awaddr = 0, araddr = 0;
    reg          awvalid = 0, wvalid = 0, arvalid = 0;
    reg  [31:0]  wdata = 0;
    wire         awready, wready, bvalid, arready, rvalid;
    wire [31:0]  rdata;
    wire [1:0]   bresp, rresp;

`ifdef NETLIST
    design_1_effvit_0 dut (               // post-synthesis functional netlist of the block design cell (same ports)
`else
    effvit_top dut (
`endif
        .aclk(clk), .aresetn(rst_n),
        .hm_en(hm_en), .hm_we(hm_we), .hm_addr(hm_addr), .hm_wdata(hm_wdata), .hm_rdata(hm_rdata), .hm_clk(clk), .hm_rst(1'b0),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(1'b1)
    );

    // ---------------------------------------------------------------- bus functional models
    task automatic axil_write(input [5:0] a, input [31:0] d);
        begin
            @(posedge clk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
            forever begin @(posedge clk); if (awready && wready) break; end
            awvalid <= 0; wvalid <= 0;
            forever begin @(posedge clk); if (bvalid) break; end
        end
    endtask
    task automatic axil_read(input [5:0] a, output [31:0] d);
        begin
            @(posedge clk); araddr <= a; arvalid <= 1;
            forever begin @(posedge clk); if (arready) break; end
            arvalid <= 0;
            forever begin @(posedge clk); if (rvalid) begin d = rdata; break; end end
        end
    endtask

    reg [255:0] buf_a [0:32767];              // data to write / data read
    // write n words buf_a[0..n-1] to word address base.., one word per cycle
    task automatic hm_write_block(input [18:0] base, input integer n);
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(posedge clk); hm_en <= 1; hm_we <= 32'hFFFF_FFFF; hm_addr <= {base + i[18:0], 5'd0}; hm_wdata <= buf_a[i];
            end
            @(posedge clk); hm_en <= 0; hm_we <= 0;
        end
    endtask
    // read n words from word address base.. into buf_a[0..n-1]: one address per cycle, data HRL cycles later
    task automatic hm_read_block(input [18:0] base, input integer n);
        integer i;
        begin
            for (i = 0; i < n + HRL; i = i + 1) begin
                @(posedge clk);
                if (i < n) begin hm_en <= 1; hm_we <= 0; hm_addr <= {base + i[18:0], 5'd0}; end
                else begin hm_en <= 0; end
                if (i >= HRL) begin #0.1; buf_a[i - HRL] = hm_rdata; end
            end
        end
    endtask

    // ---------------------------------------------------------------- vectors
    string hexdir, seq;
    reg [255:0] f_mem [0:32767];
    reg [255:0] gold [0:3][0:4095];
    reg [255:0] xin  [0:3][0:4095];
    integer total_err = 0, rb_err = 0, snoop_err = 0, snoop_cnt = 0;
    integer dump_on = 0, nload = 4, prog = 0;
    reg dump_en = 0;

    task automatic load_file(input string name, input integer b, input integer n);
        integer i;
        begin
            for (i = 0; i < 32768; i = i + 1) f_mem[i] = 'x;
            $readmemh($sformatf("%s/block_%0d_%s.hex", hexdir, b, name), f_mem);
            if (^f_mem[n-1] === 1'bx) begin $display(" [TB] ERROR: %s/block_%0d_%s.hex has fewer than %0d words", hexdir, b, name, n); total_err = total_err + 1; end
        end
    endtask

    // file words [src .. src+n-1] -> host word address dst, then (when chk > 0) read the first and last chk words back
    task automatic put_region(input string name, input integer b, input integer src, input integer n, input [18:0] dst, input integer chk);
        integer i, m;
        begin
            for (i = 0; i < n; i = i + 1) buf_a[i] = f_mem[src + i];
            hm_write_block(dst, n);
            if (chk > 0) begin
                m = (chk >= n) ? n : chk;
                hm_read_block(dst, m);
                for (i = 0; i < m; i = i + 1) if (buf_a[i] !== f_mem[src + i]) begin
                    rb_err = rb_err + 1;
                    if (rb_err <= 8) $display(" [RB] %s block %0d word %0d: read %064x, wrote %064x", name, b, i, buf_a[i], f_mem[src + i]);
                end
                if (m < n) begin
                    hm_read_block(dst + n - m, m);
                    for (i = 0; i < m; i = i + 1) if (buf_a[i] !== f_mem[src + n - m + i]) begin
                        rb_err = rb_err + 1;
                        if (rb_err <= 8) $display(" [RB] %s block %0d word %0d: read %064x, wrote %064x", name, b, n - m + i, buf_a[i], f_mem[src + n - m + i]);
                    end
                end
            end
        end
    endtask

    task automatic load_weights(input integer b);
        begin
            load_file("wbig", b, 26624);    put_region("wbig", b, 0, 26624, 19'h00000 + b*28672,           64);
            load_file("dww", b, 3248);      put_region("dww", b, 0, 1600, 19'h40000 + b*4096,              1600);
                                            put_region("dww(ms)", b, 2048, 1200, 19'h40000 + b*4096 + 2048, 1200);
            load_file("mspww", b, 768);     put_region("mspww", b, 0, 768, 19'h44000 + b*1024,             768);
            load_file("dwb", b, 224);       put_region("dwb", b, 0, 224, 19'h45000 + b*256,                224);
            load_file("mspwb", b, 96);      put_region("mspwb", b, 0, 96, 19'h46000 + b*128,               96);
            load_file("clb", b, 384);       put_region("clb", b, 0, 384, 19'h47000 + b*512,                384);
            $display(" [TB] block %0d weights written (read-back mismatches so far: %0d)   t = %0t", b, rb_err, $time);
        end
    endtask

    // ---------------------------------------------------------------- one run
    task automatic run(input integer first, input integer nblk, input integer xblk);
        integer i, ch, d, err_cnt, abs_max, polls, last;
        longint abs_sum;
        reg [31:0] st, cyc, c0, c1, c2, c3;
        reg signed [15:0] hw_v, gd_v;
        begin
            last = first + nblk - 1;
            for (i = 0; i < 4096; i = i + 1) buf_a[i] = xin[xblk][i];
            hm_write_block(19'h48000, 4096);
            axil_write(6'h0C, {26'd0, nblk[1:0] - 2'd1, 2'd0, first[1:0]});
            axil_write(6'h00, 32'd1);
            polls = 0; st = 0;
            while (!st[0]) begin
                repeat (500) @(posedge clk);
                axil_read(6'h04, st);
                polls = polls + 1;
                if (prog > 0 && !st[0] && polls % prog == 0) begin
                    axil_read(6'h08, cyc);
                    $display(" [TB] poll %0d: status %08x, cycle counter %0d   t = %0t", polls, st, cyc, $time);
                end
                if (polls > 8000) begin $display(" [TB] ERROR: done never asserted (status %08x)", st); total_err = total_err + 1; $display("TB_FAIL"); $finish; end
            end
            axil_read(6'h08, cyc);
            axil_read(6'h20, c0); axil_read(6'h24, c1); axil_read(6'h28, c2); axil_read(6'h2C, c3);
            hm_read_block(19'h48000, 4096);
            axil_write(6'h00, 32'd0);
            err_cnt = 0; abs_sum = 0; abs_max = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                if (buf_a[i] !== gold[last][i]) begin
                    err_cnt = err_cnt + 1;
                    if (err_cnt <= 4) $display(" [DIFF] word %04d | HW: %064x | golden: %064x", i, buf_a[i], gold[last][i]);
                end
                for (ch = 0; ch < 16; ch = ch + 1) begin
                    hw_v = buf_a[i][ch*16 +: 16]; gd_v = gold[last][i][ch*16 +: 16];
                    d = hw_v - gd_v; if (d < 0) d = -d;
                    abs_sum = abs_sum + d; if (d > abs_max) abs_max = d;
                end
            end
            total_err = total_err + err_cnt;
            $display(" [TB] RUN blocks %0d..%0d: %0d cycles (per block end: %0d %0d %0d %0d), words differing %0d / 4096, max |HW - golden| = %0d LSB   t = %0t",
                     first, last, cyc, c0, c1, c2, c3, err_cnt, abs_max, $time);
            repeat (20) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------- check every block end inside the design
`ifndef NETLIST
    integer si, s_err;
    always @(posedge clk) if (rst_n && dut.u_sys.u_core.state == 5'd18) begin
        s_err = 0;
        for (si = 0; si < 4096; si = si + 1)
            if (dut.u_sys.u_core.inst_sram_x.ram[si] !== gold[dut.u_sys.u_core.blk][si]) s_err = s_err + 1;
        snoop_cnt = snoop_cnt + 1;
        snoop_err = snoop_err + s_err;
        $display(" [SNOOP] end of block %0d (cycle %0d): x memory differs from block_%0d_y_golden in %0d / 4096 words",
                 dut.u_sys.u_core.blk, dut.u_sys.u_core.reg_cycle_cnt, dut.u_sys.u_core.blk, s_err);
    end
`endif

    // ---------------------------------------------------------------- main
    integer b, r, k;
    reg [31:0] ver, info;
    initial begin
        if (!$value$plusargs("HEX=%s", hexdir)) hexdir = ".";
        if (!$value$plusargs("SEQ=%s", seq)) seq = "c011332021cp";
        if (!$value$plusargs("DUMP=%d", dump_on)) dump_on = 0;
        if (!$value$plusargs("PROG=%d", prog)) prog = 0;         // > 0: print the cycle counter every PROG status polls (500 cycles each)
        if (!$value$plusargs("NLOAD=%d", nload)) nload = 4;      // blocks whose weights are written (4 unless a quick single-block run)
        for (b = 0; b < 4; b = b + 1) begin
            load_file("x", b, 4096);        for (k = 0; k < 4096; k = k + 1) xin[b][k]  = f_mem[k];
            load_file("y_golden", b, 4096); for (k = 0; k < 4096; k = k + 1) gold[b][k] = f_mem[k];
        end
        repeat (20) @(posedge clk); rst_n <= 1; repeat (10) @(posedge clk);
        axil_read(6'h10, ver); axil_read(6'h14, info);
        $display(" [TB] VERSION %08x INFO %08x, sequence %s", ver, info, seq);
        for (b = 0; b < nload; b = b + 1) load_weights(b);
        $display(" [TB] weights of %0d blocks written; read-back mismatches: %0d", nload, rb_err);

        for (r = 0; r < seq.len(); r = r + 1) begin
            dump_en = (dump_on != 0) && (r == seq.len() - 1);
            k = seq[r];
            if (k == "c")      run(0, 4, 0);
            else if (k == "p") run(1, 2, 1);
            else               run(k - "0", 1, k - "0");
        end

        $display("=================================================");
        $display(" [TB] runs %0d, differing output words %0d, read-back mismatches %0d, block ends checked %0d with %0d differing words",
                 seq.len(), total_err, rb_err, snoop_cnt, snoop_err);
`ifdef NETLIST
        snoop_cnt = 1;                    // no hierarchy to look into: only the outputs read through the host port count
`endif
        if (total_err == 0 && rb_err == 0 && snoop_err == 0 && snoop_cnt > 0) $display("TB_PASS");
        else $display("TB_FAIL");
        $display("=================================================");
        #100; $finish;
    end

    // ---------------------------------------------------------------- what every stage produced during the last run
`ifndef NETLIST
    integer f_cl, f_ms, f_att, f_dw, kk;
    initial begin
        f_cl  = $fopen("hw_cl.txt", "w");      // layer chunk token data(16 x 16 bit)
        f_ms  = $fopen("hw_ms.txt", "w");      // chunk token data(16 x 8 bit)
        f_att = $fopen("hw_att.txt", "w");     // src head token data(16 x 8 bit)
        f_dw  = $fopen("hw_dw.txt", "w");      // chunk token data(16 x 8 bit)
        forever @(posedge clk) if (dump_en) begin
            if (dut.u_sys.u_core.cl_out_valid)
                for (kk = 0; kk < 16; kk = kk + 1)
                    $fdisplay(f_cl, "%0d %0d %0d %064x", dut.u_sys.u_core.layer, dut.u_sys.u_core.cl_out_oc_base + kk,
                              dut.u_sys.u_core.cl_out_token, dut.u_sys.u_core.cl_out_data[kk*256 +: 256]);
            if (dut.u_sys.u_core.ms_pw_buf_valid)
                $fdisplay(f_ms, "%0d %0d %032x", dut.u_sys.u_core.ms_chunk, dut.u_sys.u_core.ms_out_token, dut.u_sys.u_core.ms_pw_buf_data);
            if (dut.u_sys.u_core.mla_out_valid)
                $fdisplay(f_att, "%0d %0d %0d %032x", dut.u_sys.u_core.att_src, dut.u_sys.u_core.att_out_head, dut.u_sys.u_core.att_out_tok,
                          dut.u_sys.u_core.mla_out_data);
            if (dut.u_sys.u_core.dw_valid && dut.u_sys.u_core.dw_token_cnt < 256)
                $fdisplay(f_dw, "%0d %0d %032x", dut.u_sys.u_core.dw_chunk, dut.u_sys.u_core.dw_token_cnt, dut.u_sys.u_core.dw_out_act);
        end
    end
`endif
endmodule
