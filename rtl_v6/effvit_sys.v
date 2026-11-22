`timescale 1ns/1ps

// ============================================================================
// effvit_sys (v6, INT8): accel_core + the weight store of all four blocks + the host
// memory port. Everything runs on one clock.
//
// Host memory port = the BRAM port of one axi_bram_ctrl (256-bit data, byte address,
// read latency HOST_RD_LAT). Word address = hm_addr[23:5] (32-byte words):
//
//   0x00000 + b*28672 + off   big weights of block b (UltraRAM): off 0 = QKV (6144 words), 6144 = proj (4096),
//                             10240 = expand (8192), 18432 = point (8192). A word holds 16 pairs {hi', lo} of int8.
//   0x40000 + b*4096 + off    depthwise weights (16 x int8 in the low 128 bits): off 0 = MBConv DW (1600), 2048 = MS DW (1200)
//   0x44000 + b*1024          MS group 1x1 weights (768, 16 x int8)
//   0x45000 + b*256  + 2*e+h  depthwise parameters, entry e (512 bits = two words, h = 0 low half): e = chunk (MBConv DW), 64 + chunk (MS DW)
//   0x46000 + b*128  + 2*e+h  MS group 1x1 parameters, e = chunk
//   0x47000 + b*512  + 2*e+h  cluster parameters: e = 0 QKV (48 chunks), 48 proj (16), 64 expand (64), 128 point (16),
//                             144 x -> int8 shifts, 145 mid -> int8 shifts, 160 + src*16 + head attention
//   0x48000                   x memory: block input before the run, block output after it (4096 words, 16 x int16 Q10.5)
//
// A parameter entry holds 16 channels x {shift[4:0], fsel[1:0], bias'[24:0]}.
// A write stores the whole 256-bit word when any byte enable is set (the host library only
// sends whole words). Writes and reads are honoured only while the accelerator is idle or done.
// ============================================================================
module effvit_sys #(
    parameter HOST_RD_LAT = 8        // informational: must match the axi_bram_ctrl READ_LATENCY
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         reg_start,
    input  wire [1:0]   cfg_first,
    input  wire [1:0]   cfg_nm1,
    output wire         reg_done,
    output wire         mem_idle,
    output wire [31:0]  reg_cycle_cnt,
    output wire [127:0] blk_cycles,
    output wire [1:0]   cur_blk,

    input  wire         hm_en,
    input  wire [31:0]  hm_we,
    input  wire [23:0]  hm_addr,
    input  wire [255:0] hm_wdata,
    output reg  [255:0] hm_rdata
);
    localparam WBIG_LAT = 6;

    // ---------------------------------------------------------------- host side, registered once
    reg         h_wr;
    reg [18:0]  h_wa, h_ra;
    reg [255:0] h_wd;
    always @(posedge clk) begin
        if (!rst_n) h_wr <= 1'b0;
        else        h_wr <= hm_en && (|hm_we) && mem_idle;
        h_wa <= hm_addr[23:5];
        h_wd <= hm_wdata;
        h_ra <= hm_addr[23:5];
    end
    wire h_big  = !h_wa[18];
    wire [6:0] h_slot = h_wa[18:12];             // 4096-word slots above 0x40000

    // ---------------------------------------------------------------- accelerator
    wire [17:0] a_wbig_raddr;  wire [255:0] wbig_rdata;
    wire [9:0]  a_clb_raddr;   wire [511:0] clb_rdata;
    wire [13:0] a_dww_raddr;   wire [127:0] dww_rdata;
    wire [8:0]  a_dwb_raddr;   wire [511:0] dwb_rdata;
    wire [11:0] a_mspww_raddr; wire [127:0] mspww_rdata;
    wire [7:0]  a_mspwb_raddr; wire [511:0] mspwb_rdata;
    wire [255:0] x_rdata;

    accel_core #(.WBIG_LAT(WBIG_LAT)) u_core (
        .clk(clk), .rst_n(rst_n),
        .reg_start(reg_start), .cfg_first(cfg_first), .cfg_nm1(cfg_nm1),
        .reg_done(reg_done), .mem_idle(mem_idle), .reg_cycle_cnt(reg_cycle_cnt), .blk_cycles(blk_cycles), .cur_blk(cur_blk),
        .wbig_raddr(a_wbig_raddr), .wbig_rdata(wbig_rdata),
        .clb_raddr(a_clb_raddr), .clb_rdata(clb_rdata),
        .dww_raddr(a_dww_raddr), .dww_rdata(dww_rdata),
        .dwb_raddr(a_dwb_raddr), .dwb_rdata(dwb_rdata),
        .mspww_raddr(a_mspww_raddr), .mspww_rdata(mspww_rdata),
        .mspwb_raddr(a_mspwb_raddr), .mspwb_rdata(mspwb_rdata),
        .host_x_we(h_wr && (h_slot == 7'h48)), .host_x_addr(h_ra[11:0]), .host_x_wdata(h_wd), .host_x_rdata(x_rdata)
    );

    // ---------------------------------------------------------------- weight store (read ports go to the host while idle)
    // idle_q follows mem_idle by one cycle: the accelerator reads its first weight word three cycles after
    // it leaves idle, and the host reads only after it has seen done.
    reg idle_q;
    always @(posedge clk) idle_q <= mem_idle;
    wbig_mem #(.NPAGE(28)) u_wbig (.clk(clk), .we(h_wr && h_big), .waddr(h_wa[17:0]), .wdata(h_wd),
        .raddr(idle_q ? h_ra[17:0] : a_wbig_raddr), .rdata(wbig_rdata));
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(16384), .ADDR_WIDTH(14), .OREG(2)) u_dww (.clk(clk), .we(h_wr && (h_slot[6:2] == 5'h10)), .waddr(h_wa[13:0]), .wdata(h_wd[127:0]),
        .raddr(idle_q ? h_ra[13:0] : a_dww_raddr), .rdata(dww_rdata));
    sdp_bram #(.DATA_WIDTH(128), .DEPTH(4096), .ADDR_WIDTH(12)) u_mspww (.clk(clk), .we(h_wr && (h_slot == 7'h44)), .waddr(h_wa[11:0]), .wdata(h_wd[127:0]),
        .raddr(idle_q ? h_ra[11:0] : a_mspww_raddr), .rdata(mspww_rdata));
    // parameter memories: 512-bit entries, the two halves in two memories (host word address = 2 * entry + half)
    genvar hf;
    generate
        for (hf = 0; hf < 2; hf = hf + 1) begin : half
            sdp_bram #(.DEPTH(512), .ADDR_WIDTH(9)) u_dwb (.clk(clk), .we(h_wr && (h_slot == 7'h45) && (h_wa[0] == hf)), .waddr(h_wa[9:1]), .wdata(h_wd),
                .raddr(idle_q ? h_ra[9:1] : a_dwb_raddr), .rdata(dwb_rdata[hf*256 +: 256]));
            sdp_bram #(.DEPTH(256), .ADDR_WIDTH(8)) u_mspwb (.clk(clk), .we(h_wr && (h_slot == 7'h46) && (h_wa[0] == hf)), .waddr(h_wa[8:1]), .wdata(h_wd),
                .raddr(idle_q ? h_ra[8:1] : a_mspwb_raddr), .rdata(mspwb_rdata[hf*256 +: 256]));
            sdp_bram #(.DEPTH(1024), .ADDR_WIDTH(10)) u_clb (.clk(clk), .we(h_wr && (h_slot == 7'h47) && (h_wa[0] == hf)), .waddr(h_wa[10:1]), .wdata(h_wd),
                .raddr(idle_q ? h_ra[10:1] : a_clb_raddr), .rdata(clb_rdata[hf*256 +: 256]));
        end
    endgenerate

    // ---------------------------------------------------------------- host read-back, uniform latency
    // cycle 0 hm_addr | 1 h_ra | 2 small memories and x out | 3 small mux | 4 depthwise weights out (two output registers) |
    // 5 depthwise weights merged | 6 delay | 7 big weights out, small delayed | 8 hm_rdata
    reg [18:0]  ra_d2;
    reg [255:0] small_q3, small_q4, small_q5, small_q6, small_q7;
    reg         big_s2, big_s3, big_s4, big_s5, big_s6, big_s7, dww_s3, dww_s4;
    wire [6:0]  slot_d2 = ra_d2[18:12];
    always @(posedge clk) begin
        ra_d2 <= h_ra;
        small_q3 <= (slot_d2 == 7'h44) ? {128'd0, mspww_rdata} :
                    (slot_d2 == 7'h45) ? (ra_d2[0] ? dwb_rdata[511:256]   : dwb_rdata[255:0]) :
                    (slot_d2 == 7'h46) ? (ra_d2[0] ? mspwb_rdata[511:256] : mspwb_rdata[255:0]) :
                    (slot_d2 == 7'h47) ? (ra_d2[0] ? clb_rdata[511:256]   : clb_rdata[255:0]) :
                    (slot_d2 == 7'h48) ? x_rdata : 256'd0;
        dww_s3 <= (slot_d2[6:2] == 5'h10); dww_s4 <= dww_s3;
        small_q4 <= small_q3;
        small_q5 <= dww_s4 ? {128'd0, dww_rdata} : small_q4;
        small_q6 <= small_q5; small_q7 <= small_q6;
        big_s2 <= !h_ra[18]; big_s3 <= big_s2; big_s4 <= big_s3; big_s5 <= big_s4; big_s6 <= big_s5; big_s7 <= big_s6;
        hm_rdata <= big_s7 ? wbig_rdata : small_q7;         // wbig_rdata is the output register of the store (cycle 7)
    end
endmodule
