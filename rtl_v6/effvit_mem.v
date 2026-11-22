`timescale 1ns/1ps

// ============================================================================
// Memories of the v6 accelerator (the v5 memories; the banked SRAM takes a word width). One clock everywhere. No reset on any memory
// output and no initial contents: an activation or weight word that is read
// before it was written shows up as X in simulation.
// ============================================================================

// ---------------------------------------------------------------- simple dual port block RAM
// Read latency 1 + OREG (OREG = 0, 1 or 2 extra output registers). WREG = 1 registers the write port once
// (the word is in the memory one cycle later).
module sdp_bram #(parameter DATA_WIDTH = 256, parameter DEPTH = 1024, parameter ADDR_WIDTH = 10,
                  parameter OREG = 0, parameter WREG = 0) (
    input  wire                  clk,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [DATA_WIDTH-1:0] wdata,
    input  wire [ADDR_WIDTH-1:0] raddr,
    output wire [DATA_WIDTH-1:0] rdata
);
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];
    reg                  we_r;
    reg [ADDR_WIDTH-1:0] wa_r;
    reg [DATA_WIDTH-1:0] wd_r, q0, q1, q2;
    wire                  we_i = (WREG != 0) ? we_r : we;
    wire [ADDR_WIDTH-1:0] wa_i = (WREG != 0) ? wa_r : waddr;
    wire [DATA_WIDTH-1:0] wd_i = (WREG != 0) ? wd_r : wdata;
    always @(posedge clk) begin we_r <= we; wa_r <= waddr; wd_r <= wdata; end
    always @(posedge clk) begin
        if (we_i) ram[wa_i] <= wd_i;
        q0 <= ram[raddr];
        q1 <= q0;
        q2 <= q1;
    end
    assign rdata = (OREG == 0) ? q0 : (OREG == 1) ? q1 : q2;
endmodule

// ---------------------------------------------------------------- true dual port block RAM, read latency 1
module tdp_bram #(parameter DATA_WIDTH = 256, parameter DEPTH = 1024, parameter ADDR_WIDTH = 10) (
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
    always @(posedge clk) begin
        if (we_a) ram[addr_a] <= din_a;
        dout_a <= ram[addr_a];
    end
    always @(posedge clk) begin
        if (we_b) ram[addr_b] <= din_b;
        dout_b <= ram[addr_b];
    end
endmodule

// ---------------------------------------------------------------- SRAM with a K-word write port and two 1-word read ports
// word address = token*STRIDE + chunk; a write stores K consecutive chunks (waddr multiple of K). Read latency 1.
// The write port is registered once per bank: a word is in the memory one cycle after we.
module banked_sram #(
    parameter K = 8,
    parameter W = 256,               // word width
    parameter DEPTH = 4096,          // total words
    parameter AW = 12
)(
    input  wire             clk,
    input  wire             we,
    input  wire [AW-1:0]    waddr,
    input  wire [K*W-1:0]   wdata,
    input  wire [AW-1:0]    raddr,
    output wire [W-1:0]     rdata,
    input  wire [AW-1:0]    raddr2,
    output wire [W-1:0]     rdata2
);
    localparam KB = $clog2(K);
    localparam RD = DEPTH / K;
    reg [KB-1:0] sel_d1, sel2_d1;
    wire [K*W-1:0] rq, rq2;
    genvar k;
    generate
        for (k = 0; k < K; k = k + 1) begin : bank
            (* ram_style = "block" *) reg [W-1:0] mem [0:RD-1];
            reg [W-1:0] q, q2, wd_r;
            reg            we_r;
            reg [AW-1:0]   wa_r;
            always @(posedge clk) begin we_r <= we; wa_r <= waddr >> KB; wd_r <= wdata[k*W +: W]; end
            always @(posedge clk) begin
                if (we_r) mem[wa_r] <= wd_r;
                q  <= mem[raddr >> KB];
                q2 <= mem[raddr2 >> KB];
            end
            assign rq[k*W +: W]  = q;
            assign rq2[k*W +: W] = q2;
        end
    endgenerate
    always @(posedge clk) begin sel_d1 <= raddr[KB-1:0]; sel2_d1 <= raddr2[KB-1:0]; end
    assign rdata  = rq[sel_d1*W +: W];
    assign rdata2 = rq2[sel2_d1*W +: W];
endmodule

// ---------------------------------------------------------------- one UltraRAM page: 4096 x 256 (4 URAM288), read latency 2
module uram_page (
    input  wire         clk,
    input  wire         we,
    input  wire [11:0]  waddr,
    input  wire [255:0] wdata,
    input  wire [11:0]  raddr,
    output reg  [255:0] rdata
);
    (* ram_style = "ultra" *) reg [255:0] mem [0:4095];
    reg [255:0] q0;
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        q0    <= mem[raddr];
        rdata <= q0;                 // URAM output register
    end
endmodule

// ---------------------------------------------------------------- big weight store: NPAGE UltraRAM pages, one write port, one read port
// The weights of all four blocks stay here (4 x 7 pages in v6). Read latency WBIG_LAT = 6:
//   raddr -> a0 (global) -> a1 (per page group) -> URAM read -> URAM output reg -> group mux reg -> final mux reg
module wbig_mem #(parameter NPAGE = 28) (
    input  wire         clk,
    input  wire         we,
    input  wire [17:0]  waddr,
    input  wire [255:0] wdata,
    input  wire [17:0]  raddr,
    output reg  [255:0] rdata
);
    localparam NGRP = (NPAGE + 7) / 8;
    reg         we0;
    reg [17:0]  wa0, ra0;
    reg [255:0] wd0;
    always @(posedge clk) begin we0 <= we; wa0 <= waddr; wd0 <= wdata; ra0 <= raddr; end

    reg [5:0] ps2, ps3, ps4;                      // page of the read address, aligned with the page outputs (ps4) and the group mux (ps4[5:3] one later)
    reg [2:0] gs5;
    always @(posedge clk) begin ps2 <= ra0[17:12]; ps3 <= ps2; ps4 <= ps3; gs5 <= ps4[5:3]; end

    wire [255:0] pq [0:NGRP*8-1];
    reg  [255:0] m1 [0:NGRP-1];
    genvar g, p;
    generate
        for (g = 0; g < NGRP; g = g + 1) begin : grp
            (* dont_touch = "true" *) reg         we1;
            (* dont_touch = "true" *) reg [17:0]  wa1;
            (* dont_touch = "true" *) reg [11:0]  ra1;
            (* dont_touch = "true" *) reg [255:0] wd1;
            always @(posedge clk) begin we1 <= we0; wa1 <= wa0; wd1 <= wd0; ra1 <= ra0[11:0]; end
            for (p = 0; p < 8; p = p + 1) begin : pg
                if (g*8 + p < NPAGE) begin : pg_real
                    uram_page u_page (.clk(clk), .we(we1 && (wa1[17:12] == g*8 + p)), .waddr(wa1[11:0]), .wdata(wd1),
                                      .raddr(ra1), .rdata(pq[g*8 + p]));
                end else begin : pg_none
                    assign pq[g*8 + p] = 256'd0;
                end
            end
            always @(posedge clk) m1[g] <= pq[g*8 + ps4[2:0]];
        end
    endgenerate
    always @(posedge clk) rdata <= m1[gs5];
endmodule
