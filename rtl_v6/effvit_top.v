`timescale 1 ns / 1 ps

// ============================================================================
// effvit_top (v6): block-design top of the accelerator (used as a module reference).
//   s_axi   AXI4-Lite control registers
//   hm_*    host memory port, connected to the BRAM port of one axi_bram_ctrl
//           (256-bit data, read latency 8); see effvit_sys.v for the address map
// One clock (aclk) and one synchronous active-low reset (aresetn) for everything.
//
// Registers (32-bit, offset):
//   0x00 CTRL      [0] start: write 1 to start, write 0 after done to return to idle
//   0x04 STATUS    [0] done, [1] busy (memories not accessible), [5:4] current block
//   0x08 CYCLES    clock cycles from start to done
//   0x0C CFG       [1:0] first block, [5:4] number of blocks - 1 (weights of all four blocks stay loaded)
//   0x10 VERSION   0x0600_0001
//   0x14 INFO      [7:0] host read latency, [15:8] parallel arrays K, [31:16] tokens
//   0x20..0x2C     CYCLES at the end of the 1st .. 4th block of the last run
// ============================================================================
module effvit_top #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:HM, ASSOCIATED_RESET aresetn" *)
    input  wire         aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM EN" *)
    (* X_INTERFACE_PARAMETER = "MODE Slave, MASTER_TYPE BRAM_CTRL, MEM_SIZE 16777216, MEM_WIDTH 256, MEM_ECC NONE, READ_WRITE_MODE READ_WRITE, READ_LATENCY 8" *)
    input  wire         hm_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM WE" *)
    input  wire [31:0]  hm_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM ADDR" *)
    input  wire [23:0]  hm_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM DIN" *)
    input  wire [255:0] hm_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM DOUT" *)
    output wire [255:0] hm_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM CLK" *)
    input  wire         hm_clk,          // same clock as aclk (not used inside)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 HM RST" *)
    input  wire         hm_rst,          // not used

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire [2:0]                        s_axi_awprot,
    input  wire                              s_axi_awvalid,
    output wire                              s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output wire                              s_axi_wready,
    output wire [1:0]                        s_axi_bresp,
    output wire                              s_axi_bvalid,
    input  wire                              s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire [2:0]                        s_axi_arprot,
    input  wire                              s_axi_arvalid,
    output wire                              s_axi_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output wire [1:0]                        s_axi_rresp,
    output wire                              s_axi_rvalid,
    input  wire                              s_axi_rready
);
    localparam [31:0] VERSION = 32'h0600_0001;
    localparam [7:0] HOST_RD_LAT = 8'd8;

    wire         reg_done, mem_idle;
    wire [31:0]  reg_cycle_cnt;
    wire [127:0] blk_cycles;
    wire [1:0]   cur_blk;

    // ---------------------------------------------------------------- AXI4-Lite slave (the Vivado template of v4.1, more registers)
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg                          axi_awready;
    reg                          axi_wready;
    reg [1:0]                    axi_bresp;
    reg                          axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg                          axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0]                    axi_rresp;
    reg                          axi_rvalid;
    reg                          aw_en;

    localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
    localparam integer OPT_MEM_ADDR_BITS = 3;

    reg [31:0] slv_ctrl, slv_cfg;
    wire       slv_reg_rden, slv_reg_wren;
    reg [31:0] reg_data_out;
    integer    byte_index;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = axi_bresp;
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = axi_rresp;
    assign s_axi_rvalid  = axi_rvalid;

    always @(posedge aclk) begin
        if (aresetn == 1'b0) begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end else begin
            if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                axi_awready <= 1'b1;
                aw_en <= 1'b0;
            end else if (s_axi_bready && axi_bvalid) begin
                aw_en <= 1'b1;
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) axi_awaddr <= 0;
        else if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) axi_awaddr <= s_axi_awaddr;
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) axi_wready <= 1'b0;
        else if (~axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en) axi_wready <= 1'b1;
        else axi_wready <= 1'b0;
    end

    assign slv_reg_wren = axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid;

    always @(posedge aclk) begin
        if (aresetn == 1'b0) begin
            slv_ctrl <= 0;
            slv_cfg  <= 0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                4'h0: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                          if (s_axi_wstrb[byte_index] == 1) slv_ctrl[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
                4'h3: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                          if (s_axi_wstrb[byte_index] == 1) slv_cfg[(byte_index*8) +: 8] <= s_axi_wdata[(byte_index*8) +: 8];
                default: ;
            endcase
        end
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) begin
            axi_bvalid <= 0;
            axi_bresp  <= 2'b0;
        end else begin
            if (axi_awready && s_axi_awvalid && ~axi_bvalid && axi_wready && s_axi_wvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b0;
            end else if (s_axi_bready && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 0;
        end else begin
            if (~axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_araddr  <= s_axi_araddr;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) begin
            axi_rvalid <= 0;
            axi_rresp  <= 0;
        end else begin
            if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b0;
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    assign slv_reg_rden = axi_arready & s_axi_arvalid & ~axi_rvalid;

    always @(*) begin
        case (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
            4'h0:    reg_data_out = slv_ctrl;
            4'h1:    reg_data_out = {26'd0, cur_blk, 2'd0, ~mem_idle, reg_done};
            4'h2:    reg_data_out = reg_cycle_cnt;
            4'h3:    reg_data_out = slv_cfg;
            4'h4:    reg_data_out = VERSION;
            4'h5:    reg_data_out = {16'd256, 8'd8, HOST_RD_LAT};
            4'h8:    reg_data_out = blk_cycles[31:0];
            4'h9:    reg_data_out = blk_cycles[63:32];
            4'hA:    reg_data_out = blk_cycles[95:64];
            4'hB:    reg_data_out = blk_cycles[127:96];
            default: reg_data_out = 0;
        endcase
    end

    always @(posedge aclk) begin
        if (aresetn == 1'b0) axi_rdata <= 0;
        else if (slv_reg_rden) axi_rdata <= reg_data_out;
    end

    // ---------------------------------------------------------------- accelerator + weight store
    // the reset of the accelerator is registered twice here so that its large fan-out starts from a local flop
    reg rst_q1, rst_q2;
    always @(posedge aclk) begin rst_q1 <= aresetn; rst_q2 <= rst_q1; end
    effvit_sys #(.HOST_RD_LAT(HOST_RD_LAT)) u_sys (
        .clk(aclk), .rst_n(rst_q2),
        .reg_start(slv_ctrl[0]), .cfg_first(slv_cfg[1:0]), .cfg_nm1(slv_cfg[5:4]),
        .reg_done(reg_done), .mem_idle(mem_idle), .reg_cycle_cnt(reg_cycle_cnt), .blk_cycles(blk_cycles), .cur_blk(cur_blk),
        .hm_en(hm_en), .hm_we(hm_we), .hm_addr(hm_addr), .hm_wdata(hm_wdata), .hm_rdata(hm_rdata)
    );
endmodule
