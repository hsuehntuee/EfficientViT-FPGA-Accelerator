`timescale 1 ns / 1 ps

module Efficient_vit_v2_v1_0 #(
    parameter integer C_S00_AXI_DATA_WIDTH = 32, 
    parameter integer C_S00_AXI_ADDR_WIDTH = 4
)(
    output wire bram_x_clk, bram_x_en, 
    output wire bram_w_clk, bram_w_en, 
    output wire bram_b_clk, bram_b_en,
    output wire bram_dw_w_clk, bram_dw_w_en, 
    output wire bram_dw_b_clk, bram_dw_b_en,
    output wire bram_pw_w_clk, bram_pw_w_en, 
    output wire bram_pw_b_clk, bram_pw_b_en, 
    output wire bram_y_clk, bram_y_en,
    
    // [新增] LiteMLA BRAM 接口
    output wire bram_mla_w_clk, bram_mla_w_en,
    output wire bram_mla_b_clk, bram_mla_b_en,

    // MS 分支 group 1x1 的權重 / bias BRAM 接口
    output wire bram_ms_pw_w_clk, bram_ms_pw_w_en,
    output wire bram_ms_pw_b_clk, bram_ms_pw_b_en,
    output wire [31:0] bram_ms_pw_w_addr, input wire [255:0] bram_ms_pw_w_rdata,
    output wire [31:0] bram_ms_pw_b_addr, input wire [255:0] bram_ms_pw_b_rdata,

    output wire [31:0] bram_x_addr, input wire [255:0] bram_x_rdata,
    output wire [31:0] bram_w_addr, input wire [255:0] bram_w_rdata,
    output wire [31:0] bram_b_addr, input wire [255:0] bram_b_rdata,
    output wire [31:0] bram_dw_w_addr, input wire [255:0] bram_dw_w_rdata,
    output wire [31:0] bram_dw_b_addr, input wire [255:0] bram_dw_b_rdata,
    output wire [31:0] bram_pw_w_addr, input wire [255:0] bram_pw_w_rdata,
    output wire [31:0] bram_pw_b_addr, input wire [255:0] bram_pw_b_rdata,
    
    // [新增] LiteMLA 地址與資料接口
    output wire [31:0] bram_mla_w_addr, input wire [255:0] bram_mla_w_rdata,
    output wire [31:0] bram_mla_b_addr, input wire [255:0] bram_mla_b_rdata,
    
    output wire [31:0] bram_y_addr, output wire [255:0] bram_y_wdata, output wire [31:0] bram_y_we,
    
    input wire s00_axi_aclk, s00_axi_aresetn,
    input wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr, input wire [2:0] s00_axi_awprot,
    input wire s00_axi_awvalid, output wire s00_axi_awready,
    input wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_wdata, input wire [(C_S00_AXI_DATA_WIDTH/8)-1:0] s00_axi_wstrb,
    input wire s00_axi_wvalid, output wire s00_axi_wready,
    output wire [1:0] s00_axi_bresp, output wire s00_axi_bvalid, input wire s00_axi_bready,
    input wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_araddr, input wire [2:0] s00_axi_arprot,
    input wire s00_axi_arvalid, output wire s00_axi_arready,
    output wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_rdata, output wire [1:0] s00_axi_rresp,
    output wire s00_axi_rvalid, input wire s00_axi_rready
);

    wire reg_start, reg_done; wire [31:0] reg_cycle_cnt;

    Efficient_vit_v2_v1_0_S00_AXI #(.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH), .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)) Efficient_vit_v2_v1_0_S00_AXI_inst (
        .reg_start(reg_start), .reg_done(reg_done), .reg_cycle_cnt(reg_cycle_cnt),
        .S_AXI_ACLK(s00_axi_aclk), .S_AXI_ARESETN(s00_axi_aresetn),
        .S_AXI_AWADDR(s00_axi_awaddr), .S_AXI_AWPROT(s00_axi_awprot), .S_AXI_AWVALID(s00_axi_awvalid), .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA(s00_axi_wdata), .S_AXI_WSTRB(s00_axi_wstrb), .S_AXI_WVALID(s00_axi_wvalid), .S_AXI_WREADY(s00_axi_wready),
        .S_AXI_BRESP(s00_axi_bresp), .S_AXI_BVALID(s00_axi_bvalid), .S_AXI_BREADY(s00_axi_bready),
        .S_AXI_ARADDR(s00_axi_araddr), .S_AXI_ARPROT(s00_axi_arprot), .S_AXI_ARVALID(s00_axi_arvalid), .S_AXI_ARREADY(s00_axi_arready),
        .S_AXI_RDATA(s00_axi_rdata), .S_AXI_RRESP(s00_axi_rresp), .S_AXI_RVALID(s00_axi_rvalid), .S_AXI_RREADY(s00_axi_rready)
    );

    // 512x512 input: Stage 4 is 16x16 = 256 tokens (IMG_SIZE^2 must equal N_TOKENS)
    accelerator_top #(
        .ROWS(16), .COLS(16), .N_TOKENS(256), .IMG_SIZE(16), .C_IN_CHUNKS(16), .C_OUT_CHUNKS(64), .FRAC_BITS(11), .A_FRAC_BITS(5), .EXPAND_ACT(1), .DW_ACT(1)
    ) u_accelerator_top (
        .clk(s00_axi_aclk), .rst_n(s00_axi_aresetn),
        .reg_start(reg_start), .reg_done(reg_done), .reg_cycle_cnt(reg_cycle_cnt),
        
        .bram_x_addr(bram_x_addr), .bram_x_rdata(bram_x_rdata), 
        .bram_w_addr(bram_w_addr), .bram_w_rdata(bram_w_rdata),
        .bram_b_addr(bram_b_addr), .bram_b_rdata(bram_b_rdata), 
        .bram_dw_w_addr(bram_dw_w_addr), .bram_dw_w_rdata(bram_dw_w_rdata),
        .bram_dw_b_addr(bram_dw_b_addr), .bram_dw_b_rdata(bram_dw_b_rdata), 
        .bram_pw_w_addr(bram_pw_w_addr), .bram_pw_w_rdata(bram_pw_w_rdata),
        .bram_pw_b_addr(bram_pw_b_addr), .bram_pw_b_rdata(bram_pw_b_rdata), 
        
        .bram_mla_w_addr(bram_mla_w_addr), .bram_mla_w_rdata(bram_mla_w_rdata),
        .bram_mla_b_addr(bram_mla_b_addr), .bram_mla_b_rdata(bram_mla_b_rdata),
        
        .bram_y_addr(bram_y_addr), .bram_y_wdata(bram_y_wdata), .bram_y_we(bram_y_we),
        
        .bram_x_clk(bram_x_clk), .bram_x_en(bram_x_en), 
        .bram_w_clk(bram_w_clk), .bram_w_en(bram_w_en),
        .bram_b_clk(bram_b_clk), .bram_b_en(bram_b_en), 
        .bram_dw_w_clk(bram_dw_w_clk), .bram_dw_w_en(bram_dw_w_en),
        .bram_dw_b_clk(bram_dw_b_clk), .bram_dw_b_en(bram_dw_b_en), 
        .bram_pw_w_clk(bram_pw_w_clk), .bram_pw_w_en(bram_pw_w_en),
        .bram_pw_b_clk(bram_pw_b_clk), .bram_pw_b_en(bram_pw_b_en), 
        
        // [新增] 時脈與致能
        .bram_mla_w_clk(bram_mla_w_clk), .bram_mla_w_en(bram_mla_w_en),
        .bram_mla_b_clk(bram_mla_b_clk), .bram_mla_b_en(bram_mla_b_en),

        .bram_ms_pw_w_clk(bram_ms_pw_w_clk), .bram_ms_pw_w_en(bram_ms_pw_w_en),
        .bram_ms_pw_b_clk(bram_ms_pw_b_clk), .bram_ms_pw_b_en(bram_ms_pw_b_en),
        .bram_ms_pw_w_addr(bram_ms_pw_w_addr), .bram_ms_pw_w_rdata(bram_ms_pw_w_rdata),
        .bram_ms_pw_b_addr(bram_ms_pw_b_addr), .bram_ms_pw_b_rdata(bram_ms_pw_b_rdata),

        .bram_y_clk(bram_y_clk), .bram_y_en(bram_y_en)
    );
endmodule