`timescale 1ns/1ps

module bram_wrapper #(
    parameter DEPTH = 256,
    parameter WIDTH = 256
)(
    input  wire clk,
    input  wire we,                     // Write Enable
    input  wire [$clog2(DEPTH)-1:0] addr, // Address (Read/Write 共用)
    input  wire [WIDTH-1:0] wdata,      // Write Data
    output reg  [WIDTH-1:0] rdata       // Read Data (延遲 1 Cycle)
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= wdata;
        end
        rdata <= mem[addr]; 
    end

endmodule