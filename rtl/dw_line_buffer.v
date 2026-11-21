`timescale 1ns/1ps

module dw_line_buffer #(
    parameter IMG_SIZE = 16,    
    parameter CHANNELS = 128    
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire                 in_valid,
    input  wire [CHANNELS*16-1:0] in_data,
    
    output reg                  out_valid,
    output wire [25*CHANNELS*16-1:0] window_data
);

    // ==========================================
    // Dynamic Mathematical Parameters
    // ==========================================
    localparam PIX_W        = CHANNELS * 16;
    localparam SR_SIZE      = 4 * IMG_SIZE + 5;           // 5x5 requires 4 full rows + 5 pixels
    localparam CENTER_DELAY = 2 * IMG_SIZE + 2;           // Cycles for 1st pixel to reach center
    localparam TOTAL_PIXELS = IMG_SIZE * IMG_SIZE;        // Total pixels in the feature map

    reg [PIX_W-1:0] sr [0:SR_SIZE-1];
    integer i;
    
    reg [15:0] in_cnt;
    reg [15:0] out_r, out_c;
    reg [15:0] out_cnt;
    reg        out_active;

    wire shift_en = in_valid || out_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < SR_SIZE; i = i + 1) sr[i] <= 0;
            in_cnt <= 0; out_r <= 0; out_c <= 0; 
            out_cnt <= 0; out_active <= 0; out_valid <= 0;
        end else begin
            out_valid <= 0; 
            if (shift_en) begin
                sr[0] <= in_valid ? in_data : {PIX_W{1'b0}};
                for (i = 1; i < SR_SIZE; i = i + 1) sr[i] <= sr[i-1];
                
                if (in_valid) in_cnt <= in_cnt + 1;
                
                // Trigger when the first pixel reaches the 5x5 center
                if (in_cnt == CENTER_DELAY || out_active) begin
                    out_active <= 1;
                    out_valid <= 1;
                    
                    if (out_active) begin
                        if (out_c == IMG_SIZE - 1) begin
                            out_c <= 0;
                            out_r <= out_r + 1;
                        end else begin
                            out_c <= out_c + 1;
                        end
                        out_cnt <= out_cnt + 1;
                        
                        // Finish condition dynamically based on image size
                        if (out_cnt == TOTAL_PIXELS - 1) begin
                            out_active <= 0; out_valid <= 0; out_cnt <= 0;
                            out_r <= 0; out_c <= 0; in_cnt <= 0;
                        end
                    end
                end
            end
        end
    end

    // ==========================================
    // [CRITICAL FIX] Strict 5x5 Boundary Padding Check
    // Replaced loose inequalities with strict absolute coordinate masks
    // ==========================================
    // Check if the specific window row (-2, -1, 0, 1, 2 relative to out_r) is out of bounds
    wire pad_row_0 = (out_r < 2);                           // Window Row 0 is out if Center Row < 2
    wire pad_row_1 = (out_r < 1);                           // Window Row 1 is out if Center Row < 1
    wire pad_row_3 = (out_r + 1 >= IMG_SIZE);               // Window Row 3 is out if Center Row + 1 >= IMG_SIZE
    wire pad_row_4 = (out_r + 2 >= IMG_SIZE);               // Window Row 4 is out if Center Row + 2 >= IMG_SIZE

    // Check if the specific window col (-2, -1, 0, 1, 2 relative to out_c) is out of bounds
    wire pad_col_0 = (out_c < 2);                           // Window Col 0 is out if Center Col < 2
    wire pad_col_1 = (out_c < 1);                           // Window Col 1 is out if Center Col < 1
    wire pad_col_3 = (out_c + 1 >= IMG_SIZE);               // Window Col 3 is out if Center Col + 1 >= IMG_SIZE
    wire pad_col_4 = (out_c + 2 >= IMG_SIZE);               // Window Col 4 is out if Center Col + 2 >= IMG_SIZE

    // ==========================================
    // Dynamic Window Data Packing with Strict Masking
    // ==========================================
    
    // Window Row 0 (Top-Most)
    assign window_data[ 0*PIX_W +: PIX_W] = (pad_row_0 || pad_col_0) ? {PIX_W{1'b0}} : sr[4*IMG_SIZE + 4];
    assign window_data[ 1*PIX_W +: PIX_W] = (pad_row_0 || pad_col_1) ? {PIX_W{1'b0}} : sr[4*IMG_SIZE + 3];
    assign window_data[ 2*PIX_W +: PIX_W] = (pad_row_0)              ? {PIX_W{1'b0}} : sr[4*IMG_SIZE + 2];
    assign window_data[ 3*PIX_W +: PIX_W] = (pad_row_0 || pad_col_3) ? {PIX_W{1'b0}} : sr[4*IMG_SIZE + 1];
    assign window_data[ 4*PIX_W +: PIX_W] = (pad_row_0 || pad_col_4) ? {PIX_W{1'b0}} : sr[4*IMG_SIZE + 0];

    // Window Row 1 (Top-Mid)
    assign window_data[ 5*PIX_W +: PIX_W] = (pad_row_1 || pad_col_0) ? {PIX_W{1'b0}} : sr[3*IMG_SIZE + 4];
    assign window_data[ 6*PIX_W +: PIX_W] = (pad_row_1 || pad_col_1) ? {PIX_W{1'b0}} : sr[3*IMG_SIZE + 3];
    assign window_data[ 7*PIX_W +: PIX_W] = (pad_row_1)              ? {PIX_W{1'b0}} : sr[3*IMG_SIZE + 2];
    assign window_data[ 8*PIX_W +: PIX_W] = (pad_row_1 || pad_col_3) ? {PIX_W{1'b0}} : sr[3*IMG_SIZE + 1];
    assign window_data[ 9*PIX_W +: PIX_W] = (pad_row_1 || pad_col_4) ? {PIX_W{1'b0}} : sr[3*IMG_SIZE + 0];

    // Window Row 2 (Center)
    assign window_data[10*PIX_W +: PIX_W] = (pad_col_0)              ? {PIX_W{1'b0}} : sr[2*IMG_SIZE + 4];
    assign window_data[11*PIX_W +: PIX_W] = (pad_col_1)              ? {PIX_W{1'b0}} : sr[2*IMG_SIZE + 3];
    assign window_data[12*PIX_W +: PIX_W] =                                            sr[2*IMG_SIZE + 2]; // Absolute Center
    assign window_data[13*PIX_W +: PIX_W] = (pad_col_3)              ? {PIX_W{1'b0}} : sr[2*IMG_SIZE + 1];
    assign window_data[14*PIX_W +: PIX_W] = (pad_col_4)              ? {PIX_W{1'b0}} : sr[2*IMG_SIZE + 0];

    // Window Row 3 (Bottom-Mid)
    assign window_data[15*PIX_W +: PIX_W] = (pad_row_3 || pad_col_0) ? {PIX_W{1'b0}} : sr[1*IMG_SIZE + 4];
    assign window_data[16*PIX_W +: PIX_W] = (pad_row_3 || pad_col_1) ? {PIX_W{1'b0}} : sr[1*IMG_SIZE + 3];
    assign window_data[17*PIX_W +: PIX_W] = (pad_row_3)              ? {PIX_W{1'b0}} : sr[1*IMG_SIZE + 2];
    assign window_data[18*PIX_W +: PIX_W] = (pad_row_3 || pad_col_3) ? {PIX_W{1'b0}} : sr[1*IMG_SIZE + 1];
    assign window_data[19*PIX_W +: PIX_W] = (pad_row_3 || pad_col_4) ? {PIX_W{1'b0}} : sr[1*IMG_SIZE + 0];

    // Window Row 4 (Bottom-Most)
    assign window_data[20*PIX_W +: PIX_W] = (pad_row_4 || pad_col_0) ? {PIX_W{1'b0}} : sr[0*IMG_SIZE + 4];
    assign window_data[21*PIX_W +: PIX_W] = (pad_row_4 || pad_col_1) ? {PIX_W{1'b0}} : sr[0*IMG_SIZE + 3];
    assign window_data[22*PIX_W +: PIX_W] = (pad_row_4)              ? {PIX_W{1'b0}} : sr[0*IMG_SIZE + 2];
    assign window_data[23*PIX_W +: PIX_W] = (pad_row_4 || pad_col_3) ? {PIX_W{1'b0}} : sr[0*IMG_SIZE + 1];
    assign window_data[24*PIX_W +: PIX_W] = (pad_row_4 || pad_col_4) ? {PIX_W{1'b0}} : sr[0*IMG_SIZE + 0];

endmodule