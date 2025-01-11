// HEX decoder
// 7-segment decoder

module decoder 
#(
	parameter width = 7
)
(
	input  [3:0]         SW,
	output [width - 1:0] HEX
);

	assign HEX = (SW == 4'b0000) ? 7'b1000000 :
					 (SW == 4'b0001) ? 7'b1111001 :
					 (SW == 4'b0010) ? 7'b0100100 :
					 (SW == 4'b0011) ? 7'b0110000 :
					 (SW == 4'b0100) ? 7'b0011001 :
					 (SW == 4'b0101) ? 7'b0010010 :
					 (SW == 4'b0110) ? 7'b0000010 :
					 (SW == 4'b0111) ? 7'b1111000 :
					 (SW == 4'b1000) ? 7'b0000000 :
					 (SW == 4'b1001) ? 7'b0010000 :
					 7'b0111111;
					 
endmodule

module decoder_hex 
(
	input  [7:0] in,
	output [6:0] HEX0,
	output [6:0] HEX1,
	output [6:0] HEX2
);

	wire [7:0] in1;
	wire [7:0] in2;
	wire [7:0] in3;
	wire [7:0] n1;
	wire [7:0] n2;
	wire [7:0] n3;
	
	assign in1 = in % 8'd10;
	assign n1 = in % 8'd100;
	assign n2 = n1 - in1;
	assign in2 = n2 / 8'd10;
	assign n3 = in - n1;
	assign in3 = n3 / 8'd100;
	decoder		H0(in1[3:0], HEX0),
					H1(in2[3:0], HEX1),
					H2(in3[3:0], HEX2);
	

endmodule

