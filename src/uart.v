/*
 * Copyright (c) 2024 Moldir Azhimukhanbet, Maveric Lab
 * SPDX-License-Identifier: Apache-2.0
 */

module uart #(
    parameter data_width = 8,
	 parameter ascii_zero = 48
)
(
	input        clk, 
	input        rst,

	input        rx_data_bit,
	output       rx_done,

	output       tx_data_bit,
	input  [data_width-1:0] data_tx,
	input        tx_en,
	output       tx_done,

	output [data_width-1:0] recieved_data,
	
	output [6:0] HEX0,
	output [6:0] HEX1,
	output [6:0] HEX2


);
	wire [12:0] clks_per_bit;
	
	assign clks_per_bit = 5208;

	uart_rx R0(
		.data_bit(rx_data_bit),
		.clk(clk),
		.rst(rst),
    	.CLKS_PER_BIT(clks_per_bit),
		.done(rx_done),
		.data_bus(recieved_data)
	);


	uart_tx T0(
		.data_bus(data_tx),
		.clk(clk),
		.rstn(rst), 
    	.CLKS_PER_BIT(clks_per_bit),
		.run(tx_en), //active when low
		.done(tx_done),
		.data_bit(tx_data_bit)
	);		

	decoder_hex D0(
	.in(recieved_data),
	.HEX0(HEX0),
	.HEX1(HEX1),
	.HEX2(HEX2)
);

endmodule
