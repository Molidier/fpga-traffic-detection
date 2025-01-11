module uart_rx
#(
    parameter data_width   = 8,
          
              IDLE         = 3'b000,
              START_BIT    = 3'b001,
              DATA_BITS    = 3'b010, 
              STOP_BIT     = 3'b011,
              DONE         = 3'b101,
              ERROR_ST     = 3'b110
) 
(
    input                     data_bit,
    input                     clk,
    input                     rst,
	 input [12:0]              CLKS_PER_BIT,
    output                    done,
    output [data_width - 1:0] data_bus
);

    // FSM states
    reg [2:0] PS;
    reg [2:0] NS;

    // Internal nets
    reg [2:0]  bit_counter;
    reg [12:0] clk_counter;
    reg [data_width - 1:0] data_bus_wire;

    // Output assignments
    assign done      = (PS == DONE);
    assign data_bus  = data_bus_wire;

    // FSM: PS synchronization
    always @(posedge clk) begin
        if (!rst) begin
            PS <= IDLE;
        end
        else begin
            PS <= NS;
        end
    end

    // FSM: Data and control logic
    always @(negedge clk) begin
        // Default values
        clk_counter   <= clk_counter;
        bit_counter   <= bit_counter;
        data_bus_wire <= data_bus_wire;

        case (PS)
            IDLE: begin
                clk_counter <= 0;
                bit_counter <= 0;

                if (data_bit == 1'b0) begin
                    // Start bit detected
                end
            end

            START_BIT: begin
                if (clk_counter == CLKS_PER_BIT / 2) begin
                    if (data_bit == 1'b0) begin
                        clk_counter <= 0;
                    end
                end
                else begin
                    clk_counter <= clk_counter + 1;
                end
            end

            DATA_BITS: begin
                if (clk_counter < CLKS_PER_BIT - 1) begin
                    clk_counter <= clk_counter + 1;
                end
                else begin
                    clk_counter <= 0;
                    data_bus_wire[bit_counter] <= data_bit;

                    if (bit_counter < data_width - 1) begin
                        bit_counter <= bit_counter + 1;
                    end
                end
            end

            STOP_BIT: begin
                if (clk_counter < CLKS_PER_BIT - 1) begin
                    clk_counter <= clk_counter + 1;
                end
                else begin
                    clk_counter <= 0;
                end
            end

            ERROR_ST: begin
                // Handle error state
            end

            DONE: begin
                // Completion state
            end

            default: begin
                clk_counter <= 0;
                bit_counter <= 0;
                data_bus_wire <= 0;
            end
        endcase
    end

    // Next state transition logic
    always @(negedge clk) begin
        // Default next state
        NS <= PS;

        case (PS)
            IDLE: begin
                NS <= (data_bit == 1'b0) ? START_BIT : IDLE;
            end

            START_BIT: begin
                NS <= (clk_counter == CLKS_PER_BIT / 2 && data_bit == 1'b0) ? DATA_BITS : 
                      (clk_counter == CLKS_PER_BIT / 2 && data_bit == 1'b1) ? ERROR_ST : START_BIT;
            end

            DATA_BITS: begin
                if (clk_counter == CLKS_PER_BIT - 1) begin
                    NS <= (bit_counter < data_width - 1) ? DATA_BITS : STOP_BIT;
                end
            end

            STOP_BIT: begin
                NS <= (clk_counter == CLKS_PER_BIT - 1) ? DONE : STOP_BIT;
            end

            ERROR_ST: begin
                NS <= ERROR_ST;
            end

            DONE: begin
                NS <= IDLE;
            end

            default: begin
                NS <= IDLE;
            end
        endcase
    end

endmodule
