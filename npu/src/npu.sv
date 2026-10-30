module npu
  import npu_pkg::*;
#(
    parameter int IN  = 784,
    parameter int H   = 64,
    parameter int OUT = 10

) (
    input logic clk,
    rst,
    start,
    input logic [1:0] l0_w[IN*H],
    l1_w[H*OUT],
    input logic signed [31:0] a[IN],
    l0_b[H],
    l1_b[OUT],
    l0_w0,
    l1_w0,
    output logic signed [31:0] result[OUT],
    output logic done
);

  logic signed [31:0] l0_result[H];
  logic l0_done;

  layer_engine #(
      .IN (IN),
      .OUT(H)
  ) u_l0 (
      .clk,
      .rst,
      .start(start && state == IDLE),
      .w(l0_w),
      .a,
      .w0(l0_w0),
      .b(l0_b),
      .act_fn(npu_pkg::RELU),
      .result(l0_result),
      .done(l0_done)
  );
  layer_engine #(
      .IN (H),
      .OUT(OUT)
  ) u_l1 (
      .clk,
      .rst,
      .start(l0_done),
      .w(l1_w),
      .a(l0_result),
      .w0(l1_w0),
      .b(l1_b),
      .act_fn(npu_pkg::IDENTITY),
      .result(result),
      .done(done)
  );

  npu_pkg::state_t state;
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state <= npu_pkg::IDLE;
    end else begin
      if (start && state == npu_pkg::IDLE) begin
        state <= npu_pkg::RUN;
      end else begin
        if (state == npu_pkg::RUN && done) begin
          state <= npu_pkg::IDLE;
        end
      end

    end
  end
endmodule
