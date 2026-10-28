module pe (
    input logic clk,
    rst,
    init,
    en,
    input logic unsigned [1:0] w,
    input logic signed [31:0] a,
    output logic signed [63:0] accumulator
);
  logic unsigned w_en, w_s;
  t3_dec u_t3 (
      .w,
      .w_en,
      .w_s
  );
  logic [63:0] term;
  assign term = 64'(w_s ? 64'(-(33'(a))) : a);
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      accumulator <= 0;
    end else if (en) begin
      if (init) begin
        accumulator <= w_en ? term : 0;
      end else if (w_en) begin
        accumulator <= accumulator + term;
      end
    end
  end
endmodule
