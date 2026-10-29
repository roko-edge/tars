`timescale 1ns / 1ps
module tb;
  logic clk;
  logic rst;
  logic start;
  localparam int IN  = 2;
  localparam int OUT = 1;
  logic signed [31:0] a[IN], b[OUT], result[OUT], w0;
  logic [1:0] w[IN*OUT];
  logic done;

  layer_engine #(
      .IN (IN),
      .OUT(OUT)
  ) dut (
      .clk(clk),
      .rst(rst),
      .a(a),
      .w(w),
      .b(b),
      .w0(w0),
      .start(start),
      .act_fn(1),
      .done(done),
      .result(result)
  );

  function automatic real to_float(input logic signed [31:0] q8_24);
    return $itor(q8_24) / 16777216.0;
  endfunction


  always #5 clk = ~clk;
  logic signed [31:0] data[8];
  assign w0 = 16777216;
  initial begin
    $dumpfile("build/dump.vcd");
    $dumpvars(0, tb);

    clk = 0;

    $readmemh("data/bias.mem", b);
    $readmemh("data/weights.mem", w);
    $readmemh("data/activations.mem", data);

    $display("\n");

    for (int s = 0; s < 4; ++s) begin
      rst   = 1;
      start = 0;
      #10;
      rst  = 0;
      a[0] = data[s*2];
      a[1] = data[s*2+1];

      #10;
      start = 1;
      #10;
      start = 0;
      wait (done);
      $display("Entries: %0.6f %0.6f the result is %0.6f\n", to_float(data[s*2]), to_float(
               data[s*2+1]), to_float(result[0]));
    end

    $finish;
  end
endmodule
