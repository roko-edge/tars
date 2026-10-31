`timescale 1ns / 1ps
module tb_core;

  import npu_pkg::*;

  localparam int IN = 2, H = 2, OUT = 2;
  logic clk, rst, start;
  logic [1:0] l0_w[IN*H], l1_w[H*OUT];
  logic signed [31:0] a[IN], l0_b[H], l1_b[OUT], l0_w0, l1_w0;
  logic done;
  npu_pkg::activation_t l0_act_fn, l1_act_fn;
  logic signed [31:0] result[OUT];

  npu #(
      .IN,
      .H,
      .OUT
  ) dut (
      .clk,
      .rst,
      .start,
      .l0_w,
      .l1_w,
      .a,
      .l0_b,
      .l1_b,
      .l0_w0,
      .l1_w0,
      .l0_act_fn,
      .l1_act_fn,
      .done,
      .result
  );

  task automatic forward(input logic signed [31:0] a0, input logic signed [31:0] a1);
    rst   = 1;
    start = 0;
    repeat (2) @(posedge clk);
    rst  = 0;
    a[0] = a0;
    a[1] = a1;
    @(posedge clk);
    start = 1;
    @(posedge clk);
    start = 0;  // start in one cicle
    do @(posedge clk); while (!done);  // waits for the inference result
  endtask
  task automatic check(input string nome, input logic signed [31:0] got,
                       input logic signed [31:0] exp);
    if (got !== exp) $fatal(1, "%s: got %h, expected %h", nome, got, exp);
    else $display("PASS %s (%h)", nome, got);
  endtask

  initial begin
    #100_000 $fatal(1, "timeout: done nunca chegou");
  end
  initial begin
    $dumpfile("build/dump.vcd");
    $dumpvars(0, tb_core);
  end
  localparam logic signed [31:0] Q1 = 32'sh0100_0000;  // 1.0 em Q8.24
  assign l0_w0 = Q1;
  assign l1_w0 = Q1;

  always #5 clk = ~clk;
  initial begin
    clk = 0;
    l0_w[0] = 2'b01;
    l0_w[1] = 2'b01;
    l0_w[2] = 2'b10;
    l0_w[3] = 2'b10;
    //           1                1              -1                -1
    l1_w[0] = 2'b01;
    l1_w[1] = 2'b00;
    l1_w[2] = 2'b00;
    l1_w[3] = 2'b10;
    //           1                0              0                -1
    for (int i = 0; i < H; i++) l0_b[i] = 32'sh0200_0000;  // +2.0
    for (int i = 0; i < OUT; i++) l1_b[i] = 32'shFE00_0000;  // -2.0
    l0_act_fn = npu_pkg::RELU;
    l1_act_fn = npu_pkg::IDENTITY;

    forward(32'sh0100_0000, 32'sh0080_0000);  // a = [1.0, 0.5]
    check("case1 result[0]", result[0], 32'sh0180_0000);
    check("case1 result[1]", result[1], 32'shFD80_0000);

    $display("tb_core: OK");
    $finish;
  end
endmodule
