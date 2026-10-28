module t3_dec (
    input logic unsigned [1:0] w,
    output logic w_en,
    output logic w_s
);
  assign w_en = w[1] ^ w[0];
  assign w_s  = w[1];
endmodule
