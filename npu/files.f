# Single source list of NPU RTL files - shared by Makefile and LSP.
# npu_pkg.sv must always be first (package declared before dependent modules).
src/npu_pkg.sv
src/t3_dec.sv
src/pe.sv
src/epilogue.sv
src/layer_engine.sv
src/npu.sv
