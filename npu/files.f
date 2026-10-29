# Fonte única de ficheiros do NPU - partilhada pelo Makefile e pelo LSP.
# npu_pkg.sv sempre primeiro (package tem de estar declarado antes dos módulos).
src/npu_pkg.sv
src/t3_dec.sv
src/pe.sv
src/epilogue.sv
src/layer_engine.sv
