# NPU - fluxo Verilator-only (Icarus foi removido do fluxo).
# A lista de fontes única é files.f (partilhada com o LSP).

HAVE_VERILATOR := $(shell command -v verilator 2> /dev/null)

ifeq ($(HAVE_VERILATOR),)
  RUN = nix develop ..\#hardware -c
else
  RUN =
endif

BUILD_DIR = build
FILELIST  = files.f
# Fontes derivadas de files.f (fonte única; não duplicar a lista aqui).
SRCS := $(shell sed -e '/^#/d' -e '/^[[:space:]]*$$/d' $(FILELIST))
TB   = tests/tb.sv

.PHONY: all sim run test wave lint clean

all: test

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 1. Compila o binário C++ APENAS se algum ficheiro .sv ou a lista mudou.
#    -Wno-fatal: o lint (alvo abaixo) é o gate de warnings; o build do tb
#    continua a expô-los sem bloquear a simulação.
$(BUILD_DIR)/npu_sim: $(FILELIST) $(SRCS) $(TB) | $(BUILD_DIR)
	@$(RUN) verilator --binary --timing --assert --trace \
		--quiet --quiet-build --quiet-stats \
		-Wno-fatal -Wno-TIMESCALEMOD \
		-f $(FILELIST) $(TB) --top-module tb \
		-o npu_sim -Mdir $(BUILD_DIR)

# 2. Executa a simulação; o exit code do tb ($fatal -> não-zero) propaga
#    pelo make (é o que liga o `make test` ao self-checking do tb).
sim: $(BUILD_DIR)/npu_sim
	@$(RUN) ./$(BUILD_DIR)/npu_sim +verilator+quiet

# Alias prático para execução
run: sim

test: sim

wave: sim
	@$(RUN) gtkwave $(BUILD_DIR)/dump.vcd

# Gate de qualidade do RTL: -Wall sem supressões de WIDTH*; tem de sair limpo.
# Topo de lint = npu (cobre wrapper + engines + primitivos inteiros).
lint:
	@$(RUN) verilator --lint-only -Wall --top-module npu -f $(FILELIST)

clean:
	@rm -rf $(BUILD_DIR)
