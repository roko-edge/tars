{
  description = "tars-ml: Machine Learning & NPU Hardware built from scratch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      hardwareTools = with pkgs; [
        gnumake # npu/Makefile
        python3 # driver de build do Verilator (--binary), exigido pelo fluxo de simulação
        verilator # simulação (--binary --timing --trace) e lint estático do fluxo npu/Makefile
        iverilog # simulador IEEE 1800-2012 alternativo; não usado pelo npu/Makefile atual
        gtkwave # ondas VCD
        verible # LSP e formatador SystemVerilog
      ];

      rustTools = with pkgs; [
        rustc
        cargo
        clippy
        rustfmt
        rust-analyzer
        # dependências de sistema do plotters
        pkg-config
        graphviz
        fontconfig
      ];

      nixTools = with pkgs; [
        nil
        nixfmt-rfc-style
      ];
    in
    {
      devShells.${system} = {
        # Shell padrão (carregado pelo direnv na raiz)
        default = pkgs.mkShell {
          name = "tars-ml-dev";
          packages = hardwareTools ++ rustTools ++ nixTools;
        };
        # Apenas hardware/NPU
        hardware = pkgs.mkShell {
          name = "tars-ml-hardware";
          packages = hardwareTools;
        };
        # Apenas Rust
        rust = pkgs.mkShell {
          name = "tars-ml-rust";
          packages = rustTools;
        };
      };

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
