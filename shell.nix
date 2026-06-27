{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312.withPackages (ps: [
    ps.pip
    ps.setuptools
    ps.wheel
  ]);
in
pkgs.mkShell {
  packages = [
    pkgs.libopus
    pkgs.nodejs
    pkgs.pkg-config
    pkgs.portaudio
    python
  ];

  shellHook = ''
    export PYTHONPATH="$PWD/moshi''${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export HF_HOME="''${HF_HOME:-$PWD/.cache/huggingface}"
    export PATH="$PWD/.venv-nix/bin:$PATH"
    echo "PersonaPlex nix shell ready. Run ./scripts/bootstrap-nix-macos.sh once, and ./scripts/build-client.sh to serve the Web UI."
  '';
}
