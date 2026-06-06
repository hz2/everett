{
  description = "everett - a zero-dependency statevector quantum simulator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      fenix,
      flake-utils,
      treefmt-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        fenixPkgs = fenix.packages.${system};

        # stable toolchain pinned via rust-toolchain.toml, used by crane for
        # all production builds and checks.
        stableToolchain = fenixPkgs.fromToolchainFile {
          file = ./rust-toolchain.toml;
          # hash of the pinned toolchain from rust-toolchain.toml. if the pin
          # changes, `nix build` prints the new expected hash to paste here.
          sha256 = "sha256-gh/xTkxKHL4eiRXzWv8KP7vfjSk61Iq48x47BEDFgfk=";
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;

        src = craneLib.cleanCargoSource ./.;
        commonArgs = {
          inherit src;
          strictDeps = true;
        };
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        everett = craneLib.buildPackage (commonArgs // { inherit cargoArtifacts; });

        # nightly toolchain carrying miri + rust-src (cargo-miri builds its own
        # sysroot from rust-src). lives only in the `.#miri` devShell, separate
        # from the stable build toolchain above.
        miriToolchain = fenixPkgs.complete.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "miri"
          "clippy"
          "rustfmt"
        ];

        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
      in
      {
        packages.default = everett;

        checks = {
          inherit everett;

          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-features --all-targets -- --deny warnings";
            }
          );

          nextest = craneLib.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            }
          );

          doctest = craneLib.cargoDocTest (commonArgs // { inherit cargoArtifacts; });

          doc = craneLib.cargoDoc (
            commonArgs
            // {
              inherit cargoArtifacts;
              env.RUSTDOCFLAGS = "--deny warnings";
            }
          );

          formatting = treefmtEval.config.build.check self;
        };

        # `nix develop` - stable toolchain for normal work.
        devShells.default = craneLib.devShell {
          checks = self.checks.${system};
          packages = [
            pkgs.cargo-nextest
            pkgs.cargo-llvm-cov
          ];
        };

        # `nix develop .#miri` - nightly + miri for undefined-behavior checking.
        devShells.miri = pkgs.mkShell {
          buildInputs = [ miriToolchain ];
          shellHook = ''
            echo "everett miri shell (nightly + miri)"
            echo "  MIRIFLAGS=\"-Zmiri-strict-provenance -Zmiri-tree-borrows\" cargo miri test miri_"
          '';
        };

        # `nix develop .#kani` - kani bounded model checker for the index math.
        devShells.kani = pkgs.mkShell {
          buildInputs = [
            stableToolchain
            pkgs.python3
            pkgs.kani
          ];
          shellHook = ''
            echo "everett kani shell"
            echo "  cargo kani"
          '';
        };

        formatter = treefmtEval.config.build.wrapper;
      }
    );
}
