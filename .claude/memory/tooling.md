---
name: tooling
description: nix-first dev workflow — all commands run through nix develop
metadata:
  type: feedback
---

Always use `nix develop --command <cmd>` to run cargo, rustfmt, clippy, llvm-cov, etc. Never invoke cargo or rustfmt directly.

**Why:** the project pins a specific stable toolchain via rust-toolchain.toml and fenix in flake.nix. Running cargo outside the nix shell can use a different toolchain and produce different results (e.g. rustfmt formatting differences that fail CI).

**How to apply:** prefix every cargo invocation with `nix develop --command`. For miri, use `nix develop .#miri --command`. For the kani shell, use `nix develop .#kani --command`.

Key commands:
- `nix develop --command cargo test` — run tests
- `nix develop --command cargo fmt` — format
- `nix develop --command cargo clippy --all-features -- -D warnings` — lint
- `nix develop --command cargo llvm-cov --all-features --summary-only` — coverage
- `nix flake check` — full CI check (build, clippy, nextest, doctests, fmt, doc)
