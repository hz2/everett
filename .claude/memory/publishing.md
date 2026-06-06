---
name: publishing
description: crates.io publish workflow and version bump procedure
metadata:
  type: project
---

Publishing checklist:
1. Bump `version` in Cargo.toml
2. `git add Cargo.toml Cargo.lock && git commit -m "chore: bump version to X.Y.Z"`
3. `git push`
4. `nix develop --command cargo publish`
5. `git tag vX.Y.Z && git push origin vX.Y.Z`

**Why:** the tag triggers the `release-check` CI job which dry-runs the publish as a sanity check. Always tag after publishing so the tag reflects what's actually on crates.io.

**How to apply:** do not use `cargo publish --dry-run` with uncommitted changes — cargo requires a clean tree unless `--allow-dirty` is passed. Commit first, then publish.

The crates.io login token is stored in `~/.cargo/credentials.toml` via `cargo login`. Never paste tokens into chat.
