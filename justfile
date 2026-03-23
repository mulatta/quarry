# Quarry — Rust crate build, test, and install recipes
# Dependencies: quarry-ingest → quarry-core, quarry-graph independent
#
# Usage:
#   just sync         # uv sync — fast dev iteration
#   just dev          # cargo debug build quarry-ingest binary
#   just release      # cargo release build quarry-ingest binary
#   just check        # clippy + test all
#   just kill-dagster # stop all dagster processes

# List available recipes
default:
    @just --list

# ── Sync (uv) ──

# Sync Python deps
sync:
    uv sync --all-extras

# ── Dev (debug) ──

# Debug build quarry-core (.so for Python)
dev-core:
    cargo build --manifest-path quarry-core/Cargo.toml --features extension-module
    @echo "quarry-core built (debug)"

# Debug build quarry-ingest binary
dev-ingest:
    cargo build --manifest-path quarry-ingest/Cargo.toml --bin quarry-ingest
    @echo "quarry-ingest built (debug): quarry-ingest/target/debug/quarry-ingest"

# Debug build quarry-graph (.so for Python)
dev-graph:
    cargo build --manifest-path quarry-graph/Cargo.toml --features extension-module
    @echo "quarry-graph built (debug)"

# Debug build all crates
dev: dev-core dev-ingest dev-graph

# ── Release ──

# Release build quarry-core
release-core:
    cargo build --manifest-path quarry-core/Cargo.toml --features extension-module --release
    @echo "quarry-core built (release)"

# Release build quarry-ingest binary
release-ingest:
    cargo build --manifest-path quarry-ingest/Cargo.toml --bin quarry-ingest --release
    @echo "quarry-ingest built (release): quarry-ingest/target/release/quarry-ingest"

# Release build quarry-graph
release-graph:
    cargo build --manifest-path quarry-graph/Cargo.toml --features extension-module --release
    @echo "quarry-graph built (release)"

# Release build all crates
release: release-core release-ingest release-graph

# ── Clean ──

# Clean all Rust crate targets
clean:
    cargo clean --manifest-path quarry-core/Cargo.toml
    cargo clean --manifest-path quarry-ingest/Cargo.toml
    cargo clean --manifest-path quarry-graph/Cargo.toml

# Clean + release rebuild
rebuild: clean release

# ── Lint & Test ──

# Clippy quarry-core
clippy-core:
    cargo clippy --manifest-path quarry-core/Cargo.toml --all-features -- -D warnings

# Clippy quarry-ingest
clippy-ingest: clippy-core
    cargo clippy --manifest-path quarry-ingest/Cargo.toml -- -D warnings

# Clippy quarry-graph
clippy-graph:
    cargo clippy --manifest-path quarry-graph/Cargo.toml --all-features -- -D warnings

# Clippy all crates
clippy: clippy-ingest clippy-graph

# Test quarry-core
test-core:
    cargo test --manifest-path quarry-core/Cargo.toml

# Test quarry-ingest
test-ingest: test-core
    cargo test --manifest-path quarry-ingest/Cargo.toml

# Test quarry-graph
test-graph:
    cargo test --manifest-path quarry-graph/Cargo.toml

# Test all crates
test: test-ingest test-graph

# Clippy + test all
check: clippy test

# ── Dagster ──

# Kill all Dagster processes (grpc, daemon, workers)
kill-dagster:
    #!/usr/bin/env bash
    set -euo pipefail
    pids=$(pgrep -f 'dagster|dg dev' -u "$USER" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "No Dagster processes found"
        exit 0
    fi
    echo "SIGTERM: $pids"
    echo "$pids" | xargs kill 2>/dev/null || true
    for i in 1 2 3 4 5; do
        pids=$(pgrep -f 'dagster|dg dev' -u "$USER" 2>/dev/null || true)
        [ -z "$pids" ] && break
        sleep 1
    done
    pids=$(pgrep -f 'dagster|dg dev' -u "$USER" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "SIGKILL: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    pids=$(pgrep -f 'dagster|dg dev' -u "$USER" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "WARNING: still alive: $pids"
        exit 1
    fi
    echo "Dagster processes stopped"
