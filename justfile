# Quarry — build, lint, test recipes (no side effects on external services)
# Dependencies: quarry-parse → quarry-core, quarry-graph independent
#
# Usage:
#   just sync         # uv sync — fast dev iteration
#   just dev          # cargo debug build all crates
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

# Debug build quarry-parse binary
dev-parse:
    cargo build --manifest-path quarry-parse/Cargo.toml --bin quarry-parse
    @echo "quarry-parse built (debug): quarry-parse/target/debug/quarry-parse"

# Debug build quarry-graph (.so for Python)
dev-graph:
    cargo build --manifest-path quarry-graph/Cargo.toml --features extension-module
    @echo "quarry-graph built (debug)"

# Debug build all crates
dev: dev-core dev-parse dev-graph

# ── Release ──

# Release build quarry-core
release-core:
    cargo build --manifest-path quarry-core/Cargo.toml --features extension-module --release
    @echo "quarry-core built (release)"

# Release build quarry-parse binary
release-parse:
    cargo build --manifest-path quarry-parse/Cargo.toml --bin quarry-parse --release
    @echo "quarry-parse built (release): quarry-parse/target/release/quarry-parse"

# Release build quarry-graph
release-graph:
    cargo build --manifest-path quarry-graph/Cargo.toml --features extension-module --release
    @echo "quarry-graph built (release)"

# Release build all crates
release: release-core release-parse release-graph

# ── Clean ──

# Clean all Rust crate targets
clean:
    cargo clean --manifest-path quarry-core/Cargo.toml
    cargo clean --manifest-path quarry-parse/Cargo.toml
    cargo clean --manifest-path quarry-graph/Cargo.toml

# Clean + release rebuild
rebuild: clean release

# ── Lint & Test ──

# Clippy quarry-core
clippy-core:
    cargo clippy --manifest-path quarry-core/Cargo.toml --all-features -- -D warnings

# Clippy quarry-parse
clippy-parse: clippy-core
    cargo clippy --manifest-path quarry-parse/Cargo.toml -- -D warnings

# Clippy quarry-graph
clippy-graph:
    cargo clippy --manifest-path quarry-graph/Cargo.toml --all-features -- -D warnings

# Clippy all crates
clippy: clippy-parse clippy-graph

# Test quarry-core
test-core:
    cargo test --manifest-path quarry-core/Cargo.toml

# Test quarry-parse
test-parse: test-core
    cargo test --manifest-path quarry-parse/Cargo.toml

# Test quarry-graph
test-graph:
    cargo test --manifest-path quarry-graph/Cargo.toml

# Test all crates
test: test-parse test-graph

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
