# Quarry — build, lint, test recipes (no side effects on external services)
# Dependencies: quarry-parse → quarry-core, quarry-graph independent
#
# Usage:
#   just sync         # uv sync — fast dev iteration
#   just dev          # cargo debug build all crates
#   just check        # clippy + test all
#   just cleanup      # stop all project processes (dagster, PG, CH)

# List available recipes
default:
    @just --list

# ── Sync (uv) ──

# Sync Python deps + rebuild Rust extensions (uv sync overwrites maturin .so)
sync:
    uv sync --all-extras
    maturin develop --release -m crates/quarry-core/Cargo.toml
    maturin develop --release -m crates/quarry-graph/Cargo.toml

# ── Dev (debug) ──

# Debug build quarry-core (.so for Python)
dev-core:
    cargo build --manifest-path crates/quarry-core/Cargo.toml --features extension-module
    @echo "quarry-core built (debug)"

# Debug build quarry-parse binary
dev-parse:
    cargo build --manifest-path crates/quarry-parse/Cargo.toml --bin quarry-parse
    @echo "quarry-parse built (debug): crates/quarry-parse/target/debug/quarry-parse"

# Debug build quarry-graph (.so for Python)
dev-graph:
    cargo build --manifest-path crates/quarry-graph/Cargo.toml --features extension-module
    @echo "quarry-graph built (debug)"

# Debug build all crates
dev: dev-core dev-parse dev-graph

# ── Release ──

# Release build quarry-core
release-core:
    cargo build --manifest-path crates/quarry-core/Cargo.toml --features extension-module --release
    @echo "quarry-core built (release)"

# Release build quarry-parse binary
release-parse:
    cargo build --manifest-path crates/quarry-parse/Cargo.toml --bin quarry-parse --release
    @echo "quarry-parse built (release): crates/quarry-parse/target/release/quarry-parse"

# Release build quarry-graph
release-graph:
    cargo build --manifest-path crates/quarry-graph/Cargo.toml --features extension-module --release
    @echo "quarry-graph built (release)"

# Release build all crates
release: release-core release-parse release-graph

# ── Clean ──

# Clean all Rust crate targets
clean:
    cargo clean --manifest-path crates/quarry-core/Cargo.toml
    cargo clean --manifest-path crates/quarry-parse/Cargo.toml
    cargo clean --manifest-path crates/quarry-graph/Cargo.toml

# Clean + release rebuild
rebuild: clean release

# ── Lint & Test ──

# Clippy quarry-core
clippy-core:
    cargo clippy --manifest-path crates/quarry-core/Cargo.toml --all-features -- -D warnings

# Clippy quarry-parse
clippy-parse: clippy-core
    cargo clippy --manifest-path crates/quarry-parse/Cargo.toml -- -D warnings

# Clippy quarry-graph
clippy-graph:
    cargo clippy --manifest-path crates/quarry-graph/Cargo.toml --all-features -- -D warnings

# Clippy all crates
clippy: clippy-parse clippy-graph

# Test quarry-core
test-core:
    cargo test --manifest-path crates/quarry-core/Cargo.toml

# Test quarry-parse
test-parse: test-core
    cargo test --manifest-path crates/quarry-parse/Cargo.toml

# Test quarry-graph
test-graph:
    cargo test --manifest-path crates/quarry-graph/Cargo.toml

# Test all Rust crates
test: test-parse test-graph

# Test Python integration (requires CH/PG)
test-py:
    python -m pytest tests/integration -v

# Clippy + test all (Rust + Python)
check: clippy test test-py

# ── Process management ──

# Stop all project processes (Dagster, process-compose, PG, CH)
# Only kills processes owned by current user matching project-specific patterns.
cleanup:
    #!/usr/bin/env bash
    set -euo pipefail

    # Project-specific patterns — won't match system services
    patterns=(
        'dagster|dg dev'                      # Dagster grpc, daemon, workers
        'process-compose'                     # process-compose orchestrator
        'postgres.*/tmp/quarry-pg'            # PG with quarry socket dir
        'postgres.*\.pg-data'                 # PG with project data dir
        'clickhouse-server.*/nix/store/'      # CH started via nix (not system pkg)
    )

    all_pids=""
    for pat in "${patterns[@]}"; do
        pids=$(pgrep -f "$pat" -u "$USER" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "[$pat] found: $pids"
            all_pids="$all_pids $pids"
        fi
    done
    all_pids=$(echo "$all_pids" | xargs -n1 | sort -u | xargs)

    if [ -z "$all_pids" ]; then
        echo "No project processes found"
        exit 0
    fi

    echo "SIGTERM: $all_pids"
    echo "$all_pids" | xargs kill 2>/dev/null || true

    for i in 1 2 3 4 5; do
        remaining=""
        for pat in "${patterns[@]}"; do
            pids=$(pgrep -f "$pat" -u "$USER" 2>/dev/null || true)
            [ -n "$pids" ] && remaining="$remaining $pids"
        done
        remaining=$(echo "$remaining" | xargs -n1 | sort -u | xargs)
        [ -z "$remaining" ] && break
        sleep 1
    done

    if [ -n "$remaining" ]; then
        echo "SIGKILL: $remaining"
        echo "$remaining" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi

    # Final check
    still_alive=""
    for pat in "${patterns[@]}"; do
        pids=$(pgrep -f "$pat" -u "$USER" 2>/dev/null || true)
        [ -n "$pids" ] && still_alive="$still_alive $pids"
    done
    still_alive=$(echo "$still_alive" | xargs)

    if [ -n "$still_alive" ]; then
        echo "WARNING: still alive: $still_alive"
        exit 1
    fi
    echo "All project processes stopped"

# ── Evaluation ──

# Prepare seed-recall eval: resolve PMIDs → OpenAlex work_ids (requires PG)
eval-prep:
    python eval/seed-recall/prep.py

# Run seed-recall evaluation (requires PG + CSR graph)
eval-run:
    python eval/seed-recall/run.py

# Show evaluation report
eval-report:
    python eval/seed-recall/report.py
