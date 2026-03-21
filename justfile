# Quarry — Rust crate build & test recipes
# Dependencies: quarry-build → quarry-core, quarry-graph independent

crates := "quarry-core quarry-build quarry-graph"
venv := ".venv/lib/python3.13/site-packages"
so_suffix := ".cpython-313-x86_64-linux-gnu.so"

# List available recipes
default:
    @just --list

# ── Build ──

# Build quarry-core .so (cargo build + direct copy, bypasses maturin wheel cache)
build-core:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --manifest-path quarry-core/Cargo.toml --features extension-module
    mkdir -p {{ venv }}/quarry_core
    cp quarry-core/target/debug/libquarry_core.so {{ venv }}/quarry_core/quarry_core{{ so_suffix }}
    echo "quarry_core .so installed"

# Build quarry-build .so (depends on core)
build-build: build-core
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --manifest-path quarry-build/Cargo.toml --features extension-module
    mkdir -p {{ venv }}/quarry_build
    cp quarry-build/target/debug/libquarry_build.so {{ venv }}/quarry_build/quarry_build{{ so_suffix }}
    echo "quarry_build .so installed"

# Build quarry-graph .so
build-graph:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --manifest-path quarry-graph/Cargo.toml --features extension-module
    mkdir -p {{ venv }}/quarry_graph
    cp quarry-graph/target/debug/libquarry_graph.so {{ venv }}/quarry_graph/quarry_graph{{ so_suffix }}
    echo "quarry_graph .so installed"

# Build all crates (respects dependency order)
build-all: build-build build-graph

# ── Clean ──

# Clean all Rust crate targets
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in {{ crates }}; do
        echo "=== clean $d ==="
        cargo clean --manifest-path "$d/Cargo.toml"
    done

# Clean + rebuild all
rebuild: clean build-all

# ── Lint & Test ──

# Clippy quarry-core
clippy-core:
    cargo clippy --manifest-path quarry-core/Cargo.toml --all-features -- -D warnings

# Clippy quarry-build (depends on core)
clippy-build: clippy-core
    cargo clippy --manifest-path quarry-build/Cargo.toml --all-features -- -D warnings

# Clippy quarry-graph
clippy-graph:
    cargo clippy --manifest-path quarry-graph/Cargo.toml --all-features -- -D warnings

# Clippy all crates
clippy: clippy-build clippy-graph

# Test quarry-core
test-core:
    cargo test --manifest-path quarry-core/Cargo.toml

# Test quarry-build (depends on core)
test-build: test-core
    cargo test --manifest-path quarry-build/Cargo.toml

# Test quarry-graph
test-graph:
    cargo test --manifest-path quarry-graph/Cargo.toml

# Test all crates
test: test-build test-graph

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
    echo "Killing Dagster processes: $pids"
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 2
    # Force kill survivors
    pids=$(pgrep -f 'dagster|dg dev' -u "$USER" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Force killing: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
    echo "Dagster processes stopped"

# ── Verify ──

# Verify installed .so signatures match source
verify-so:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== quarry_core ==="
    .venv/bin/python -c "import quarry_core; print(dir(quarry_core))"
    echo "=== quarry_build.build_oa_s3_pg ==="
    .venv/bin/python -c "import quarry_build, inspect; print(inspect.signature(quarry_build.build_oa_s3_pg))"
    echo "=== quarry_graph ==="
    .venv/bin/python -c "import quarry_graph; print(dir(quarry_graph))"
