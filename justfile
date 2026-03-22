# Quarry — Rust crate build, test, and install recipes
# Dependencies: quarry-build → quarry-core, quarry-graph independent
#
# Usage:
#   just sync         # uv sync (debug) — fast dev iteration
#   just sync-release # uv sync (release) — production
#   just dev          # cargo debug build + install .so to venv
#   just release      # cargo release build + install .so to venv
#   just check        # clippy + test all
#   just kill-dagster # stop all dagster processes

venv := ".venv/lib/python3.13/site-packages"
so_suffix := ".cpython-313-x86_64-linux-gnu.so"

# List available recipes
default:
    @just --list

# ── Sync (uv + maturin) ──

# Sync Python deps + Rust .so (debug build)
sync:
    uv sync --all-extras

# Sync Python deps + Rust .so (release build)
sync-release:
    uv sync --all-extras -C build-args='--profile=release'

# ── Internal helpers ──

[private]
_build crate profile:
    #!/usr/bin/env bash
    set -euo pipefail
    flags=(--manifest-path "{{crate}}/Cargo.toml" --features extension-module)
    [[ "{{profile}}" == "release" ]] && flags+=(--release)
    cargo build "${flags[@]}"

[private]
_install crate pymod profile:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ venv }}/{{pymod}}"
    cp "{{crate}}/target/{{profile}}/lib{{pymod}}.so" \
       "{{ venv }}/{{pymod}}/{{pymod}}{{ so_suffix }}"
    echo "{{pymod}} .so installed ({{profile}})"

# ── Dev (debug) ──

# Debug build + install quarry-core
dev-core:
    just _build quarry-core debug
    just _install quarry-core quarry_core debug

# Debug build + install quarry-build (depends on core)
dev-build: dev-core
    just _build quarry-build debug
    just _install quarry-build quarry_build debug

# Debug build + install quarry-graph
dev-graph:
    just _build quarry-graph debug
    just _install quarry-graph quarry_graph debug

# Debug build + install all crates
dev: dev-build dev-graph

# ── Release ──

# Release build + install quarry-core
release-core:
    just _build quarry-core release
    just _install quarry-core quarry_core release

# Release build + install quarry-build (depends on core)
release-build: release-core
    just _build quarry-build release
    just _install quarry-build quarry_build release

# Release build + install quarry-graph
release-graph:
    just _build quarry-graph release
    just _install quarry-graph quarry_graph release

# Release build + install all crates
release: release-build release-graph

# ── Clean ──

# Clean all Rust crate targets
clean:
    cargo clean --manifest-path quarry-core/Cargo.toml
    cargo clean --manifest-path quarry-build/Cargo.toml
    cargo clean --manifest-path quarry-graph/Cargo.toml

# Clean + release rebuild
rebuild: clean release

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

# ── Verify ──

# Verify installed .so signatures match source
verify-so:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== quarry_core ==="
    .venv/bin/python -c "import quarry_core; print(dir(quarry_core))"
    echo "=== quarry_build.build_pubmed_pg ==="
    .venv/bin/python -c "import quarry_build, inspect; print(inspect.signature(quarry_build.build_pubmed_pg))"
    echo "=== quarry_build.build_oa_s3_pg ==="
    .venv/bin/python -c "import quarry_build, inspect; print(inspect.signature(quarry_build.build_oa_s3_pg))"
    echo "=== quarry_graph ==="
    .venv/bin/python -c "import quarry_graph; print(dir(quarry_graph))"
