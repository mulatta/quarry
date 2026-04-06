"""Shared subprocess helpers for pipeline assets."""

import subprocess

from dagster import AssetExecutionContext


def run(
    cmd: list[str],
    context: AssetExecutionContext,
    label: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run subprocess, stream stderr to Dagster log, raise on failure."""
    context.log.info(label or f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.rstrip("\n")
        if line:
            context.log.info(line)
    proc.wait()
    assert proc.stdout is not None
    stdout = proc.stdout.read()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    if stdout.strip():
        import json

        try:
            stats = json.loads(stdout)
            for k, v in stats.items():
                context.log.info(f"  {k}: {v}")
        except json.JSONDecodeError:
            pass


def run_parse(args: list[str], context: AssetExecutionContext) -> None:
    """Run quarry-parse subprocess."""
    run(["quarry-parse"] + args, context)
