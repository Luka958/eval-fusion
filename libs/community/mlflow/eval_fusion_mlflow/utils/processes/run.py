from __future__ import annotations

from subprocess import run

from eval_fusion_core.exceptions import EvalFusionException


def run_process(args: list[str]):
    completed_process = run(args, capture_output=True, text=True)

    if completed_process.returncode != 0:
        raise EvalFusionException(completed_process.stderr)
