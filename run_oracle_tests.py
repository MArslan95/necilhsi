from __future__ import annotations

"""Run the two post-hoc decision-information oracle tests from saved checkpoints.

This script never trains the model and never writes model/checkpoint state.  Test 2
intentionally reads historical TRAIN rows only through the guarded oracle API, so
its numbers are diagnostic and must not be reported as NECIL method results.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from main import _build_data, resolve_device, set_seed
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from oracle_decision_head import run_test1_test2_suite


def _load_config(path: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise RuntimeError("experiment config must contain one JSON object")
    return dict(value)


def _build_finalized_trainer(
    config: Mapping[str, Any],
    checkpoint: str,
    *,
    device_override: str | None,
) -> Trainer:
    args = SimpleNamespace(**dict(config))
    if device_override:
        args.device = str(device_override)
    args.device = resolve_device(str(args.device))
    set_seed(int(args.seed), bool(args.deterministic))
    dataset, _, _ = _build_data(args)
    model = NECILModel(args)
    trainer = Trainer(model, dataset, args)
    trainer.load_checkpoint(str(Path(checkpoint).expanduser().resolve()))
    return trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only Phase-0 and Phase-1 analytic-head oracle diagnostics."
    )
    parser.add_argument("--config", required=True, help="experiment_config.json from the matching run")
    parser.add_argument("--base_checkpoint", required=True, help="finalized Phase-0 checkpoint.pth")
    parser.add_argument("--phase1_checkpoint", required=True, help="finalized Phase-1 checkpoint.pth")
    parser.add_argument("--output", required=True, help="destination oracle_decision_report.json")
    parser.add_argument("--device", default="", help="optional device override, e.g. cuda:0 or cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ridge_scale", type=float, default=1.0e-3)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    config = _load_config(cli.config)
    base = _build_finalized_trainer(
        config,
        cli.base_checkpoint,
        device_override=cli.device or None,
    )
    phase1 = _build_finalized_trainer(
        config,
        cli.phase1_checkpoint,
        device_override=cli.device or None,
    )
    report = run_test1_test2_suite(
        base_trainer=base,
        phase1_trainer=phase1,
        batch_size=int(cli.batch_size),
        ridge_scale=float(cli.ridge_scale),
        output_json=cli.output,
    )

    t1 = report["test1"]
    t2 = report["test2"]
    base_current = 100.0 * float(t1["current_head"]["all_seen"]["balanced_accuracy"])
    base_oracle = 100.0 * float(t1["oracle"]["class_balanced"]["metrics"]["all_seen"]["balanced_accuracy"])
    old_current = 100.0 * float(t2["current_head"]["old_all_seen"]["balanced_accuracy"])
    new_current = 100.0 * float(t2["current_head"]["new_all_seen"]["balanced_accuracy"])
    new_only = 100.0 * float(t2["current_head"]["new_only"]["balanced_accuracy"])
    oracle_metrics = t2["oracle"]["class_balanced"]["metrics"]
    old_oracle = 100.0 * float(oracle_metrics["old"]["balanced_accuracy"])
    new_oracle = 100.0 * float(oracle_metrics["new"]["balanced_accuracy"])
    all_oracle = 100.0 * float(oracle_metrics["all_seen"]["balanced_accuracy"])
    closure = t2["comparison"].get("oracle_new_gap_closure_fraction")

    print("\n[Oracle Test 1 — Phase 0]")
    print(f"Current BA={base_current:.2f}% | Analytic oracle BA={base_oracle:.2f}%")
    print("\n[Oracle Test 2 — Phase 1]")
    print(f"Current OldAll={old_current:.2f}% | NewAll={new_current:.2f}% | NewOnly={new_only:.2f}%")
    print(f"Oracle  OldAll={old_oracle:.2f}% | NewAll={new_oracle:.2f}% | AllSeenBA={all_oracle:.2f}%")
    print(f"New-class decision-gap closure={closure if closure is not None else 'n/a'}")
    print(f"\nDiagnostic report: {report['output_json']}")
    print("WARNING: Test 2 uses historical TRAIN rows and is diagnostic only; do not report it as NECIL performance.")


if __name__ == "__main__":
    main()
