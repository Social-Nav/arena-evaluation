from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .io import load_episode, save_json
from .metrics import MetricConfig, MetricEngine, ScoringProfile


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena-eval", description="Arena metric engine (VLN + social dynamics).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("compute", help="Compute metrics for a single episode JSON")
    c1.add_argument("episode", type=str, help="Path to episode JSON")
    c1.add_argument("--out", type=str, default="-", help="Output path (default stdout)")
    c1.add_argument("--score", action="store_true", help="Also compute a scalar score")

    c2 = sub.add_parser("schema", help="Print episode JSON schema summary")

    c3 = sub.add_parser("bddl-eval", help="Evaluate an episode against a BDDL task directory")
    c3.add_argument("task_dir", type=str, help="Path to task directory (containing task.yaml + .bddl files)")
    c3.add_argument("episode", type=str, help="Path to episode JSON")
    c3.add_argument("--out", type=str, default="-", help="Output path (default stdout)")

    sub.add_parser("bddl-demo", help="Run the built-in BDDL evaluation demo")

    return p


def _cmd_schema() -> int:
    schema = {
        "meta": {
            "run_id": "str?",
            "episode_id": "str?",
            "scenario_id": "str?",
            "seed": "int?",
            "instruction": "str?",
            "context": "str? (normal|urgent|...)",
            "time_limit_s": "float?",
            "goal_tolerance_m": "float?",
            "robot_radius_m": "float?",
            "human_radius_m": "float?",
            "terminated": "bool?",
            "truncated": "bool?",
            "done_reason": "str?",
        },
        "goal": {"x": "float", "y": "float"},
        "reference_path": "optional list[{x,y}]",
        "frames": [
            {
                "t": "float",
                "robot": {"x": "float", "y": "float", "yaw": "float", "v": "float?", "w": "float?"},
                "pedestrians": [
                    {"id": "str", "x": "float", "y": "float", "vx": "float?", "vy": "float?"}
                ],
                "events": "dict (optional; e.g. {collision: bool, collision_with: str})",
            }
        ],
    }
    sys.stdout.write(json.dumps(schema, indent=2) + "\n")
    return 0


def _cmd_compute(args: argparse.Namespace) -> int:
    ep = load_episode(args.episode)
    engine = MetricEngine(MetricConfig())
    scoring = ScoringProfile() if args.score else None
    metrics = engine.compute(ep, scoring=scoring)

    out = args.out
    if out == "-":
        sys.stdout.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    else:
        save_json(metrics, pathlib.Path(out))
    return 0


def _cmd_bddl_eval(args: argparse.Namespace) -> int:
    from .bddl import BDDLEvaluator

    ep = load_episode(args.episode)
    evaluator = BDDLEvaluator.from_task_dir(args.task_dir)
    result = evaluator.evaluate(ep)

    out = args.out
    if out == "-":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        save_json(result.to_dict(), pathlib.Path(out))
    return 0


def _cmd_bddl_demo() -> int:
    from .bddl.demo import main as demo_main

    demo_main()
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "schema":
        raise SystemExit(_cmd_schema())
    if args.cmd == "compute":
        raise SystemExit(_cmd_compute(args))
    if args.cmd == "bddl-eval":
        raise SystemExit(_cmd_bddl_eval(args))
    if args.cmd == "bddl-demo":
        raise SystemExit(_cmd_bddl_demo())

    raise SystemExit(2)