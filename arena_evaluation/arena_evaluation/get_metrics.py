#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import arena_evaluation.scripts.metrics as Metrics
from ament_index_python.packages import get_package_share_directory


def _resolve_metrics_dir(relative_dir: str) -> str:
    data_root = Path(get_package_share_directory("arena_evaluation")) / "data"
    requested = data_root / relative_dir

    if (requested / "params.yaml").exists():
        return str(requested)

    manifest_path = requested / "run_manifest.yaml"
    if not manifest_path.exists():
        return str(requested)

    manifest_mtime = manifest_path.stat().st_mtime
    candidates = []
    for child in data_root.iterdir():
        if not child.is_dir() or not (child / "params.yaml").exists():
            continue
        mtime = child.stat().st_mtime
        if mtime >= manifest_mtime:
            candidates.append((mtime, child))

    if candidates:
        candidates.sort(reverse=True)
        return str(candidates[0][1])

    return str(requested)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="Directory where the data is stored")
    parser.add_argument("--pedsim", action="store_const", const=True, default=False, help="Flag to enable Pedsim metrics")
    arguments = parser.parse_args()

    dir_arg = _resolve_metrics_dir(arguments.dir)

    if arguments.pedsim:
        metrics = Metrics.PedsimMetrics(dir=dir_arg)
    else:
        metrics = Metrics.Metrics(dir=dir_arg)

    # Save the calculated metrics to a CSV file
    metrics.data.to_csv(os.path.join(dir_arg, "metrics.csv"))

if __name__ == "__main__":
    main()
