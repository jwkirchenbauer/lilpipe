"""Prespecify the aggregate, freeze its source and emit a launcher dry run."""

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from factorial_static_report import CELLS, DEFINITIONS
from static_recipe import dump, file_identity, read, sha


EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2"
LAUNCHER = "/p/vast1/kirchenb/llnl-tools/launch_tuo.py"


def inputs(run):
    code = Path(__file__).resolve().parent
    require_new = code / "inputs.json"
    assert not require_new.exists()
    program = read(run / "study.json")
    original = Path(program["original_run"])
    recorded = Path(program["baseline_dynamics"])
    main = run.parents[3] / "artifacts/tuolumne/tar-hparams/20261116T2310Z"
    static = original / "loss-landscape/defense-step125-random-seed42-v3"
    legacy = read(static / "provenance.json")
    entries = [
        {
            "cell": "B",
            "original_run": str(original),
            "baseline_dynamics": str(recorded),
            "static_folder": str(static),
            "get_ll_source": str(main / "source/profiles/static-v1/get_ll.py"),
            "checkpoint_files": file_identity(legacy["checkpoint"]),
        }
    ]
    pinned = {
        str(path): sha(path)
        for path in (
            static / "grid.csv",
            static / "provenance.json",
            static / "validation.json",
            original / "training_validation.json",
            recorded / "training-validation.json",
            recorded / "geometry/geometry.json",
        )
    }
    for cell in CELLS[1:]:
        source = run if cell in ("RI", "RO", "IO") else main
        name = f"hparam-{cell.lower()}-defense"
        folder = source / "cases" / name
        target = source / "static/cases" / name
        entry = {
            "cell": cell,
            "case_folder": str(folder),
            "static_folder": str(target),
            "get_ll_source": str(source / "source/profiles/static-v1/get_ll.py"),
            "baseline_observations": str(
                recorded / "models/defense/observed-batches.jsonl"
            ),
        }
        entries.append(entry)
        if source == main:
            for path in (
                target / "grid.csv",
                target / "protocol.json",
                target / "validation.json",
                folder / "training-validation.json",
                folder / "config.yml",
                folder / "case.json",
            ):
                pinned[str(path)] = sha(path)
    dump(
        require_new,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run": str(run),
            "cells": entries,
            "pinned_files": pinned,
            "new_pair_grid_results_read": False,
            "legacy_B_evidence": "Original passed defense125/K16 proof, recorded defense metrics identical and endpoint distance0. Model file identities frozen at comparison design; tokenizer/collation equality to original B is recorded in all guarded static workers.",
        },
    )
    dump(
        code / "design.json",
        {
            "prespecified_utc": datetime.now(timezone.utc).isoformat(),
            "definitions": DEFINITIONS,
            "new_pair_grid_results_read": False,
            "figure_cells": list(CELLS),
            "focus_radii": [125, 100],
            "no_attacks_or_benchmarks_required": True,
        },
    )
    print("Prespecified8-cell inputs and metrics without reading any new pair grid.")


def freeze_and_dryrun(run):
    assert Path(os.environ["CONDA_PREFIX"]).resolve() == Path(EVAL).resolve()
    code = Path(__file__).resolve().parent
    assert (code / "inputs.json").exists() and not (code / "manifest.sha256").exists()
    wrapper = code / "run.sh"
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${{FLUX_JOB_ID:-}}" ]]
run=${{1:?absolute factorial study required}}
[[ "$run" == /* ]]
code=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
mm_activate {EVAL}
cd {shlex.quote(str(run.parents[3]))}
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=8
mkdir -p "$run/static/cache"
export MPLCONFIGDIR=$(mktemp -d "$run/static/cache/factorial-report-matplotlib-XXXXXXXX")
exec 9>"$run/status/static-factorial-report.lock"
flock -n 9
python "$code/factorial_static_report.py" validate "$run"
exec python "$code/factorial_static_report.py" report "$run"
"""
    )
    profile = {
        "entry": str(wrapper),
        "environment": EVAL,
        "device_policy": "CPU-only rendering; no Torch/model imports in aggregate",
        "inputs_sha256": sha(code / "inputs.json"),
        "design_sha256": sha(code / "design.json"),
        "original_static_recipe_helper_sha256": sha(code / "static_recipe.py"),
        "coordinate_recovery": "Exactly equal FP32 coefficient bits required; losses untouched",
        "no_submission_authorized_by_this_preparation": True,
    }
    dump(code / "profile.json", profile)
    name = "static-full-factorial-report-v1"
    directory = run / "launches" / name
    assert not (directory / "launch.sh").exists()
    argv = [
        "python",
        LAUNCHER,
        f"--environment={EVAL}",
        "--env_act_style=mm_activate",
        f"--output_dir={run / 'launches'}",
        f"--sub_output_dir_name={name}",
        f"--run_name=tar-{name}",
        "--nodes=1",
        "--tasks_per_node=1",
        "--gpus_per_node=1",
        "--cpus_per_task=96",
        "--minutes=10",
        "--qos=pdebug",
        "--bank=verival",
        "--torchrun=false",
        "--pass_run_name=false",
        "--quiet_launch=true",
        "--quiet_localization=true",
        "--squonce=false",
        "--rm_core_dumps=false",
        "--dryrun",
        "--",
        "bash",
        "-i",
        str(wrapper),
        str(run),
    ]
    log = run / "logs/factorial-static-report-dryrun.log"
    with log.open("x") as stream:
        subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert "No jobs were launched." in text
    submission = next(
        line for line in text.splitlines() if line.startswith("Submission: ")
    )
    for path in (wrapper, directory / "launch.sh", directory / "subrunner.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    dump(
        code / "dryrun.json",
        {
            "state": "dryrun-complete-not-submitted",
            "minutes": 10,
            "queue": "pdebug",
            "bank": "verival",
            "argv_dryrun": argv,
            "submission_argv_after_all_eight_caches_complete_and_reviewed": shlex.split(
                submission.removeprefix("Submission: ")
            ),
            "launch_directory": str(directory),
            "dryrun_log": str(log),
            "requires_completed": ["B", "R", "I", "O", "RI", "RO", "IO", "RIO"],
            "worker_guard": "Validate all defense/static/protocol/checkpoint/source/probe/order identities before rendering. No A/evaluation inputs.",
        },
    )
    paths = sorted(path for path in code.iterdir() if path.is_file())
    (code / "manifest.sha256").write_text(
        "".join(f"{sha(path)}  {path}\n" for path in paths)
    )
    for path in code.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(
        "Frozen comparison-v1 and10min CPU launcher dry run only; report requires all8 complete caches."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "freeze-dryrun"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    (inputs if args.stage == "inputs" else freeze_and_dryrun)(args.run.resolve())
