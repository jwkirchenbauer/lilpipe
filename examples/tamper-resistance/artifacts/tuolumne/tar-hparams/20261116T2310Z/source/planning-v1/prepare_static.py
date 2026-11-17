"""Freeze the exact original static recipe and emit dry runs, never submissions."""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from static_worker import dump, read, sha


EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2"
LAUNCHER = "/p/vast1/kirchenb/llnl-tools/launch_tuo.py"


def prepare(run):
    assert Path(os.environ["CONDA_PREFIX"]).resolve() == Path(EVAL).resolve()
    code = Path(__file__).resolve().parent
    original = (
        run.parents[3]
        / "artifacts/tuolumne/tar-collaborator/20261116T1725Z/source/profiles/static-v1"
    )
    profile = run / "source/profiles/static-v1"
    profile.mkdir(parents=True, exist_ok=False)
    for filename in ("get_ll.py", "plot_ll_2d.py", "plot_ll_3d.py"):
        shutil.copy2(original / filename, profile / filename)
        assert (profile / filename).read_bytes() == (original / filename).read_bytes()
    for filename in ("static_worker.py", "static_report.py"):
        shutil.copy2(code / filename, profile / filename)
    wrapper = profile / "run.sh"
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${{FLUX_JOB_ID:-}}" ]]
code=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
sha256sum --quiet -c {shlex.quote(str(run / "operational-inputs.sha256"))}
mm_activate {EVAL}
cd {shlex.quote(str(run.parents[3]))}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mode=${{1:?case or report}}
shift
mkdir -p {shlex.quote(str(run / "static/cache"))}
export MPLCONFIGDIR=$(mktemp -d {shlex.quote(str(run / "static/cache/matplotlib-XXXXXXXX"))})
case "$mode" in
  case)
    export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
    exec python "$code/static_worker.py" case "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
    exec python "$code/static_report.py" "$@"
    ;;
  *) exit 2 ;;
esac
"""
    )
    metadata = {
        "original_source": str(original),
        "unchanged_source_files": {
            name: sha(profile / name)
            for name in ("get_ll.py", "plot_ll_2d.py", "plot_ll_3d.py")
        },
        "entry": str(wrapper),
        "environment": EVAL,
        "points_per_case": 648,
        "static_convention": "Original BF16 seed42 directions and arithmetic/default attention. Separate from dynamics FP32 orthonormal/eager convention.",
        "renderer": "CPU child process after numerical process exits",
    }
    dump(profile / "profile.json", metadata)
    paths = sorted(path for path in profile.iterdir() if path.is_file())
    (profile / "manifest.sha256").write_text(
        "".join(f"{sha(path)}  {path}\n" for path in paths)
    )
    for path in profile.iterdir():
        path.chmod(0o444)
    defenses = {
        item["case"]: item["job_id"] for item in read(run / "defense-jobs.json")
    }
    jobs = []
    for condition in ("r", "i", "o", "rio", "comparison"):
        name = f"static-{condition}"
        is_report = condition == "comparison"
        case_name = f"hparam-{condition}-defense"
        arguments = ["report", str(run)] if is_report else ["case", str(run), case_name]
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
            f"--minutes={30 if is_report else 75}",
            f"--qos={'pdebug' if is_report else 'pbatch'}",
            "--bank=verival",
            "--torchrun=false",
            "--pass_run_name=false",
            "--quiet_launch=true",
            "--quiet_localization=true",
            "--squonce=false",
            "--rm_core_dumps=false",
            "--dryrun",
        ]
        if not is_report:
            argv.append(f"--dependency=afterok:{defenses[case_name]}")
        argv.extend(["--", "bash", "-i", str(wrapper), *arguments])
        directory = run / "launches" / name
        assert not (directory / "launch.sh").exists(), directory
        log = run / "logs" / f"dryrun-{name}.log"
        with log.open("x") as stream:
            subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text()
        assert "No jobs were launched." in text
        submission = next(
            line for line in text.splitlines() if line.startswith("Submission: ")
        )
        for filename in ("launch.sh", "subrunner.sh"):
            subprocess.run(["bash", "-n", str(directory / filename)], check=True)
        jobs.append(
            {
                "name": name,
                "minutes": 30 if is_report else 75,
                "environment": EVAL,
                "argv_dryrun": argv,
                "submission_argv_after_review": shlex.split(
                    submission.removeprefix("Submission: ")
                ),
                "requires_completed": (
                    ["static-r", "static-i", "static-o", "static-rio"]
                    if is_report
                    else [f"train:{case_name}"]
                ),
                "launch_directory": str(directory),
                "dryrun_log": str(log),
            }
        )
        print(f"DRY RUN ONLY: {name}", flush=True)
    dump(
        run / "static-dryrun-plan.json",
        {
            "state": "dryruns-complete-not-submitted",
            "profile": metadata,
            "manifest_sha256": sha(profile / "manifest.sha256"),
            "jobs": jobs,
            "report": "B/R/I/O/RIO plus uploaded HF contextual checkpoint: per-radius common absolute z/color/camera and common symmetric center-relative contours; each group contains all six checkpoints.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    prepare(parser.parse_args().run.resolve())
