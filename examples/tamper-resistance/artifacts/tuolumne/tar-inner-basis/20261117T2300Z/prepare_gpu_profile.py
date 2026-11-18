"""Freeze a GPU-only change to aggregate measurement, preserving old execution inputs."""

import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    example = Path.cwd().resolve()
    run = Path(__file__).resolve().parent
    profile = run / "source/reduction-gpu-v2"
    assert not (profile / "manifest.sha256").exists()
    shutil.copy2(
        example / "scripts/misc/tar_inner_recording.py",
        profile / "tar_inner_recording.py",
    )
    worker = (run / "source/run.sh").read_text()
    worker = replace_once(
        worker,
        'sha256sum --quiet -c "$run/source-manifest.sha256"',
        'profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\nsha256sum --quiet -c "$profile/manifest.sha256"\nsha256sum --quiet -c "$run/source-manifest.sha256"',
    )
    worker = replace_once(
        worker,
        'export PYTHONPATH="$run/source/misc:',
        'export PYTHONPATH="$profile:$run/source/misc:',
    )
    worker = replace_once(
        worker,
        'export TAR_INNER_RECORD_ROOT="$run/inner"',
        'export TAR_INNER_RECORD_ROOT="$run/inner-gpu-v2"\n    export TAR_INNER_REDUCTION_DEVICE=cuda:0 TAR_INNER_CHUNK_ELEMENTS=1048576',
    )
    worker = replace_once(
        worker,
        "    axolotl train ",
        '    python "$profile/gpu_preflight.py" "$run"\n    axolotl train ',
    )
    worker = replace_once(
        worker,
        'python "$run/source/validate_replay.py" "$run"',
        'python "$profile/validate_replay.py" "$run" --inner-root "$run/inner-gpu-v2"',
    )
    worker = replace_once(
        worker, '"$run/analysis-spec.json"', '"$profile/analysis-spec.json"'
    )
    (profile / "run.sh").write_text(worker)
    validation = (run / "source/validate_replay.py").read_text()
    validation = replace_once(
        validation,
        'parser.add_argument("run", type=Path)',
        'parser.add_argument("run", type=Path)\n    parser.add_argument("--inner-root", type=Path)',
    )
    validation = replace_once(
        validation,
        'path = run / f"inner/outer-{outer:06d}.json"',
        'path = (args.inner_root or run / "inner") / f"outer-{outer:06d}.json"',
    )
    (profile / "validate_replay.py").write_text(validation)
    spec = json.loads((run / "analysis-spec.json").read_text())
    spec["aggregates"] = [
        str(run / f"inner-gpu-v2/outer-{outer:06d}.json")
        for outer in (1, 32, 64, 96, 125)
    ]
    (profile / "analysis-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    metadata = {
        "scientific_config_unchanged": True,
        "original_source_preserved": str(run / "source-manifest.sha256"),
        "change": "Only aggregate FP64 inner-product reduction uses cuda:0 and 1048576-element chunks; training optimizer and data unchanged.",
        "new_inner_record_root": str(run / "inner-gpu-v2"),
        "old_cpu_attempt": "f3kiN4NibZAw; slow initial aggregate, no completed outer step at profile preparation",
    }
    (profile / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
