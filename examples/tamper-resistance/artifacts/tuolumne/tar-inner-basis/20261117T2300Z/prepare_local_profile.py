"""Freeze exact local transport of the unchanged archived reference inputs."""

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
    previous = run / "source/reduction-gpu-v2"
    profile = run / "source/reference-local-v3"
    profile.mkdir(parents=True, exist_ok=True)
    assert not (profile / "manifest.sha256").exists()
    for name in ("tar_inner_recording.py", "gpu_preflight.py"):
        shutil.copy2(previous / name, profile / name)
    shutil.copy2(
        example / "scripts/misc/tar_reference_cache.py",
        profile / "tar_reference_cache.py",
    )
    shutil.copy2(
        example.parents[1] / "tests/test_tar_reference_cache.py",
        profile / "test_tar_reference_cache.py",
    )
    worker = (previous / "run.sh").read_text()
    worker = replace_once(
        worker,
        'status="$run/status/$mode.json"',
        'status="$run/status/$mode.json"\ncache_info=""\ncleanup_reference_cache() {\n    if [[ -n "$cache_info" && -s "$cache_info/provenance.json" ]]; then\n        python "$profile/tar_reference_cache.py" cleanup --provenance "$cache_info/provenance.json"\n    fi\n}',
    )
    old_trap = next(line for line in worker.splitlines() if line.startswith("trap "))
    new_trap = 'trap \'result=$?; trap - EXIT; if ! cleanup_reference_cache; then result=1; fi; if (( result != 0 )); then printf "{\\"state\\":\\"failed\\",\\"exit_code\\":%d}\\n" "$result" > "$status"; fi; exit "$result"\' EXIT'
    worker = replace_once(worker, old_trap, new_trap)
    worker = worker.replace("inner-gpu-v2", "inner-local-v3")
    old_train = '    axolotl train "$run/cases/baseline-inner-recording/config.yml" --launcher python'
    new_train = '''    cache_key=${FLUX_JOB_ID//[^a-zA-Z0-9_-]/_}-${BASHPID}-${EPOCHSECONDS}
    cache_info="$run/cache-provenance/$cache_key"
    ram_root=${FLUX_JOB_TMPDIR:-${TMPDIR:-/tmp}}
    numactl --interleave=all python "$profile/tar_reference_cache.py" stage \\
        --spec "$run/references.json" --ssd-root "/l/ssd/$USER" \\
        --ram-root "$ram_root" --output "$cache_info" \\
        --reserve-ssd-bytes 21474836480 --max-ram-bytes 240000000000 \\
        --max-model-bytes 450000000000
    export TAR_INNER_REFERENCE_SPEC="$cache_info/references.json"
    numactl --interleave=all axolotl train "$run/cases/baseline-inner-recording/config.yml" --launcher python
    cleanup_reference_cache
    cache_info=""'''
    worker = replace_once(worker, old_train, new_train)
    (profile / "run.sh").write_text(worker)
    shutil.copy2(previous / "validate_replay.py", profile / "validate_replay.py")
    spec = json.loads((previous / "analysis-spec.json").read_text())
    spec["aggregates"] = [
        path.replace("inner-gpu-v2", "inner-local-v3") for path in spec["aggregates"]
    ]
    (profile / "analysis-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    metadata = {
        "purpose": "Exact input transport only; no numerical recorder, defense config, optimizer or data changes",
        "recorder_byte_identical_to": str(previous / "tar_inner_recording.py"),
        "recorder_sha256": hashlib.sha256(
            (profile / "tar_inner_recording.py").read_bytes()
        ).hexdigest(),
        "original_reference_spec": str(run / "references.json"),
        "reference_checkpoints": 26,
        "cache": "Verified temporary SSD/RAM copies under allocation-owned roots; persistent logical-source/transfer hash map in cache-provenance",
        "limits": {
            "total_bytes": 450000000000,
            "ram_bytes": 240000000000,
            "ssd_reserved_bytes": 21474836480,
        },
        "memory_policy": "Interleave CPU file-cache and training host allocations across allowed NUMA nodes; explicit four-GPU model map unchanged",
        "new_inner_root": str(run / "inner-local-v3"),
    }
    (profile / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
