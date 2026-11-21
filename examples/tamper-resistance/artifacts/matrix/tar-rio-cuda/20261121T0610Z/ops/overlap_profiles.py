"""Release a profile after the previously validated immutable-input gate passes."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT.parents[1] / "tar-rio/20261120T2120Z/source/transition/overlap_profiles.py"


def require_profile_reference(root, case):
    program = json.loads((root / "study.json").read_text())
    reference = program["profile_reference_case"]
    if case == reference:
        return
    status = root / "cases" / reference / "status/step-profiles.json"
    if (
        not status.is_file()
        or json.loads(status.read_text()).get("state") != "complete"
    ):
        raise ValueError(
            f"Shared profile reference {reference} must finish before releasing {case}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("rio-k16-cuda-lr1e-6", "rio-k64-cuda-lr1e-6"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require_profile_reference(ROOT, args.case)
    spec = importlib.util.spec_from_file_location("validated_overlap_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    path = ROOT / "matrix-jobs.json"
    ledger = json.loads(path.read_text())
    profile = ledger["profiles:" + args.case]
    analysis = ledger["analysis:" + args.case]
    report = ledger["report"]
    folder = ROOT / "cases" / args.case
    proof = gate.readiness(folder)
    details = subprocess.check_output(
        ["scontrol", "show", "job", profile["job_id"], "-o"], text=True
    )
    assert "JobState=PENDING " in details, details
    final = subprocess.check_output(
        ["scontrol", "show", "job", report["job_id"], "-o"], text=True
    )
    assert (
        "afterok:" + analysis["job_id"] in final
    ), "Final report must retain full-analysis dependency"
    proof.update(
        case=args.case,
        profile_job=profile["job_id"],
        analysis_job=analysis["job_id"],
        final_report_job=report["job_id"],
        verified_epoch=time.time(),
        before_dependencies=profile["dependencies"],
        applied=args.apply,
        readiness_source_sha256=gate.digest(GATE),
    )
    if args.apply:
        subprocess.run(
            ["scontrol", "update", "JobId=" + profile["job_id"], "Dependency=0"],
            check=True,
        )
        profile["dependencies"] = []
        profile.setdefault("dependency_amendments", []).append(
            {
                "reason": "Immutable geometry and complete loss inputs verified; independent profile can overlap remaining grids. Final report retains full analysis.",
                "readiness": proof,
            }
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(path)
        (folder / "status/profile-overlap-readiness.json").write_text(
            json.dumps(proof, indent=2) + "\n"
        )
    output = (
        ROOT
        / "ops"
        / (
            "profile-overlap-"
            + args.case
            + ("-applied" if args.apply else "-dryrun")
            + ".json"
        )
    )
    output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == "__main__":
    main()
