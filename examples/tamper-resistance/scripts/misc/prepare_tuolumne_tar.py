"""Prepare the isolated ROCm runtime and effective configs for one TAR replay."""

import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    example = Path(__file__).resolve().parents[2]
    repo = example.parents[1]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "source", "logs", "models", "evals", "launches"):
        (run / directory).mkdir()
    lock = tomllib.loads((example / "uv.lock").read_text())
    constraints = sorted(
        f'{package["name"]}=={package["version"]}'
        for package in lock["package"]
        if "registry" in package["source"]
    )
    excluded = [
        "xformers",
        "triton",
        "fla-core",
        "flash-linear-attention",
    ]
    common = [
        "lilpipe-tamper-resistance",
        "torch==2.12.1+rocm7.2",
        "triton-rocm==3.7.1",
        "sentencepiece>=0.2,<0.3",
        "protobuf>=6,<7",
    ]
    torchao_url = next(
        wheel["url"]
        for package in lock["package"]
        if package["name"] == "torchao"
        for wheel in package["wheels"]
        if "py3-none-any" in wheel["url"]
    )
    for stage, dependencies in {
        "train": ["axolotl==0.18.0", "torchao==0.17.0"],
        "eval": [
            "lm-eval[hf]==0.4.11",
            "transformers==5.14.1",
            "matplotlib>=3.9,<4",
            "seaborn>=0.13,<0.14",
        ],
    }.items():
        runtime = example / "runtime" / "tuolumne" / stage
        if runtime.exists():
            continue
        runtime.mkdir(parents=True, exist_ok=False)
        dev_extra = (
            "[project.optional-dependencies]\n"
            'dev = ["black==25.1.0", "pathspec==0.12.1"]\n\n'
            if stage == "train"
            else ""
        )
        torchao_source = (
            f'torchao = {{ url = "{torchao_url}" }}\n' if stage == "train" else ""
        )
        spec = (
            "[project]\n"
            f'name = "lilpipe-tar-tuo-{stage}"\nversion = "0.1.0"\n'
            'requires-python = ">=3.12,<3.13"\n'
            f"dependencies = {json.dumps(common + dependencies, indent=2)}\n\n"
            + dev_extra
            + "[tool.uv]\nenvironments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
            f"exclude-dependencies = {json.dumps(excluded)}\n"
            f"constraint-dependencies = {json.dumps(constraints, indent=2)}\n\n"
            "[tool.uv.sources]\n"
            + torchao_source
            + 'lilpipe-tamper-resistance = { path = "../../..", editable = true }\n'
            'torch = { index = "pytorch-rocm72" }\n'
            'triton-rocm = { index = "pytorch-rocm72" }\n\n'
            '[[tool.uv.index]]\nname = "pytorch-rocm72"\n'
            'url = "https://download.pytorch.org/whl/rocm7.2"\nexplicit = true\n'
        )
        (runtime / "pyproject.toml").write_text(spec)
    installer_source = Path(
        "/p/vast1/kirchenb/llnl-tools/install_tuolumne_mm_uv_env.sh"
    )
    installer = installer_source.read_text()
    needle = 'UV="${CONDA_ENV_PREFIX}/bin/uv"'
    assert installer.count(needle) == 1
    installer = installer.replace(
        needle,
        'mm_activate "${CONDA_ENV_PREFIX}"\n' + needle,
    )
    (run / "install_environment.sh").write_text(installer)
    shutil.copytree(
        example / "src",
        run / "source" / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(example / "lm_eval_tasks", run / "source" / "lm_eval_tasks")
    shutil.copy2(example / "uv.lock", run / "source" / "reference.uv.lock")
    provenance = {
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "reference_defense_eval_commit": "8e90c520",
        "reference_attack_eval_commit": "f7461ae1",
        "model_revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
        "optional_gpu_dependencies_excluded": excluded,
        "installer_source": str(installer_source),
        "installer_change": "Activate the new absolute prefix before invoking its Python or uv.",
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (run / "status.json").write_text(
        json.dumps({"stage": "runtime_preparation", "run_dir": str(run)}, indent=2)
        + "\n"
    )
    print(run)


if __name__ == "__main__":
    main()
