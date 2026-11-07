"""Evaluate unique checkpoints and reuse only verified identical final weights."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch"
)


@pytest.mark.parametrize(
    "max_step,first_step,last_step",
    [(250, 50, 250), (1000, 50, 1000), (1000, 50, 500), (1000, 550, 1000)],
)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("tokenizer_layout", ["legacy", "modern", "mismatched"])
@pytest.mark.parametrize("identical", [True, False])
@pytest.mark.parametrize("include_gsm8k", [False, True])
@pytest.mark.parametrize("save_results", [False, True])
def test_checkpoint_evaluation_and_final_reuse(
    tmp_path,
    identical,
    include_gsm8k,
    save_results,
    tokenizer_layout,
    sharded,
    max_step,
    first_step,
    last_step,
):
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    shared = workdir / "scripts/slurm/tamia/eval_capabilities.sbatch"
    shared.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT.with_name("eval_capabilities.sbatch"), shared)
    wrapper = tmp_path / "seam/scripts/eval_harmfulness.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()
    model = tmp_path / "model"
    for step in range(50, max_step + 1, 50):
        (model / f"checkpoint-{step}").mkdir(parents=True)
    for directory in (model, model / f"checkpoint-{max_step}"):
        directory.mkdir(parents=True, exist_ok=True)
        index = {"weight_map": {"weight": "model.safetensors"}}
        if sharded:
            (directory / "model.safetensors.index.json").write_text(json.dumps(index))
        for name in (
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (directory / name).write_text("identical")
        asset = (
            "special_tokens_map.json"
            if tokenizer_layout == "legacy"
            else "chat_template.jinja"
        )
        (directory / asset).write_text("identical")
    if tokenizer_layout == "mismatched":
        (model / "chat_template.jinja").write_text("different")
    if not identical:
        (model / "model.safetensors").write_text("different")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[2:]\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'stage': sys.argv[1], 'args': args, 'cache': os.environ['HF_DATASETS_CACHE'], 'cuda_cache': os.environ['CUDA_CACHE_PATH']}) + '\\n')\n"
        "if sys.argv[1] == 'capabilities':\n"
        "    output = Path(args[args.index('--output_path') + 1])\n"
        "    output.mkdir(parents=True, exist_ok=True)\n"
        "    if os.environ['SAVE_RESULTS'] == '1':\n"
        "        tasks = args[args.index('--tasks') + 1].split(',')\n"
        "        model = args[args.index('--model_args') + 1].split('pretrained=')[1].split(',')[0]\n"
        "        shots = int(args[args.index('--num_fewshot') + 1])\n"
        "        data = {'results': {t: {'acc,none': 0.1} for t in tasks}, 'config': {'model_args': {'pretrained': model}, 'limit': None}, 'configs': {t: {'num_fewshot': shots} for t in tasks}}\n"
        "        (output / 'results_2026.json').write_text(json.dumps(data))\n"
        "else:\n"
        "    output = Path(args[args.index('--output') + 1])\n"
        "    output.write_text(json.dumps({'model': args[args.index('--batch-size') - 1], 'harmfulness': 0.1, 'samples': 100}))\n"
    )
    seam_python = tmp_path / "env/bin/python"
    seam_python.parent.mkdir(parents=True)
    seam_python.write_text(
        '#!/bin/bash\nif [[ $1 == - ]]; then exec "$TEST_PYTHON" "$@"; fi\n'
        'exec "$TEST_PYTHON" "$TEST_LAUNCHER" harmfulness "$@"\n'
    )
    seam_python.chmod(0o755)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        "uv() { :; }\n"
        'accelerate() { "$TEST_PYTHON" "$TEST_LAUNCHER" capabilities "$@"; }\n'
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "evals"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--checkpoints", str(model), str(output)],
        cwd=workdir,
        env={
            **os.environ,
            "SCRATCH": str(tmp_path),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
            "SEAM_ENV": str(seam_python.parents[1]),
            "BASH_ENV": str(bash_env),
            "TEST_PYTHON": sys.executable,
            "TEST_LAUNCHER": str(launcher),
            "CALLS": str(calls),
            "INCLUDE_GSM8K": "1" if include_gsm8k else "0",
            "SAVE_RESULTS": "1" if save_results else "0",
            "CHECKPOINT_MAX_STEP": str(max_step),
            "CHECKPOINT_START_STEP": str(first_step),
            "CHECKPOINT_END_STEP": str(last_step),
        },
        capture_output=True,
        text=True,
    )
    if not identical or tokenizer_layout == "mismatched":
        assert result.returncode != 0
        assert not calls.exists()
        return
    if not save_results:
        assert result.returncode != 0
        assert "lm-eval did not save results" in result.stderr
        assert not (output / "final-evaluation.json").exists()
        return
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    stages = (
        ["capabilities", "capabilities", "harmfulness"]
        if include_gsm8k
        else ["capabilities", "harmfulness"]
    )
    assert [record["stage"] for record in records] == stages * (
        (last_step - first_step) // 50 + 1
    )
    for index, step in enumerate(range(first_step, last_step + 1, 50)):
        per_step = records[index * len(stages) : (index + 1) * len(stages)]
        cap, harm = per_step[0], per_step[-1]
        assert cap["cache"] == str(tmp_path / ".cache/huggingface/datasets")
        assert harm["cache"] == str(tmp_path / ".cache/huggingface-seam-datasets-3.3.1")
        for record in per_step:
            assert record["cuda_cache"] == str(tmp_path / "job/cuda-cache-123")
            assert Path(record["cuda_cache"]).is_dir()
        assert str(model / f"checkpoint-{step}") in harm["args"]
        model_args = cap["args"][cap["args"].index("--model_args") + 1]
        assert f"pretrained=../model/checkpoint-{step}," in model_args
        assert (output / f"checkpoint-{step}/harmfulness.json").is_file()
        if include_gsm8k:
            gsm = per_step[1]["args"]
            assert gsm[gsm.index("--tasks") + 1] == "gsm8k"
            assert gsm[gsm.index("--num_fewshot") + 1] == "5"
            assert gsm[gsm.index("--max_batch_size") + 1] == "32"
            assert (
                output / f"checkpoint-{step}/gsm8k-5shot/results_2026.json"
            ).is_file()
    if last_step < max_step:
        assert not (output / "final-evaluation.json").exists()
        return
    if include_gsm8k:
        assert (output / "gsm8k-5shot/results_2026.json").is_file()
    assert (output / "capabilities/results_2026.json").is_file()
    report = json.loads((output / "harmfulness.json").read_text())
    assert report["model"] == str(model)
    assert report["evaluated_model"] == str(model / f"checkpoint-{max_step}")
    assert report["reused_from"] == f"checkpoint-{max_step}/harmfulness.json"

    if include_gsm8k and first_step == 50:
        # A retry preserves complete stages and repairs only corrupt/missing JSON.
        keep = output / "checkpoint-50/capabilities/results_2026.json"
        before = keep.stat().st_mtime_ns
        (output / "checkpoint-100/capabilities/results_2026.json").write_text("{")
        (output / "checkpoint-150/gsm8k-5shot/results_2026.json").unlink()
        (output / "checkpoint-200/harmfulness.json").write_text("{")
        calls.unlink()
        retried = subprocess.run(
            ["bash", str(SCRIPT), "--checkpoints", str(model), str(output)],
            cwd=workdir,
            env={
                **os.environ,
                "SCRATCH": str(tmp_path),
                "SLURM_TMPDIR": str(tmp_path / "job"),
                "SLURM_JOB_ID": "123",
                "SEAM_ENV": str(seam_python.parents[1]),
                "BASH_ENV": str(bash_env),
                "TEST_PYTHON": sys.executable,
                "TEST_LAUNCHER": str(launcher),
                "CALLS": str(calls),
                "INCLUDE_GSM8K": "1",
                "SAVE_RESULTS": "1",
                "RESUME_EVALUATION": "1",
                "CHECKPOINT_MAX_STEP": str(max_step),
                "EVAL_LOCAL_LOG": "1",
            },
            capture_output=True,
            text=True,
        )
        assert retried.returncode == 0, retried.stderr
        archived_log = (
            workdir / "artifacts/tamia/logs/eval_harmfulness_and_capabilities-123.log"
        )
        assert "Reusing completed" in archived_log.read_text()

        records = [json.loads(line) for line in calls.read_text().splitlines()]
        assert [r["stage"] for r in records] == [
            "capabilities",
            "capabilities",
            "harmfulness",
        ]
        assert keep.stat().st_mtime_ns == before
        assert (
            output / "checkpoint-100/capabilities/results_2026.json.incomplete"
        ).exists()
        assert (
            json.loads((output / "final-evaluation.json").read_text())["reused_from"]
            == f"checkpoint-{max_step}"
        )


def test_node_local_logging_preserves_failure_and_copies_diagnostics(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "SCRATCH": str(tmp_path),
            "SLURM_TMPDIR": str(local),
            "SLURM_JOB_ID": "test",
            "EVAL_LOCAL_LOG": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    archived = (
        tmp_path / "artifacts/tamia/logs/eval_harmfulness_and_capabilities-test.log"
    )
    assert "Usage:" in archived.read_text()
    assert archived.read_text() == (local / archived.name).read_text()
