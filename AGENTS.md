# Repository Guidelines

## Project Structure

`lilpipe` is a small declarative Slurm pipeline compiler for ML experiments (Python 3.12+, `src` layout):

- `src/lilpipe/core.py` — loading, validation, `select()`/`plan()` (topological order, cycle checks), and `submit()`.
- `src/lilpipe/results.py` — `lilpipe results`: collects lm-eval artifacts into markdown/csv/json tables.
- `src/lilpipe/cli.py` — argparse CLI with `run` and `results` subcommands; a bare `lilpipe <config>.yml` is accepted as an implicit `run`.
- `tests/` — pytest suite: library tests (`test_pipeline.py`, `test_results.py`) plus one regression file per example that loads the example's YAML configs and scripts by path.
- `examples/weight-steering/` and `examples/tamper-resistance/` are self-contained uv projects (own `pyproject.toml`, `uv.lock`, `.venv*`) that install the root package editable. The other `examples/*` directories are docs only.
- `scripts/job.sh` / `safe_job.sh` are cluv job scripts; `logs/` is a cluv symlink to `$SCRATCH` (cluster-only, untracked).

## Paths and Working Directory

Every relative path — pipeline, registries, sbatch scripts, results rows — resolves from the **current working directory**, never from the config file. Run `lilpipe` from the example root, e.g.:

```bash
(cd examples/weight-steering && lilpipe configs/experiments/smollm3/main.yml --dry-run)
(cd examples/tamper-resistance && lilpipe configs/experiments/z7b-lora-s125.yml --dry-run)
```

Pipeline manifests live at `configs/experiments/<name>.yml` inside each example (there is no top-level `pipeline.yml`).

## Commands

- Dev install (root): `python -m venv .venv && source .venv/bin/activate && python -m pip install -e '.[test]'` (add `,dev` for black). Do **not** `pip install lilpipe` from PyPI — that is an unrelated package; install from a git tag.
- `python -m pytest` from the repo root runs the suite. Focused: `python -m pytest tests/test_pipeline.py -k cycle`.
  - Gotcha: `tests/test_conic_distance_across_seeds.py` imports `numpy`, which is **not** in the root `test` extra, so full-suite collection fails in the root venv. Use `--ignore=tests/test_conic_distance_across_seeds.py` or a numpy-capable environment. Torch/transformers tests self-skip via `importorskip` when the cluster env is absent.
- Formatting: `python -m black src tests examples` (line-length 88, py312). CI fails on unformatted code.
- CI order (`.github/workflows/ci.yml`): `black --check` → `pytest` → `python -m build` → clean-wheel install + `lilpipe --help` + example dry-run. Note: the CI dry-run still points at the stale `examples/weight-steering/configs/experiments/pipeline.yml`; the live manifest is `smollm3/main.yml`.
- Cluster work uses cluv, configured under `[tool.cluv]` in the root `pyproject.toml` (results symlink `logs` → `$SCRATCH/logs/lilpipe`, offline compute nodes by default).
- Known issue (Mila cluster): when `$SCRATCH` runs out of disk space, compute nodes fail to provision and any job placed on them dies ~1 s after start with `Reason=ReqNodeNotAvail`, exit code `0:53` (`RaisedSignal:53`), and no `.out` log is written. This looks like a transient node/pool failure but is actually the user's scratch quota; free scratch space before resubmitting.

## Domain Conventions

- Immutable data model: `@dataclass(frozen=True)`, tuples, read-only mappings. `select()` and `plan()` return new objects; `load()` captures the CWD as project root.
- YAML is schema-versioned (`version: 1`). `load()` validates schema and selections; `plan()` validates producer references, duplicate stage IDs, unknown dependencies, and cycles.
- Argument templates support only `{model.<field>}` placeholders; a missing field is an error (no defaults, conditionals, or execution).
- Producer `depends_on` entries are model slugs resolved to producer stage IDs; the transitive dependency graph is retained even for models not selected for evaluation.
- `--existing-models` skips producer jobs by trusting the artifacts exist; lilpipe never inspects model paths.

## Style & Commits

Four-space indent, PEP 8 naming, modern type annotations, concise docstrings, grouped imports. Conventional Commit subjects without scopes: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`, `revert:`. Every behavior change needs a regression test; include dry-run output in PRs when submission behavior changes.

## Forward-looking development

Treat this repository as an engineering project: additions should fit its existing
structure and remain useful beyond the immediate experiment.

- Prefer extending existing entry points and configuration over adding scripts.
  Express changes in models, datasets, learning rates, and training lengths as
  configuration when the existing workflow supports them.
- Keep reusable training, evaluation, and analysis code general enough to serve
  related experiments. Avoid embedding a particular run's paths or hyperparameters
  in permanent execution scripts.
- Keep the number of files and public interfaces small. Before adding a file,
  check whether an existing module, launcher, or config family is the right home.
- Put temporary or experiment-specific scripts under `scripts/misc/` within the
  relevant project/example, rather than among permanent workflow entry points.
- Keep configuration sources authoritative; generate repeated run definitions
  instead of maintaining the same settings in multiple places. Preserve exact
  submitted settings and source snapshots as provenance.
- Generalize around demonstrated needs. Do not introduce speculative frameworks
  or abstractions merely to make a small change appear reusable.
