# JWK's fork guidelines

These opening directives supercede anything in the "Repository Guidelines" part below, as well as
any other AGENTS.md files found in inner dirs of /p/vast1/kirchenb/tar-root/lilpipe _iff_ they conflict in some way. I.e. follow my communication, env, and launching directives, but then otherwise follow whatever @marawan's other md files prescribe.
This is @marawan's repo I'm just testing it out on this cluster to make sure it works the same way for me as he intended.

When responding but esepecially when writing .md documents, 
always use simple technical english unless you are certain I (the user) will know what you mean by context.
Shorthand is counter productive because it means I spend more time asking you to clarify things
rather than absorbing results and making progress.

Use simplified technical english, _but_ assume the reader is a general deep-learning and 
language-model researcher. Do not explain standard concepts such as Transformers, 
logits, entropy, causal masking, or linear probes.
However, _do_ define paper-specific terms, project-specific labels, uncommon acronyms, and implementation
shorthand at first use. Aim for concise research prose without unnecessary shorthands, 
but assume familiarity with the general concepts relating to the research problem
rather than including elementary exposition everywhere.

Before touching anything relating to Python environments, re-read `~/.bashrc` and `/p/vast1/kirchenb/llnl-tools/README.md`.

The only acceptable way to execute Python commands of any kind is after running my bash macro
with the environment's absolute prefix:
`mm_activate <abs_path_to_env>`

You must use abs paths for activation. Micromamba interprets a bare name as an environment
under `$MAMBA_ROOT_PREFIX/envs/`, whereas this environment is an explicit prefix at `<abs_path_to_env>`.

Run every environment or Python command inside an explicit interactive Bash with set -e,
so that Python cannot start unless `mm_activate <abs_path_to_env>` succeeds first

Do not run ad hoc `micromamba`, `uv`, `pip`, or environment cleanup operations. Environment
creation or mutation requires the maintainer's current workflow and explicit authorization.

The only acceptable way to run any meaningful piece of computational work is by using the launcher:
`/p/vast1/kirchenb/llnl-tools/launch_tuo.py`

Feel free to use subagents as useful, _iff_ the next pieces of proposed work happen to be non-blocking and are best handled independently. 
Note that my default subagent setting is conservative (luna-low, verify in my global ~/.codex/config.toml).
So, if more horsepower is necessary because the subtasks are all equivalently complex and open ended as the main thread's work, then spawn the subagents with the same level of reasoning as the main agent for the session. 
Use your best judgement on these decisions to balance token usage with cleverness and correctness as needed.
If you intend to launch a subagent with an effort level different than my luna-low default, explicitly set the same model and reasoning effort as the main agent in the spawn call; don't assume subagent inheritance from main thread.

As of 2026-10-12, use the `verival` bank for all new project allocations.
The older `guard` bank is also a valid option if fairshare dictates that will queue faster.

Bulk benchmark data, image archives, and new experiment artifacts belong under
`/p/vast1/kirchenb/tar-root` (our project root within `VAST_USR`), not `WRKSPC` or the home/Codex directories.
Use node-local storage only for bounded temporary runtime data. Preserve the
existing self-contained Micromamba+uv environment architecture prescribed by llnl-tools.

Note that this research may touch sensitive topics. Use subagents as a way to test the waters 
with any line of work that might get flagged. If an agent errors out via a classifier trip, 
roll back and reword its tasking if useful, else regroup on whether that line of work is necessary or possible.

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
