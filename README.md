# lilpipe

`lilpipe` is a small declarative Slurm pipeline for ML experiments. It reads an
experiment specification, builds the dependency graph, and submits each job
with the right Slurm `afterok` dependencies.

A lilpipe experiment typically uses three configuration files:

1. Model registry (`configs/registries/models.yml`). Lists the available models
   and the Slurm jobs that produce them.
2. Evaluation registry (`configs/registries/evaluations.yml`). Defines the
   evaluation jobs and their arguments.
3. Experiment specification (`configs/experiments/pipeline.yml`). Selects the
   models to produce and the evaluations to run for one experiment.

## Install

> [!WARNING]
> The package named `lilpipe` on PyPI is unrelated. Do not run
> `pip install lilpipe`; install this project from a Git tag.

Python 3.12 or newer is required. Install the current release directly from
GitHub:

```sh
python -m pip install \
  'lilpipe @ git+https://github.com/marawangamal/lilpipe.git@v0.3.0'
```

For development:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Example walkthrough

This walkthrough uses the [minimal sample project](examples/minimal). The sample
project defines two models in
[`models.yml`](examples/minimal/configs/registries/models.yml) and how each one
is created. For example, `toy-tuned` is created by running
`scripts/produce.sbatch` with the positional arguments
`artifacts/models/toy-tuned` and `tuned`. It also depends on `toy-base`.
Therefore, running
[`pipeline.yml`](examples/minimal/configs/experiments/pipeline.yml) produces both
`toy-base` and `toy-tuned`, even though only `toy-tuned` is specified in
`pipeline.yml`.

From `examples/minimal`, run the following command to see the Slurm DAG that
would be submitted:

```sh
lilpipe configs/experiments/pipeline.yml --dry-run
```

## Essential commands

Preview or submit the selections in a manifest:

```sh
lilpipe configs/experiments/pipeline.yml --dry-run
lilpipe configs/experiments/pipeline.yml
```

Replace the manifest's selections from the command line:

```sh
lilpipe configs/experiments/pipeline.yml \
  --models toy-tuned --evaluations toy-score
```

Skip a producer by model slug (or any stage by its stage ID):

```sh
lilpipe configs/experiments/pipeline.yml --skip toy-base
```

Trust and reuse existing artifacts without submitting their producer jobs:

```sh
lilpipe configs/experiments/pipeline.yml \
  --existing-models toy-base --existing-models toy-tuned
```

`lilpipe` does not inspect declared model paths. With `--existing-models`, you
are asserting that the artifacts needed downstream already exist.

Pass additional options to every `sbatch` invocation:

```sh
lilpipe configs/experiments/pipeline.yml \
  --sbatch-args='--account=my-account --time=00:10:00'
```

Archive submission commands, Slurm job IDs, referenced source/config snapshots,
and the Git version/working-tree diff in a fresh directory:

```sh
lilpipe configs/experiments/pipeline.yml --record-dir artifacts/runs/my-run
```

Completed submissions remain recorded if a later submission fails. An existing
record directory is rejected; `--dry-run` creates no record files.

Render collected results as Markdown (the default), CSV, or JSON:

```sh
lilpipe results configs/results/main.yml
lilpipe results configs/results/main.yml --format csv
lilpipe results configs/results/main.yml --format json
```

## Guides and examples

- [Getting started](docs/getting-started.md): install, dry-run, submit, inspect
  logs, and collect results.
- [Pipeline configuration](docs/pipelines.md): registries, dependencies,
  interpolation, selections, reuse, and Slurm options.
- [Results tables](docs/results.md): lm-eval discovery, formatting, grouping,
  missing values, and best-value highlighting.
- [Minimal example](examples/minimal): the canonical tutorial project.
- [Weight steering](examples/weight-steering) and
  [tamper resistance](examples/tamper-resistance): maintained research examples.

## Example project

Use `examples/minimal` as a template rather than copying incomplete YAML
fragments. Its project layout is:

```text
examples/minimal/
├── configs/
│   ├── experiments/pipeline.yml   # selects models and evaluations
│   ├── registries/models.yml      # defines artifacts and producer jobs
│   ├── registries/evaluations.yml # defines evaluation jobs
│   └── results/main.yml           # defines the results table
└── scripts/
    ├── produce.sbatch
    └── evaluate.sbatch
```

Producer `depends_on` entries name model slugs. Evaluation arguments may use
scalar `{model.<field>}` placeholders; missing fields are errors, and there are
no defaults, conditionals, or template execution. See the
[pipeline guide](docs/pipelines.md) for the annotated configuration.
