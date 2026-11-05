"""Submit the SEAM 1%/2% noise comparison using existing experiment launchers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

root = Path('/home/m/mgamal/projects/lilpipe/examples/tamper-resistance')
os.chdir(root)
base = 'llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config'
artifacts = root / 'artifacts/tamia'
run = artifacts / 'runs' / f'{base}--perturb-gaussian-percent1-2-seed42'
assert not run.exists(), f'Run already exists: {run}'
assert (artifacts / 'models' / base / 'config.json').is_file()
for fraction in ('0.01', '0.02'):
    assert not (artifacts / 'models' / f'{base}--perturb-gaussian-l2r{fraction}-seed42').exists()
assert shutil.disk_usage(artifacts).free > 850 * 1024**3
submitted = run / 'submitted'
submitted.mkdir(parents=True)
(artifacts / 'logs').mkdir(exist_ok=True)
files = [
    'scripts/slurm/tamia/analysis_perturb_model.sbatch',
    'scripts/slurm/tamia/eval_capability_models.sbatch',
    'scripts/slurm/tamia/train_seam_relearning_sweep.sbatch',
    'scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch',
    'scripts/slurm/tamia/eval_capabilities.sbatch',
    'scripts/analysis/parameter-noise/perturb_model.py',
    'scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py',
    '../seam/scripts/train.py', '../seam/scripts/eval_harmfulness.py',
    '../seam/config/eval.yaml',
]
hashes = {}
for relative in files:
    source = root / relative
    shutil.copy2(source, submitted / source.name)
    hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
shutil.copy2(__file__, submitted / 'submit.py')
metadata = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'source_model': str(artifacts / 'models' / base),
    'percentages': [1, 2], 'fractions': [0.01, 0.02],
    'noise_seed': 42, 'training_seed': 42,
    'normalization': 'One shared global Gaussian direction, normalized relative to the SEAM-defended model parameter L2 norm; BF16 realized delta recorded separately',
    'learning_rates': ['2e-5','5e-5','8e-5','1e-4','2e-4'],
    'steps': 250, 'checkpoint_steps': [50,100,150,200,250],
    'batch_size_per_gpu': 4, 'gradient_accumulation_steps': 1,
    'samples': 1000, 'save_only_model': True,
    'hashes': hashes, 'jobs': [],
}
record = run / 'jobs.json'

def save():
    record.write_text(json.dumps(metadata, indent=2) + '\n')

def submit(stage, script, args, dependencies=(), environment=None, time=None):
    command = ['sbatch', '--parsable']
    if time:
        command += [f'--time={time}']
    if dependencies:
        command += ['--dependency=afterok:' + ':'.join(dependencies)]
    command += [str(script), *map(str,args)]
    env = os.environ.copy()
    env.update({'INCLUDE_GSM8K':'1','INCLUDE_HARMFULNESS':'1','HARMFULNESS_ONLY':'0','CAPABILITY_TASKS':'mmlu,truthfulqa_mc1,arc_easy,hellaswag','NUM_FEWSHOT':'0','MAX_BATCH_SIZE':'128'})
    env.update(environment or {})
    subprocess.run([*command[:1], '--test-only', *command[1:]], check=True, env=env)
    job = subprocess.check_output(command, text=True, env=env).strip().split(';')[0]
    metadata['jobs'].append({'stage':stage,'job_id':job,'command':command,'environment':{k:env[k] for k in ('MODEL_SLUG','INCLUDE_GSM8K','INCLUDE_HARMFULNESS','HARMFULNESS_ONLY','CAPABILITY_TASKS','NUM_FEWSHOT','MAX_BATCH_SIZE') if k in env}})
    save()
    print(stage, job, flush=True)
    return job

save()
generation = submit('noise-generation', submitted / 'analysis_perturb_model.sbatch', [
    '--model', artifacts / 'models' / base,
    '--output-prefix', artifacts / 'models' / f'{base}--perturb-gaussian',
    '--output', run / 'perturbations.json', '--percentages','1','2','--seed','42'])
noised = [f'{base}--perturb-gaussian-l2r{f}-seed42' for f in ('0.01','0.02')]
baseline = submit('baseline-evaluations', submitted / 'eval_capability_models.sbatch',
    [artifacts/'evals', *[artifacts/'models'/b for b in noised]], [generation], time='00:45:00')
evaluation_jobs = []
for b in noised:
    attack_run = artifacts / 'runs' / f'{b}--relearning-author-codebase-sweep-ms250'
    attack_run.mkdir()
    train = submit(f'train-{b}', submitted/'train_seam_relearning_sweep.sbatch',
        [artifacts/'models'/b,artifacts/'models',attack_run,'true'],[baseline],{'MODEL_SLUG':b},time='01:00:00')
    groups = [('2e-5','5e-5'),('8e-5','1e-4'),('2e-4',)]
    for group in groups:
        args = ['--checkpoints']
        for lr in group:
            slug = f'{b}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250'
            args += [artifacts/'models'/slug,artifacts/'evals'/slug]
        evaluation_jobs.append(submit(f'eval-{b}-{"-".join(group)}',submitted/'eval_harmfulness_and_capabilities.sbatch',args,[train]))
    (attack_run/'jobs.json').write_text(json.dumps({'parent_run':str(run),'source_model':str(artifacts/'models'/b),'training_job_id':train,'learning_rates':metadata['learning_rates'],'checkpoint_steps':metadata['checkpoint_steps']},indent=2)+'\n')
plot = submitted/'plot.sbatch'
plot.write_text('''#!/bin/bash
#SBATCH --job-name=plot-seam-noise-1-2
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tamia/logs/plot_seam_noise_1_2-%j.out
set -euo pipefail
export MPLBACKEND=Agg
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
'''+f'/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python {submitted / "plot_trajectory_beavertails.py"}\n'+f'/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python {submitted / "plot_trajectory_beavertails.py"} --lr-markers\n')
submit('plot-both-comparisons',plot,[],evaluation_jobs)
metadata['submission_complete'] = True
save()
print('Submission complete:',record,flush=True)
