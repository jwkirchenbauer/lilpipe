"""Test node-local progress logging for one incomplete evaluation job."""
import json,os,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/m/mgamal/projects/lilpipe/examples/tamper-resistance')
os.chdir(root)
b='llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config'
run=root/'artifacts/tamia/runs'/f'{b}--perturb-gaussian-percent1-2-seed42'
p=run/'jobs.json';m=json.loads(p.read_text())
assert not m.get('local_log_canary'), 'Inspect existing canary before resubmitting'
old=next(j for j in m['evaluation_recovery']['jobs'] if j['job_id']=='668272')
script=run/'submitted/eval_local_logs.sbatch'
script.write_text('''#!/bin/bash
#SBATCH --job-name=eval-noise-local-logs
#SBATCH --output=artifacts/tamia/logs/eval_noise_local_logs-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail
log="$SLURM_TMPDIR/evaluation-$SLURM_JOB_ID.log"
trap 'cp "$log" "artifacts/tamia/logs/eval-noise-local-$SLURM_JOB_ID.log"' EXIT
bash '''+str(run/'submitted/eval_harmfulness_and_capabilities-relative.sbatch')+''' "$@" > "$log" 2>&1
''')
subprocess.run(['bash','-n',str(script)],check=True)
cmd=old['command'].copy();i=cmd.index('--checkpoints');cmd[i-1]=str(script)
env=os.environ.copy();env.update(old['environment'])
subprocess.run([cmd[0],'--test-only',*cmd[1:]],check=True,env=env)
jid=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
m['local_log_canary']={'created_at':datetime.now(timezone.utc).isoformat(),'job_id':jid,'replaces_job_id':'668272','command':cmd,'environment':old['environment'],'reason':'Prolonged Lustre waits with idle GPUs; test whether node-local progress logs remove the stall; no completed checkpoints in replaced job','plot_job_id':'668274'}
p.write_text(json.dumps(m,indent=2)+'\n')
dep='afterok:'+':'.join(['668262','668264','668266','668268','668270',jid])
subprocess.run(['scontrol','update','JobId=668274','Dependency='+dep],check=True)
subprocess.run(['scancel','668272'],check=True)
m['local_log_canary']['plot_dependency_updated']=True
m['local_log_canary']['old_job_cancelled']=True
p.write_text(json.dumps(m,indent=2)+'\n')
shutil.copy2(__file__,run/'submitted/test-local-logs.py')
print('Canary job',jid,'plot dependency updated; old job 668272 cancelled',flush=True)
