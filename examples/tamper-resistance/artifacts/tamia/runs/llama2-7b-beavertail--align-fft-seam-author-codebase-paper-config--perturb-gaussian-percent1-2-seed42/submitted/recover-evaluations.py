"""Replace evaluations whose absolute model names exceeded NAME_MAX."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/m/mgamal/projects/lilpipe/examples/tamper-resistance')
os.chdir(root)
b='llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config'
run=root/'artifacts/tamia/runs'/f'{b}--perturb-gaussian-percent1-2-seed42'
p=run/'jobs.json'
m=json.loads(p.read_text())
assert not m.get('evaluation_recovery', {}).get('jobs'), 'Recovery has submissions; inspect IDs before retrying'
states=subprocess.check_output(['sacct','-j','668198,668206','-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
assert len([s for s in states.splitlines() if 'COMPLETED|0:0' in s])==2, states
old=[x for x in m['jobs'] if x['stage'].startswith('eval-')]
assert len(old)==6
snapshot=run/'submitted/eval_harmfulness_and_capabilities-relative.sbatch'
shutil.copy2(root/'scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch',snapshot)
shutil.copy2(__file__,run/'submitted/recover-evaluations.py')
m['evaluation_recovery']={'reason':'lm-eval caught OSError(36, File name too long) and silently skipped JSON saves for absolute model paths','created_at':datetime.now(timezone.utc).isoformat(),'superseded_job_ids':[x['job_id'] for x in old]+['668214'],'script_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'jobs':[]}
def save():p.write_text(json.dumps(m,indent=2)+'\n')
save()
ids=[]
for job in old:
 cmd=[arg for arg in job['command'] if not arg.startswith('--dependency=')]
 pos=cmd.index('--checkpoints')
 cmd[pos-1]=str(snapshot)
 for i in range(pos+1,len(cmd)):
  cmd[i]=os.path.relpath(cmd[i],root)
 for model in cmd[pos+1::2]:
  assert len((model+'/checkpoint-250').replace('/','__'))<255
 env=os.environ.copy();env.update(job['environment'])
 subprocess.run([cmd[0],'--test-only',*cmd[1:]],check=True,env=env)
 jid=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
 ids.append(jid)
 m['evaluation_recovery']['jobs'].append({'stage':job['stage'],'job_id':jid,'replaces_job_id':job['job_id'],'command':cmd,'environment':job['environment']})
 save();print(job['stage'],jid,flush=True)
plot=['sbatch','--parsable','--dependency=afterok:'+':'.join(ids),str(run/'submitted/plot.sbatch')]
subprocess.run([plot[0],'--test-only',*plot[1:]],check=True)
jid=subprocess.check_output(plot,text=True).strip().split(';')[0]
m['evaluation_recovery']['jobs'].append({'stage':'plot-both-comparisons','job_id':jid,'replaces_job_id':'668214','command':plot})
m['evaluation_recovery']['submission_complete']=True
save();print('plot',jid,flush=True)
