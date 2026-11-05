"""Keep the pending final plot job consistent with the requested palette."""
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path('/home/m/mgamal/projects/lilpipe/examples/tamper-resistance')
os.chdir(root)
base='llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config'
run=root/'artifacts/tamia/runs'/f'{base}--perturb-gaussian-percent1-2-seed42'
record=run/'jobs.json';m=json.loads(record.read_text())
assert not m.get('colorblind_plot_update'), 'Already submitted; inspect metadata'
source=run/'submitted/plot_trajectory_beavertails-colorblind.py'
shutil.copy2(root/'scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py',source)
script=run/'submitted/plot-colorblind.sbatch'
script.write_text((run/'submitted/plot.sbatch').read_text().replace('plot_trajectory_beavertails.py',source.name))
subprocess.run(['bash','-n',str(script)],check=True)
cmd=['sbatch','--parsable','--dependency=afterok:668262:668264:668268:668270:668311',str(script)]
subprocess.run([cmd[0],'--test-only',*cmd[1:]],check=True)
jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
m['colorblind_plot_update']={'created_at':datetime.now(timezone.utc).isoformat(),'job_id':jid,'replaces_job_id':'668274','command':cmd,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'palette':'seaborn colorblind; Base black','completed_only_previews_generated':True}
m['current_plot_job_id']=jid
record.write_text(json.dumps(m,indent=2)+'\n')
subprocess.run(['scancel','668274'],check=True)
shutil.copy2(__file__,run/'submitted/update-colorblind-plot.py')
print('Final colorblind plot job:',jid,flush=True)
