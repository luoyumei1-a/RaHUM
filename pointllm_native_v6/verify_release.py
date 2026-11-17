"""Verify v10 file identities and real input configuration without pretrained weights."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
subprocess.run([sys.executable,str(ROOT/'verify_real_example.py')],check=True,cwd=ROOT)
prov=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
for row in prov['reference_files']:
    assert sha(ROOT/row['package_path'])==row['sha256'],row['package_path']
r=json.loads((ROOT/'diagnostics/actual_k8_128/memory_sweep_receipt.json').read_text())
assert sha(ROOT/'backbone/rgb_runner_01/inference.pt')==r['checkpoint_sha256']
assert sha(ROOT/'backbone/rgb_runner_01_source_frozen/adapter.py')==r['adapter_source_sha256']
assert sha(ROOT/'diagnostics/sweep_native_memory.py')==r['diagnostic_script_sha256']
assert sha(ROOT/'diagnostics/actual_k8_128/local_memory_sweep.csv')==r['csv_sha256']
assert [x['local_K'] for x in r['rows']]==[8,16,24,32,48,64,96,128]
assert all(x['adapter_parameters']==3296719 for x in r['rows'])
assert r['checkpoint_completed_steps']==2
assert r['observed_reference_tokens']==256
e=json.loads((ROOT/'matlab/ONLINE_EXPORT_RECEIPT.json').read_text())
assert sha(ROOT/'matlab/RaHUM_Figures_v10.m')==e['source_sha256']
for row in e['PDFs']:
    assert sha(ROOT/'matlab/online_outputs'/row['file'])==row['sha256']
for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
assert len(list((ROOT/'matlab').glob('*.m')))==1
print(json.dumps({'status':'PASS','frozen_reference_files':len(prov['reference_files']),'native_K_parameters':3296719,'K':[8,16,24,32,48,64,96,128],'MATLAB_export_PDFs':e['PDF_count'],'source_hashes_preserved':True},indent=2))
