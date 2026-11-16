"""Check the exact text files and original-dtype numeric round trips."""
from pathlib import Path
import hashlib
import json
import numpy as np

base = Path(__file__).resolve().parent
manifest = json.loads((base / 'provenance.json').read_text())
for record in manifest['records']:
    path = base / record['csv_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['csv_sha256'], path.name
    array = np.loadtxt(path, delimiter=',', skiprows=1, dtype=record['source_dtype'])
    assert list(array.shape) == record['shape'], path.name
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    assert digest == record['source_array_sha256_c_order'], path.name
print(f'All {manifest["csv_count"]} CSV files and restored numeric arrays passed.')
