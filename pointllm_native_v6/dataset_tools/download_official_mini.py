"""Atomic official mini download with bounded disk use and SHA provenance."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import urllib.request

BASE = Path(__file__).resolve().parent
URL = 'https://www.nuscenes.org/data/v1.0-mini.tgz'
EXPECTED_BYTES = 4167696325
DEST = BASE / 'v1.0-mini.tgz'
PART = DEST.with_suffix('.tgz.part')

def main():
    BASE.mkdir(exist_ok=True)
    if DEST.exists():
        if DEST.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError('Existing archive does not have the published observed size.')
        print('An archive of the expected size already exists; inspect its receipt before reusing.', flush=True)
        return
    free = shutil.disk_usage(BASE).free
    if free < EXPECTED_BYTES + 4_000_000_000:
        raise RuntimeError(f'Insufficient free disk for atomic archive plus selected extraction: {free}')
    started = time.time()
    digest = hashlib.sha256()
    total = 0
    reported = started
    request = urllib.request.Request(URL, headers={'User-Agent': 'RaHUM academic data reproducibility audit'})
    with urllib.request.urlopen(request, timeout=90) as response:
        response_headers = dict(response.headers)
        length = int(response.headers.get('Content-Length', '0'))
        if length != EXPECTED_BYTES:
            raise RuntimeError(f'Official response size changed: {length}')
        with PART.open('wb') as stream:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                total += len(block)
                now = time.time()
                if now - reported >= 15:
                    print(json.dumps({'download_bytes': total, 'expected_bytes': EXPECTED_BYTES, 'elapsed_s': round(now-started, 1), 'free_disk_bytes': shutil.disk_usage(BASE).free}), flush=True)
                    reported = now
            stream.flush()
            os.fsync(stream.fileno())
    if total != EXPECTED_BYTES:
        raise RuntimeError(f'Partial body retained under .part: got {total}')
    receipt = {
        'url': URL, 'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'bytes': total, 'sha256': digest.hexdigest(), 'http_headers': response_headers,
        'download_elapsed_s': time.time()-started,
        'source': 'Exact unauthenticated direct archive URL published in the official nuScenes tutorial.',
        'license': 'CC BY-NC-SA 4.0 plus current official dataset terms; noncommercial academic research use, no sensor-file redistribution in our code release.',
        'scope': 'Official v1.0-mini only, not the full trainval or test benchmark.',
        'integrity_scope': 'Observed exact Content-Length and locally computed SHA-256. No independent full-file official SHA-256 was supplied by this response.'
    }
    PART.replace(DEST)
    (BASE/'download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'completed': str(DEST), 'bytes': total, 'sha256': digest.hexdigest(), 'elapsed_s': round(time.time()-started, 1)}), flush=True)

if __name__ == '__main__':
    main()
