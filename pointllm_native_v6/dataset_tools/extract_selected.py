"""Validate the full mini gzip/tar and retain metadata, all radar, front RGB."""
from pathlib import Path, PurePosixPath
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time

BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / 'v1.0-mini.tgz'
DEST = BASE / 'data'

def selected(parts):
    if len(parts) >= 2 and parts[0] == 'v1.0-mini' and parts[-1].endswith('.json'):
        return 'metadata'
    if len(parts) >= 3 and parts[0] in ('samples', 'sweeps'):
        if parts[1].startswith('RADAR_'):
            return 'radar'
        if parts[1] == 'CAM_FRONT':
            return 'front_rgb'
    if len(parts) == 1 and any(t in parts[0].lower() for t in ('license', 'readme', 'notice')):
        return 'license_or_readme'
    return None

def main():
    receipt = json.loads((BASE / 'download_receipt.json').read_text())
    if ARCHIVE.stat().st_size != receipt['bytes']:
        raise RuntimeError('Archive size differs from verified download.')
    if shutil.disk_usage(BASE).free < 3_000_000_000:
        raise RuntimeError('Less than 3 GB free for selected extraction.')
    DEST.mkdir(exist_ok=True)
    totals = {}
    full_files = full_bytes = kept_files = kept_bytes = 0
    started = reported = time.time()
    with (BASE / 'archive_members.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['archive_member', 'bytes', 'kept_kind', 'local_path', 'sha256'])
        writer.writeheader()
        with gzip.open(ARCHIVE, 'rb') as compressed:
            with tarfile.open(fileobj=compressed, mode='r|') as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    path = PurePosixPath(member.name)
                    parts = tuple(p for p in path.parts if p != '.')
                    if path.is_absolute() or '..' in parts:
                        raise RuntimeError('Unsafe archive member: ' + member.name)
                    kind = selected(parts)
                    digest = ''
                    local = ''
                    if kind:
                        dest = DEST.joinpath(*parts)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        partial = dest.with_name(dest.name + '.part')
                        h = hashlib.sha256()
                        count = 0
                        with tar.extractfile(member) as source, partial.open('wb') as target:
                            while True:
                                block = source.read(1024*1024)
                                if not block:
                                    break
                                target.write(block)
                                h.update(block)
                                count += len(block)
                        if count != member.size:
                            raise RuntimeError('Extracted member size mismatch: ' + member.name)
                        partial.replace(dest)
                        digest = h.hexdigest()
                        local = dest.relative_to(BASE).as_posix()
                        kept_files += 1
                        kept_bytes += member.size
                        bucket = totals.setdefault(kind, {'files': 0, 'bytes': 0})
                        bucket['files'] += 1
                        bucket['bytes'] += member.size
                    writer.writerow({'archive_member': member.name, 'bytes': member.size, 'kept_kind': kind or '', 'local_path': local, 'sha256': digest})
                    full_files += 1
                    full_bytes += member.size
                    now = time.time()
                    if now - reported >= 15:
                        print(json.dumps({'scanned_files': full_files, 'uncompressed_bytes_scanned': full_bytes, 'retained_files': kept_files, 'retained_bytes': kept_bytes, 'elapsed_s': round(now-started, 1), 'disk_free': shutil.disk_usage(BASE).free}), flush=True)
                        reported = now
            # Consume the remaining compressed stream to force gzip CRC/size checks.
            while compressed.read(1024*1024):
                pass
    result = {
        'archive': ARCHIVE.name, 'archive_bytes': receipt['bytes'], 'archive_sha256': receipt['sha256'],
        'gzip_crc_and_stream_complete': True, 'tar_complete_iteration': True,
        'full_archive_regular_files': full_files, 'full_archive_uncompressed_file_bytes': full_bytes,
        'retained_files': kept_files, 'retained_bytes': kept_bytes, 'retained_by_kind': totals,
        'extraction_elapsed_s': time.time()-started,
        'scope': 'Entire official mini archive validated; only metadata, all five radar channels and CAM_FRONT data retained. Other camera and LiDAR bodies were not retained.',
        'sensor_data_redistribution': 'Not included in the public project release.'
    }
    (BASE / 'extraction_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
