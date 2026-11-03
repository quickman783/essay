"""Resumable parallel upload of verified local DroneVehicle data using existing SSH authentication."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
import threading
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'datasets/prepared/dronevehicle'
STAGING = ROOT / 'experiment_support/upload_chunks'
REMOTE = '/root/autodl-tmp/dronevehicle_upload'
DESTINATION = '/root/autodl-tmp/Multimodal_Vehicle_Detection/datasets/prepared/dronevehicle'
PYTHON = '/root/miniconda3/bin/python'
LOCK = threading.Lock()
STATE = dict(status='indexing', transferred_bytes=0, total_bytes=0, completed_chunks=0, total_chunks=0)
ACTIVE = {}
COMPLETE_BYTES = 0


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('connect.nmb2.seetacloud.com', port=13549, username='root',
                   key_filename=str(Path.home() / '.ssh/object_detection_cloud'),
                   look_for_keys=False, allow_agent=False, timeout=30)
    client.get_transport().set_keepalive(20)
    return client


def command(client, text):
    _, stdout, stderr = client.exec_command(text)
    output, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error or output)
    return output


def status_writer(stop):
    previous_bytes, previous_time = 0, time.monotonic()
    while not stop.is_set():
        client = None
        try:
            client = connect()
            sftp = client.open_sftp()
            while True:
                with LOCK:
                    state = dict(STATE)
                    state['transferred_bytes'] = COMPLETE_BYTES + sum(ACTIVE.values())
                now = time.monotonic()
                state['speed_bytes_s'] = max(0, state['transferred_bytes'] - previous_bytes) / max(now - previous_time, .001)
                state['updated_at'] = datetime.now(timezone.utc).isoformat()
                previous_bytes, previous_time = state['transferred_bytes'], now
                content = json.dumps(state, ensure_ascii=False, indent=2)
                (ROOT / 'results/dronevehicle_upload_progress.json').write_text(content, encoding='utf-8')
                with sftp.file(REMOTE + '/progress.tmp', 'w') as stream:
                    stream.write(content)
                sftp.posix_rename(REMOTE + '/progress.tmp', REMOTE + '/progress.json')
                if stop.wait(3):
                    # Send the final state on the next iteration before returning.
                    with LOCK:
                        state = dict(STATE, transferred_bytes=COMPLETE_BYTES + sum(ACTIVE.values()),
                                     speed_bytes_s=0, updated_at=datetime.now(timezone.utc).isoformat())
                    content = json.dumps(state, indent=2)
                    with sftp.file(REMOTE + '/progress.tmp', 'w') as stream:
                        stream.write(content)
                    sftp.posix_rename(REMOTE + '/progress.tmp', REMOTE + '/progress.json')
                    (ROOT / 'results/dronevehicle_upload_progress.json').write_text(content, encoding='utf-8')
                    return
        except Exception as error:
            print('Status connection retry:', str(error), flush=True)
            stop.wait(3)
        finally:
            if client:
                client.close()


def transfer(chunk):
    global COMPLETE_BYTES
    name, entries, payload = chunk
    archive_path = STAGING / (name + '.tar')
    with tarfile.open(archive_path, 'w', format=tarfile.USTAR_FORMAT) as archive:
        for entry in entries:
            archive.add(SOURCE / entry['path'], arcname=entry['path'], recursive=False)
    size = archive_path.stat().st_size
    digest = hashlib.sha256()
    with archive_path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    checksum = digest.hexdigest()
    error = None
    for attempt in range(5):
        client = None
        try:
            client = connect()
            sftp = client.open_sftp()
            marker = REMOTE + '/completed/' + name + '.json'
            try:
                with sftp.file(marker) as stream:
                    acknowledged = json.loads(stream.read())
                if acknowledged['sha256'] == checksum:
                    break
            except OSError:
                pass
            part = REMOTE + '/' + name + '.part'
            metadata = REMOTE + '/' + name + '.meta.json'
            offset = 0
            try:
                with sftp.file(metadata) as stream:
                    same = json.loads(stream.read()).get('sha256') == checksum
                if same:
                    offset = min(sftp.stat(part).st_size, size)
            except OSError:
                pass
            with sftp.file(metadata, 'w') as stream:
                stream.write(json.dumps(dict(sha256=checksum, bytes=size)))
            with archive_path.open('rb') as source, sftp.file(part, 'r+b' if offset else 'wb') as target:
                source.seek(offset)
                target.seek(offset)
                target.set_pipelined(True)
                while True:
                    block = source.read(256 * 1024)
                    if not block:
                        break
                    target.write(block)
                    offset += len(block)
                    with LOCK:
                        ACTIVE[name] = int(payload * offset / size)
            remote_archive = REMOTE + '/' + name + '.tar'
            sftp.posix_rename(part, remote_archive)
            args = [PYTHON, REMOTE + '/receive_dataset_chunk.py', remote_archive, DESTINATION, marker, checksum]
            command(client, ' '.join(shlex.quote(value) for value in args))
            break
        except Exception as exc:
            error = exc
            print(name, 'retry', attempt + 1, str(exc), flush=True)
            time.sleep(min(2 ** attempt, 15))
        finally:
            if client:
                client.close()
    else:
        raise RuntimeError(f'{name}: {error}')
    with LOCK:
        ACTIVE.pop(name, None)
        COMPLETE_BYTES += payload
        STATE['completed_chunks'] += 1
    archive_path.unlink()
    print('Verified:', name, flush=True)


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    controller = connect()
    command(controller, 'mkdir -p ' + REMOTE + '/completed ' + DESTINATION)
    sftp = controller.open_sftp()
    for filename in ('receive_dataset_chunk.py', 'watch_dronevehicle_upload.py'):
        sftp.put(str(ROOT / 'experiment_support' / filename), REMOTE + '/' + filename)
    stop = threading.Event()
    reporter = threading.Thread(target=status_writer, args=(stop,), daemon=True)
    reporter.start()
    try:
        chunks, manifest = [], []
        for split in ('train', 'val', 'test'):
            ids = sorted(path.stem for path in (SOURCE / split / 'image_vi').glob('*.png'))
            for start in range(0, len(ids), 200):
                entries = []
                for image_id in ids[start:start + 200]:
                    for folder, extension in [('image_vi', '.png'), ('image_ir', '.png'), ('annofiles_vi', '.txt'), ('annofiles_ir', '.txt')]:
                        relative = f'{split}/{folder}/{image_id}{extension}'
                        entries.append(dict(path=relative, bytes=(SOURCE / relative).stat().st_size))
                chunks.append((f'{split}_{start//200:04d}', entries, sum(entry['bytes'] for entry in entries)))
                manifest.extend(entries)
        with LOCK:
            STATE.update(status='uploading', total_bytes=sum(e['bytes'] for e in manifest), total_chunks=len(chunks))
        manifest_path = STAGING / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        sftp.put(str(manifest_path), REMOTE + '/manifest.json')
        print('Uploading', STATE['total_bytes'], 'bytes in', len(chunks), 'chunks', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(transfer, chunks):
                pass
        with LOCK:
            STATE['status'] = 'verifying all files'
        verification = (
            "import json;from pathlib import Path;"
            f"root=Path({DESTINATION!r});items=json.loads(Path({(REMOTE+'/manifest.json')!r}).read_text());"
            "bad=[x['path'] for x in items if not (root/x['path']).is_file() or (root/x['path']).stat().st_size!=x['bytes']];"
            "assert not bad,str(bad[:10]);print('VERIFIED_ALL_FILES',len(items))"
        )
        print(command(controller, PYTHON + ' -c ' + shlex.quote(verification)), flush=True)
        with LOCK:
            STATE['status'] = 'complete'
    except BaseException as exc:
        with LOCK:
            STATE.update(status='interrupted; rerun uploader to resume', error=str(exc))
        raise
    finally:
        stop.set()
        reporter.join(timeout=30)
        controller.close()


if __name__ == '__main__':
    main()
