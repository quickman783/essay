"""Optional storage-only hook: archive early checkpoints and retain symlink paths."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from mmengine.hooks import Hook
from mmrotate.registry import HOOKS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def archive_checkpoint(source, destination):
    """Copy, verify, then atomically replace this run's source with a symlink."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        if source.resolve(strict=True) != destination.resolve(strict=True):
            raise RuntimeError('Unexpected existing checkpoint symlink')
        return dict(source=str(source), archive=str(destination), bytes=destination.stat().st_size,
                    sha256=sha256(destination))
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    size = source.stat().st_size
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.stat().st_size != size or sha256(destination) != digest:
            raise RuntimeError('Archive destination already exists with different contents')
    else:
        if shutil.disk_usage(destination.parent).free < size + 128 * 1024**2:
            raise RuntimeError('Not enough checkpoint archive space')
        temporary = destination.with_name(destination.name + '.partial')
        if temporary.exists():
            raise RuntimeError('Archive partial already exists; inspect before retrying')
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != size or sha256(temporary) != digest:
            raise RuntimeError('Checkpoint archive verification failed')
        os.replace(temporary, destination)
    link = source.with_name(source.name + '.archive-link')
    if link.exists() or link.is_symlink():
        raise RuntimeError('Temporary checkpoint link already exists')
    link.symlink_to(destination.resolve())
    os.replace(link, source)
    return dict(source=str(source), archive=str(destination), bytes=size, sha256=digest)


@HOOKS.register_module()
class ArchiveEarlyCheckpointsHook(Hook):
    """Run after CheckpointHook; does not touch model/optimizer/scheduler state."""
    priority = 'LOWEST'

    def __init__(self, archive_root, epochs=(1, 2, 3, 4)):
        self.archive_root = Path(archive_root)
        if not self.archive_root.is_absolute():
            raise ValueError('Checkpoint archive root must be absolute')
        self.epochs = tuple(int(epoch) for epoch in epochs)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch not in self.epochs:
            return
        work = Path(runner.work_dir).resolve()
        if not work.parent.name.startswith('run_') or not work.name.startswith('cycle_'):
            raise RuntimeError('Unexpected checkpoint work directory')
        filename = runner.cfg.default_hooks.checkpoint.filename_tmpl.format(epoch)
        if Path(filename).name != filename:
            raise RuntimeError('Checkpoint filename must be a plain basename')
        destination = self.archive_root / work.parent.name / work.name / filename
        if not destination.resolve().is_relative_to(self.archive_root.resolve()):
            raise RuntimeError('Archive destination escapes target root')
        record = archive_checkpoint(work / filename, destination)
        manifest = work / 'checkpoint_archive_manifest.json'
        records = json.loads(manifest.read_text()) if manifest.exists() else []
        records = [x for x in records if x['source'] != record['source']] + [record]
        manifest.write_text(json.dumps(records, indent=2))
        runner.logger.info('Archived checkpoint with SHA256 verification: %s -> %s',
                           record['source'], record['archive'])
