"""Prepare the local VEDAI and DroneVehicle archives for DMM, without modifying sources.

Run with an interpreter containing numpy, OpenCV, Pillow and tqdm:
    python code/prepare_local_datasets.py
Re-running preserves prepared images; use --audit-only to validate existing outputs.
"""
import argparse
import csv
import importlib.util
import json
import hashlib
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'code/DMM-main/tools'
SPLITS = ('train', 'val', 'test')

# 训练 / 验证 / 测试 的划分比例
SPLIT_RATIOS = [0.7, 0.1, 0.2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


images = load_module('prepare_images', TOOLS / 'data_process.py')
labels = load_module('prepare_labels', TOOLS / 'VOC2DOTA.py')
splitter = load_module('split_vedai', ROOT / 'code/split_vedai_dataset.py')


def prepare_dronevehicle(workers):
    dst = ROOT / 'datasets/prepared/dronevehicle'
    for split in SPLITS:
        src = ROOT / 'datasets' / split / split
        vi = src / (split + 'img')
        ids = sorted(p.stem for p in vi.glob('*.jpg'))
        expected = set(ids)
        for suffix, ext in [('imgr', '.jpg'), ('label', '.xml'), ('labelr', '.xml')]:
            actual = {p.stem for p in (src / (split + suffix)).glob('*' + ext)}
            if actual != expected:
                raise ValueError(f'{split}/{suffix}: unmatched IDs {expected ^ actual}')
        for folder in ('image_vi', 'image_ir', 'annofiles_vi', 'annofiles_ir'):
            (dst / split / folder).mkdir(parents=True, exist_ok=True)

        def process(image_id):
            for suffix, folder, gray in [('img', 'image_vi', False), ('imgr', 'image_ir', True)]:
                target = dst / split / folder / (image_id + '.png')
                if not target.exists():
                    images.convert(str(src / (split + suffix) / (image_id + '.jpg')),
                                   str(target), images.DRONEVEHICLE_CROP, gray)
            for suffix, folder in [('label', 'annofiles_vi'), ('labelr', 'annofiles_ir')]:
                target = dst / split / folder / (image_id + '.txt')
                converted = labels.lines_from_xml(str(src / (split + suffix) / (image_id + '.xml')),
                                                 'dronevehicle', -100.0)
                labels.write_lines(str(target), converted)
            return image_id

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count, _ in enumerate(pool.map(process, ids), 1):
                if count % 1000 == 0 or count == len(ids):
                    print(f'DroneVehicle {split}: {count}/{len(ids)}', flush=True)


def prepare_vedai():
    raw = ROOT / 'datasets/Annotations512/Annotations512'
    source = ROOT / 'datasets/Vehicules512/Vehicules512'
    dst = ROOT / 'datasets/prepared/vedai'
    color, infrared = splitter.index_images(str(source))
    annotation_lines, counts = {}, {}
    for path in sorted(raw.glob('*.txt')):
        if not path.stem.isdigit():
            continue
        lines = labels.lines_from_vedai_raw(str(path), 0.0)
        if lines:
            if path.stem not in color or path.stem not in infrared:
                raise ValueError('Missing VEDAI image pair: ' + path.stem)
            annotation_lines[path.stem] = lines
            counts[path.stem] = Counter(line.split()[8] for line in lines)
    assignment = splitter.stratified_split(sorted(counts), counts, SPLIT_RATIOS, 42)
    splitter.ensure_class_coverage(assignment, counts)
    for split in SPLITS:
        for folder in ('image_vi', 'image_ir', 'annofiles_vi', 'annofiles_ir'):
            (dst / split / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, split in sorted(assignment.items()):
        for lookup, folder in [(color, 'image_vi'), (infrared, 'image_ir')]:
            target = dst / split / folder / (image_id + '.png')
            if not target.exists():
                images.convert(str(source / lookup[image_id]), str(target))
        for folder in ('annofiles_vi', 'annofiles_ir'):
            labels.write_lines(str(dst / split / folder / (image_id + '.txt')), annotation_lines[image_id])
        rows.append([image_id, split] + [counts[image_id].get(c, 0) for c in labels.VEDAI_CLASSES.values()])
    with (dst / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'split'] + list(labels.VEDAI_CLASSES.values()))
        writer.writerows(rows)
    print('VEDAI split: ' + str(Counter(assignment.values())), flush=True)


def audit_dataset(name, classes, expected_shape, workers=4):
    root = ROOT / 'datasets/prepared' / name
    report, seen, seen_content = {}, set(), {}
    for split in SPLITS:
        dirs = {folder: root / split / folder for folder in
                ('image_vi', 'image_ir', 'annofiles_vi', 'annofiles_ir')}
        indices = {folder: {p.stem for p in path.iterdir()} for folder, path in dirs.items()}
        ids = indices['image_vi']
        if any(value != ids for value in indices.values()):
            raise ValueError(f'{name}/{split}: image/label IDs do not match')
        if name == 'vedai' and ids & seen:
            raise ValueError(f'{name}/{split}: split leakage: {sorted(ids & seen)[:10]}')
        seen.update(ids)
        counts = {modality: Counter() for modality in ('vi', 'ir')}
        empty = Counter()
        outside = Counter()
        different_labels = 0
        duplicate_pairs = []
        repaired_images = []
        removed_degenerate_boxes = []
        def inspect_pair(image_id):
            local_counts = {modality: Counter() for modality in ('vi', 'ir')}
            local_empty, local_outside, local_repaired = Counter(), Counter(), []
            local_degenerate = []
            texts = {}
            digest = hashlib.blake2b(digest_size=20)
            for modality in ('vi', 'ir'):
                path = dirs['image_' + modality] / (image_id + '.png')
                try:
                    with Image.open(path) as check:
                        check.load()
                except OSError:
                    if name != 'dronevehicle':
                        raise
                    suffix = 'img' if modality == 'vi' else 'imgr'
                    source = ROOT / 'datasets' / split / split / (split + suffix) / (image_id + '.jpg')
                    images.convert(str(source), str(path), images.DRONEVEHICLE_CROP, modality == 'ir')
                    local_repaired.append(str(path.relative_to(root)))
                # Decode every image, including hard links, rather than trusting extensions.
                with Image.open(path) as im:
                    im.load()
                    if im.format != 'PNG' or im.size != expected_shape:
                        raise ValueError(f'Unexpected image format/shape: {path}: {im.format}, {im.size}')
                    digest.update(im.tobytes())
                annotation = dirs['annofiles_' + modality] / (image_id + '.txt')
                texts[modality] = annotation.read_bytes()
                lines = texts[modality].decode('utf-8').splitlines()
                valid_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) != 10 or parts[8] not in classes:
                        raise ValueError(f'Invalid label: {annotation}: {line}')
                    coords = list(map(float, parts[:8]))
                    if not all(math.isfinite(x) for x in coords):
                        raise ValueError(f'Non-finite coordinates: {annotation}')
                    int(parts[9])
                    points = list(zip(coords[::2], coords[1::2]))
                    area = abs(sum(points[i][0] * points[(i+1)%4][1] -
                                   points[(i+1)%4][0] * points[i][1] for i in range(4))) / 2
                    if area <= 0:
                        local_degenerate.append(dict(split=split, image_id=image_id, modality=modality, annotation=line))
                        continue
                    valid_lines.append(line)
                    if any(x < 0 or y < 0 or x > expected_shape[0] or y > expected_shape[1] for x,y in points):
                        local_outside[modality] += 1
                    local_counts[modality][parts[8]] += 1
                if len(valid_lines) != len(lines):
                    texts[modality] = (''.join(line + '\n' for line in valid_lines)).encode('utf-8')
                    annotation.write_bytes(texts[modality])
                if not valid_lines:
                    local_empty[modality] += 1
            return image_id, digest.hexdigest(), local_counts, local_empty, local_outside, local_repaired, texts['vi'] != texts['ir'], local_degenerate

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for progress, result in enumerate(pool.map(inspect_pair, sorted(ids)), 1):
                image_id, fingerprint, pair_counts, pair_empty, pair_outside, pair_repaired, different, pair_degenerate = result
                for modality in ('vi', 'ir'):
                    counts[modality].update(pair_counts[modality])
                empty.update(pair_empty)
                outside.update(pair_outside)
                repaired_images.extend(pair_repaired)
                removed_degenerate_boxes.extend(pair_degenerate)
                different_labels += different
                previous = seen_content.get(fingerprint)
                if previous and previous[0] != split:
                    duplicate_pairs.append(dict(image_id=image_id, matches_split=previous[0], matches_id=previous[1]))
                seen_content[fingerprint] = (split, image_id)
                if progress % 2000 == 0:
                    print(f'Auditing {name}/{split}: {progress}/{len(ids)}', flush=True)
        report[split] = dict(pairs=len(ids), objects=counts, empty_labels=empty,
                             out_of_bounds_objects=outside, pairs_with_different_labels=different_labels,
                             identical_pairs_across_splits=duplicate_pairs, repaired_images=repaired_images,
                             removed_degenerate_boxes=removed_degenerate_boxes)
        print(f'Audited {name}/{split}: {len(ids)} pairs', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dataset', choices=['dronevehicle', 'vedai', 'all'], default='dronevehicle')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if not args.audit_only:
        if args.dataset in ('dronevehicle', 'all'):
            prepare_dronevehicle(args.workers)
        if args.dataset in ('vedai', 'all'):
            prepare_vedai()
    report = {}
    if args.dataset in ('dronevehicle', 'all'):
        report['dronevehicle'] = audit_dataset('dronevehicle', labels.DRONEVEHICLE_CLASSES, (640, 512))
    if args.dataset in ('vedai', 'all'):
        report['vedai'] = audit_dataset('vedai', labels.VEDAI_CLASSES.values(), (512, 512))
    target = ROOT / 'results/dataset_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report: ' + str(target), flush=True)


if __name__ == '__main__':
    main()
