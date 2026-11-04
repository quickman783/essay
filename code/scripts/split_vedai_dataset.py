"""
Split the VEDAI dataset into train / val / test sets with a DroneVehicle-like layout:

    <output_dir>/train/trainimg/{id}.png       color image
    <output_dir>/train/trainimgr/{id}.png      infrared image
    <output_dir>/train/trainlabel/{id}.xml     label of the color image
    <output_dir>/train/trainlabelr/{id}.xml    label of the infrared image (same boxes)
    <output_dir>/val/... and <output_dir>/test/...

Only images that have an XML file (i.e. contain at least one of the 6 target
classes) are split. The split is stratified by class so that every class keeps
approximately the same ratio in train, val and test.

Usage (from the repository root):
    python "code/scripts/split_vedai_dataset.py"
    python "code/scripts/split_vedai_dataset.py" --ratios 0.8 0.1 0.1 --mode copy --force
"""

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict

import xml.etree.ElementTree as ET

SPLITS = ['train', 'val', 'test']

# 默认按 7:1:2 划分训练 / 验证 / 测试集
DEFAULT_RATIOS = [0.7, 0.2, 0.1]

CLASS_NAMES = ['car', 'truck', 'tractor', 'camping car', 'van', 'pickup']


def find_dir(explicit, candidates, description):
    if explicit:
        return explicit
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit('{0} not found. Pass the matching option explicitly.'.format(description))


def index_images(image_dir):
    """Map image id -> file name for the color and infrared images."""
    color = {}
    infrared = {}
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != '.png':
            continue
        if stem.endswith('_co'):
            color[stem[:-3]] = name
        elif stem.endswith('_ir'):
            infrared[stem[:-3]] = name
    return color, infrared


def load_labels(xml_dir):
    labels = {}
    for name in sorted(os.listdir(xml_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() != '.xml' or not stem.isdigit():
            continue
        root = ET.parse(os.path.join(xml_dir, name)).getroot()
        labels[stem] = Counter(o.find('name').text for o in root.findall('object'))
    return labels


def stratified_split(image_ids, labels, ratios, seed):
    """Iterative stratification: rarest class first, assign to the split with the largest deficit."""
    rng = random.Random(seed)
    total = len(image_ids)

    class_members = defaultdict(list)
    for image_id in image_ids:
        for class_name in labels[image_id]:
            class_members[class_name].append(image_id)
    class_total = {name: len(members) for name, members in class_members.items()}

    assigned = {}
    split_size = {split: 0 for split in SPLITS}
    class_assigned = {name: {split: 0 for split in SPLITS} for name in class_members}

    for class_name in sorted(class_members, key=lambda name: (class_total[name], name)):
        members = [i for i in class_members[class_name] if i not in assigned]
        rng.shuffle(members)
        for image_id in members:
            if image_id in assigned:
                continue
            best_split = None
            best_score = None
            for index, split in enumerate(SPLITS):
                class_deficit = ratios[index] * class_total[class_name] - class_assigned[class_name][split]
                size_deficit = ratios[index] * total - split_size[split]
                score = (class_deficit, size_deficit)
                if best_score is None or score > best_score:
                    best_score = score
                    best_split = split
            assigned[image_id] = best_split
            split_size[best_split] += 1
            for name in labels[image_id]:
                class_assigned[name][best_split] += 1

    return assigned


def ensure_class_coverage(assigned, labels):
    """Move single images between splits so every class appears in every split when it has enough images."""
    all_classes = sorted({name for counts in labels.values() for name in counts})
    for class_name in all_classes:
        total = sum(1 for counts in labels.values() if class_name in counts)
        if total < len(SPLITS):
            continue
        counts = {split: sum(1 for i in assigned if assigned[i] == split and class_name in labels[i])
                  for split in SPLITS}
        for split in SPLITS:
            if counts[split] > 0:
                continue
            for donor in sorted(SPLITS, key=lambda name: -counts[name]):
                if counts[donor] < 2:
                    continue
                candidates = [i for i in assigned if assigned[i] == donor and class_name in labels[i]]
                candidates.sort(key=lambda i: (len(labels[i]), i))
                for image_id in candidates:
                    if counts[donor] < 2:
                        break
                    assigned[image_id] = split
                    counts[donor] -= 1
                    counts[split] += 1
                    print('Move {0} ({1}) from {2} to {3} to keep the class in every split.'.format(
                        image_id, class_name, donor, split))
                    break
                if counts[split] > 0:
                    break


def link_or_copy(source, destination, mode):
    if os.path.exists(destination):
        os.remove(destination)
    if mode == 'hardlink':
        try:
            os.link(source, destination)
            return True
        except OSError:
            pass
    shutil.copy2(source, destination)
    return False


def write_label(source_xml, destination_xml, split, target_folder, image_name):
    tree = ET.parse(source_xml)
    root = tree.getroot()
    folder = root.find('folder')
    if folder is None:
        folder = ET.SubElement(root, 'folder')
    folder.text = target_folder
    filename = root.find('filename')
    if filename is None:
        filename = ET.SubElement(root, 'filename')
    filename.text = image_name
    path = root.find('path')
    if path is None:
        path = ET.SubElement(root, 'path')
    path.text = '{0}/{1}/{2}'.format(split, target_folder, image_name)
    tree.write(destination_xml)


def split_dataset(xml_dir, image_dir, output_dir, ratios=DEFAULT_RATIOS, seed=42,
                  mode='hardlink', force=False):
    """按 ratios 把 VEDAI 划分为 train/val/test，返回 {image_id: split}。"""
    ratios = list(ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        total = sum(ratios)
        ratios = [r / total for r in ratios]
    if any(r < 0 for r in ratios):
        raise SystemExit('Ratios must be non-negative.')

    existing = [os.path.join(output_dir, split) for split in SPLITS if os.path.isdir(os.path.join(output_dir, split))]
    if existing and not force:
        raise SystemExit('Output folders already exist ({0}). Re-run with --force to overwrite them.'.format(', '.join(existing)))
    for split in SPLITS:
        path = os.path.join(output_dir, split)
        if os.path.isdir(path):
            shutil.rmtree(path)

    labels = load_labels(xml_dir)
    color_images, infrared_images = index_images(image_dir)

    image_ids = []
    missing_color = []
    for image_id in sorted(labels):
        if image_id not in color_images:
            missing_color.append(image_id)
            continue
        image_ids.append(image_id)
    if missing_color:
        print('WARNING: {0} XML files have no color image, skipped: {1}'.format(len(missing_color), ', '.join(missing_color[:10])))

    assigned = stratified_split(image_ids, labels, ratios, seed)
    ensure_class_coverage(assigned, labels)

    for split in SPLITS:
        split_dir = os.path.join(output_dir, split)
        for sub in ['{0}img'.format(split), '{0}imgr'.format(split), '{0}label'.format(split), '{0}labelr'.format(split)]:
            os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

    manifest_rows = []
    copied = 0
    for image_id in sorted(assigned):
        split = assigned[image_id]
        split_dir = os.path.join(output_dir, split)
        color_name = '{0}.png'.format(image_id)
        color_src = os.path.join(image_dir, color_images[image_id])
        color_dst = os.path.join(split_dir, '{0}img'.format(split), color_name)
        if link_or_copy(color_src, color_dst, mode) is False and mode == 'hardlink':
            copied += 1

        write_label(os.path.join(xml_dir, '{0}.xml'.format(image_id)),
                    os.path.join(split_dir, '{0}label'.format(split), '{0}.xml'.format(image_id)),
                    split, '{0}img'.format(split), color_name)

        labelr = ''
        ir_dst = None
        if image_id in infrared_images:
            ir_src = os.path.join(image_dir, infrared_images[image_id])
            ir_dst = os.path.join(split_dir, '{0}imgr'.format(split), color_name)
            if link_or_copy(ir_src, ir_dst, mode) is False and mode == 'hardlink':
                copied += 1
            labelr = os.path.join(split_dir, '{0}labelr'.format(split), '{0}.xml'.format(image_id))
            write_label(os.path.join(xml_dir, '{0}.xml'.format(image_id)), labelr,
                        split, '{0}imgr'.format(split), color_name)

        row = [split, image_id,
               os.path.relpath(color_dst, output_dir),
               os.path.relpath(ir_dst, output_dir) if ir_dst else '',
               os.path.relpath(os.path.join(split_dir, '{0}label'.format(split), '{0}.xml'.format(image_id)), output_dir),
               os.path.relpath(labelr, output_dir) if labelr else '']
        row += [labels[image_id].get(name, 0) for name in CLASS_NAMES]
        manifest_rows.append(row)

    manifest_path = os.path.join(output_dir, 'split_manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'image_id', 'image', 'image_ir', 'label', 'label_ir'] + CLASS_NAMES)
        writer.writerows(manifest_rows)

    print('Split {0} images from {1}'.format(len(assigned), xml_dir))
    print('Output   : {0}'.format(os.path.abspath(output_dir)))
    print('Manifest : {0}'.format(os.path.abspath(manifest_path)))
    print('')
    header = '{0:<8}{1:>8}'.format('split', 'images')
    header += ''.join('{0:>12}'.format(name) for name in CLASS_NAMES)
    print(header)
    for split in SPLITS:
        split_ids = [i for i in assigned if assigned[i] == split]
        print('{0:<8}{1:>8}'.format(split, len(split_ids)) + ''.join(
            '{0:>12}'.format(sum(labels[i].get(name, 0) for i in split_ids)) for name in CLASS_NAMES))
    print('{0:<8}{1:>8}'.format('total', len(assigned)) + ''.join(
        '{0:>12}'.format(sum(labels[i].get(name, 0) for i in assigned)) for name in CLASS_NAMES))
    if copied:
        print('\n{0} images were copied instead of hard linked.'.format(copied))
    return assigned


def main():
    parser = argparse.ArgumentParser(description='Split VEDAI into train/val/test with a DroneVehicle-like layout.')
    parser.add_argument('--xml_dir', default=None, help='directory with the generated XML files (default: data/VEDAI/AnnotationsXML512)')
    parser.add_argument('--image_dir', default=None, help='directory with the VEDAI images (default: data/VEDAI/Vehicules512)')
    parser.add_argument('--output_dir', default=None, help='output root, train/val/test are created inside it (default: data/VEDAI)')
    parser.add_argument('--ratios', nargs=3, type=float, default=DEFAULT_RATIOS,
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='image ratios of the three splits (default: 0.7 0.1 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='random seed used for the stratified split (default: 42)')
    parser.add_argument('--mode', choices=['hardlink', 'copy'], default='hardlink', help='how images are placed in the split folders (default: hardlink, falls back to copy)')
    parser.add_argument('--force', action='store_true', help='delete existing train/val/test folders inside the output directory first')
    args = parser.parse_args()

    xml_dir = find_dir(args.xml_dir, ['data/VEDAI/AnnotationsXML512', '../data/VEDAI/AnnotationsXML512'],
                       'XML directory')
    image_dir = find_dir(args.image_dir, ['data/VEDAI/Vehicules512', '../data/VEDAI/Vehicules512'],
                         'image directory')
    output_dir = find_dir(args.output_dir, ['data/VEDAI', '../data/VEDAI'], 'output directory')

    split_dataset(xml_dir, image_dir, output_dir, args.ratios, args.seed, args.mode, args.force)


if __name__ == '__main__':
    main()
