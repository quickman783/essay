"""Compare every prepared label with its own modality's original XML geometry."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('voc_converter', ROOT / 'code/DMM-main/tools/VOC2DOTA.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def inspect(task):
    split, modality, source = task
    result = Counter()
    expected = []
    anomalies = []
    for obj in ET.parse(source).getroot().findall('object'):
        name = converter.normalize_class(obj.findtext('name') or '', 'dronevehicle')
        result['source_objects'] += 1
        if name is None:
            result['unknown_class'] += 1
            anomalies.append(dict(reason='unknown_class', name=obj.findtext('name')))
            continue
        points = converter.polygon_of(obj)
        if points is None:
            result['missing_geometry'] += 1
            anomalies.append(dict(reason='missing_geometry', name=name))
            continue
        line = converter.format_line(points, name, int(float(obj.findtext('difficult') or '0')), -100.0)
        if line is None:
            result['zero_area_or_nonfinite'] += 1
            anomalies.append(dict(reason='zero_area_or_nonfinite', name=name))
            continue
        expected.append(line)
        result['valid_objects'] += 1
        xy = list(map(float, line.split()[:8]))
        if max(xy[::2]) < 0 or min(xy[::2]) > 640 or max(xy[1::2]) < 0 or min(xy[1::2]) > 512:
            result['fully_outside_image'] += 1
    target = ROOT / 'datasets/prepared/dronevehicle' / split / ('annofiles_' + modality) / (source.stem + '.txt')
    if target.read_text(encoding='utf-8') != ''.join(expected):
        raise ValueError('Prepared label differs from its source: ' + str(target))
    return split, modality, source.stem, result, anomalies


if __name__ == '__main__':
    tasks = []
    report = {}
    for split in ('train', 'val', 'test'):
        report[split] = {m: dict(counts=Counter(), anomalies=[]) for m in ('vi', 'ir')}
        for modality, suffix in [('vi', 'label'), ('ir', 'labelr')]:
            tasks.extend((split, modality, p) for p in sorted((ROOT / 'datasets' / split / split / (split + suffix)).glob('*.xml')))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (split, modality, image_id, counts, anomalies) in enumerate(pool.map(inspect, tasks), 1):
            report[split][modality]['counts'].update(counts)
            for anomaly in anomalies:
                report[split][modality]['anomalies'].append(dict(image_id=image_id, **anomaly))
            if index % 10000 == 0:
                print(f'Compared source labels: {index}/{len(tasks)}', flush=True)
    target = ROOT / 'results/dronevehicle_source_label_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('All prepared labels match their source XML. Report: ' + str(target), flush=True)
