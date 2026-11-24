import glob
import numpy as np
import collections

NAMES = ['car', 'truck', 'tractor', 'camping_car', 'van', 'pickup']
ALIAS = {'campingcar': 'camping_car', 'camping_car': 'camping_car'}

for split in ['train', 'val', 'test']:
    cnt = collections.Counter()
    sizes = collections.defaultdict(list)
    files = glob.glob(f'datasets/vedai/{split}/annofiles_ir/*.txt')
    for p in files:
        for line in open(p, errors='ignore'):
            v = line.split()
            if len(v) < 9:
                continue
            name = v[8].strip().lower()
            name = ALIAS.get(name, name)
            cnt[name] += 1
            pts = np.array([[float(v[0]), float(v[1])], [float(v[2]), float(v[3])],
                            [float(v[4]), float(v[5])], [float(v[6]), float(v[7])]])
            w = np.linalg.norm(pts[1] - pts[0])
            h = np.linalg.norm(pts[2] - pts[1])
            sizes[name].append((max(w, h), min(w, h)))
    print(f'--- {split}: {len(files)} images, {sum(cnt.values())} boxes')
    for n in NAMES:
        s = np.array(sizes[n]) if sizes[n] else np.zeros((1, 2))
        ar = np.median(s[:, 0] / np.maximum(s[:, 1], 1e-6))
        print(f'  {n:12s} {cnt[n]:5d}  median {np.median(s[:, 0]):5.1f} x {np.median(s[:, 1]):4.1f} px  AR {ar:.2f}')
