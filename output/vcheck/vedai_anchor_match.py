"""诊断：VEDAI 的 GT 与本仓库 S2ANet anchor 尺寸的 IoU 匹配度。

anchor 由 FakeRotatedAnchorGenerator(scales=[4], ratios=[1.0], strides=[8,16,32,64,128])
生成 => anchor 边长 = 4 * stride = 32/64/128/256/512 的正方形。
判定 train_cfg 里 MaxIoUAssigner(pos_iou_thr=0.5) 能否命中正样本。
"""
import glob
import numpy as np

ANCHOR_SIDES = [32, 64, 128, 256, 512]


def iou_hbb(t, a):
    """t: (w,h) target; a: (w,h) axis-aligned anchor, 同心摆放求最大 IoU。"""
    tw, th = t
    aw, ah = a
    iw = min(tw, aw)
    ih = min(th, ah)
    inter = iw * ih
    union = tw * th + aw * ah - inter
    return inter / max(union, 1e-9)


def iou_at_offset(t, a, dx, dy):
    """target 相对 anchor 平移 (dx,dy) 后的 IoU（同心时 IoU 最大）。"""
    tw, th = t
    aw, ah = a
    ow = min(tw, aw) - abs(dx)
    oh = min(th, ah) - abs(dy)
    if ow <= 0 or oh <= 0:
        return 0.0
    inter = ow * oh
    union = tw * th + aw * ah - inter
    return inter / max(union, 1e-9)


for split in ['train', 'val', 'test']:
    files = glob.glob(f'datasets/vedai/{split}/annofiles_ir/*.txt')
    best = []
    for p in files:
        for line in open(p, errors='ignore'):
            v = line.split()
            if len(v) < 9:
                continue
            pts = np.array([[float(v[0]), float(v[1])], [float(v[2]), float(v[3])],
                            [float(v[4]), float(v[5])], [float(v[6]), float(v[7])]])
            w = np.linalg.norm(pts[1] - pts[0])
            h = np.linalg.norm(pts[2] - pts[1])
            tw, th = max(w, h), min(w, h)
            best.append(max(iou_hbb((tw, th), (s, s)) for s in ANCHOR_SIDES))
    best = np.array(best)
    if len(best) == 0:
        continue
    print(f'--- {split}: {len(files)} imgs, {len(best)} boxes')
    print(f'    best-IoU vs anchor: max={best.max():.3f} '
          f'P90={np.percentile(best, 90):.3f} median={np.median(best):.3f} '
          f'mean={best.mean():.3f}')
    for thr in (0.5, 0.4, 0.3, 0.2):
        print(f'    IoU >= {thr}: {(best >= thr).mean() * 100:5.1f}%  '
              f'({(best >= thr).sum()} / {len(best)})')

# 单目标示例：中位尺寸 21.8 x 8.9
print('\n=== 中位目标 21.8 x 8.9 px 的 IoU 敏感度 ===')
t = (21.8, 8.9)
for a in ANCHOR_SIDES:
    print(f'  anchor {a:3d}x{a:3d}: 同心 IoU = {iou_hbb(t, (a, a)):.3f}')
print('  --- 与最佳 anchor(32) 的偏移敏感度 ---')
for d in (0, 1, 2, 3, 4, 5):
    print(f'  dx={d}px IoU={iou_at_offset(t, (32, 32), d, 0):.3f}   '
          f'dy={d}px IoU={iou_at_offset(t, (32, 32), 0, d):.3f}')
