import numpy as np


def raster_iou(w, h, deg, res=0.2):
    """IoU between an axis-aligned box and the same box rotated by `deg` about its center."""
    pad = max(w, h) * 1.5
    n = int(2 * pad / res)
    ys, xs = np.mgrid[0:n, 0:n]
    xs = (xs - n / 2) * res
    ys = (ys - n / 2) * res

    def mask(theta):
        c, s = np.cos(theta), np.sin(theta)
        xr = c * xs + s * ys
        yr = -s * xs + c * ys
        return (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)

    m0 = mask(0.0)
    m1 = mask(np.deg2rad(deg))
    inter = (m0 & m1).sum()
    union = (m0 | m1).sum()
    return inter / max(union, 1)


print('long side x short side (median from VEDAI train)')
cases = [('tractor', 17.8, 12.0), ('car', 19.1, 8.0), ('van', 22.7, 8.5),
         ('pickup', 23.0, 8.5), ('camping_car', 32.0, 11.0), ('truck', 35.0, 11.2)]
print(f"{'class':13s} {'AR':>5s} {'IoU@2deg':>9s} {'IoU@5deg':>9s} {'IoU@10deg':>10s} {'max deg for IoU>0.5':>21s}")
for name, w, h in cases:
    row = [raster_iou(w, h, d) for d in (2, 5, 10)]
    # bisection for IoU = 0.5
    lo, hi = 0.0, 45.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if raster_iou(w, h, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    print(f'{name:13s} {w/h:5.2f} {row[0]:9.3f} {row[1]:9.3f} {row[2]:10.3f} {lo:21.1f}')
