"""Convert the labels of DroneVehicle and VEDAI into DOTA style txt files.

The DMM model expects, for every split, two folders with one txt per image::

    <dst>/<split>/annofiles_vi/<id>.txt    label of the visible image
    <dst>/<split>/annofiles_ir/<id>.txt    label of the infrared image

Every line is ``x1 y1 x2 y2 x3 y3 x4 y4 class difficult``. The visible and the
infrared image of a pair share the same geometry, so both folders get the same
content.

Both datasets are labelled with quads (``<polygon>`` in DroneVehicle, the four
corners in the VEDAI annotation files), which are written unchanged. The
``<bndbox>`` values are only used when no quad is available. VEDAI annotations
contain a few objects that stick out of the image; the coordinates are written
as they are.

Class names must not contain spaces because the files are space separated, so
VEDAI's ``camping car`` becomes ``camping_car``.

Usage::

    python tools/VOC2DOTA.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
    python tools/VOC2DOTA.py --dataset vedai --src data/VEDAI --dst data/vedai \
        --raw_annotations data/VEDAI/Annotations512
"""

import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

SPLITS = ("train", "val", "test")

# DroneVehicle crops 100 px of white border away, so every box moves by -100
DRONEVEHICLE_SHIFT = -100.0
DRONEVEHICLE_CLASSES = ("car", "truck", "bus", "van", "freight-car")
DRONEVEHICLE_ALIASES = {
    "freight-car": "freight-car",
    "freight car": "freight-car",
    "freight_car": "freight-car",
    "feright car": "freight-car",
    "feright_car": "freight-car",
    "feright": "freight-car",
    "*": "freight-car",
}

VEDAI_CLASSES = {
    1: "car",
    2: "truck",
    4: "tractor",
    5: "camping_car",
    9: "van",
    11: "pickup",
}
VEDAI_ALIASES = {
    "car": "car",
    "truck": "truck",
    "tractor": "tractor",
    "camping car": "camping_car",
    "camping_car": "camping_car",
    "van": "van",
    "pickup": "pickup",
}


def order_points(points):
    """Sort the corners so that they form a simple, closed polygon."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def format_line(points, class_name, difficult, shift):
    pts = order_points(points) + shift
    coords = " ".join("{0:.1f}".format(value) for value in pts.reshape(-1))
    return "{0} {1} {2}\n".format(coords, class_name, difficult)


def normalize_class(name, dataset):
    key = " ".join(name.strip().lower().split())
    if dataset == "dronevehicle":
        key = DRONEVEHICLE_ALIASES.get(key, key)
        return key if key in DRONEVEHICLE_CLASSES else None
    return VEDAI_ALIASES.get(key)


def polygon_of(obj):
    """Return the four corners of an object, or None if it has no usable box."""
    polygon = obj.find("polygon")
    if polygon is not None:
        pairs = [("x{0}".format(i), "y{0}".format(i)) for i in range(1, 5)]
        if all(polygon.find(x) is not None and polygon.find(y) is not None for x, y in pairs):
            return [(float(polygon.find(x).text), float(polygon.find(y).text)) for x, y in pairs]

    bndbox = obj.find("bndbox")
    if bndbox is not None and bndbox.find("xmin") is not None:
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return None


def lines_from_xml(xml_path, dataset, shift):
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall("object"):
        name = normalize_class(obj.findtext("name") or "", dataset)
        if name is None:
            continue
        points = polygon_of(obj)
        if points is None:
            continue
        difficult = (obj.findtext("difficult") or "0").strip() or "0"
        lines.append(format_line(points, name, int(float(difficult)), shift))
    return lines


def lines_from_vedai_raw(txt_path, shift):
    lines = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) < 14:
                continue
            class_id = int(float(parts[3]))
            name = VEDAI_CLASSES.get(class_id)
            if name is None:
                continue
            xs = [float(value) for value in parts[6:10]]
            ys = [float(value) for value in parts[10:14]]
            difficult = int(float(parts[5]))  # is_occluded
            lines.append(format_line(list(zip(xs, ys)), name, difficult, shift))
    return lines


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def prepare_split(source_root, output_root, split, dataset, shift, raw_annotations, overwrite):
    label_vi = osp.join(source_root, split, "{0}label".format(split))
    if not osp.isdir(label_vi):
        print("Skip {0}: no label folder {1}".format(split, label_vi))
        return 0, 0

    output_vi = osp.join(output_root, split, "annofiles_vi")
    output_ir = osp.join(output_root, split, "annofiles_ir")
    os.makedirs(output_vi, exist_ok=True)
    os.makedirs(output_ir, exist_ok=True)

    names = sorted(name for name in os.listdir(label_vi) if name.lower().endswith(".xml"))
    written = 0
    skipped = 0
    for name in tqdm(names, desc="{0:<5}".format(split)):
        image_id = osp.splitext(name)[0]
        target_vi = osp.join(output_vi, image_id + ".txt")
        target_ir = osp.join(output_ir, image_id + ".txt")
        if not overwrite and osp.exists(target_vi) and osp.exists(target_ir):
            written += 1
            continue

        lines = None
        if raw_annotations:
            raw_path = osp.join(raw_annotations, image_id + ".txt")
            if osp.isfile(raw_path):
                lines = lines_from_vedai_raw(raw_path, shift)
        if lines is None:
            lines = lines_from_xml(osp.join(label_vi, name), dataset, shift)

        if not lines:
            skipped += 1
            continue
        write_lines(target_vi, lines)
        write_lines(target_ir, lines)
        written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description="Convert VOC style labels into DOTA style txt files for DMM.")
    parser.add_argument("--dataset", choices=["dronevehicle", "vedai"], required=True,
                        help="dataset that is being converted")
    parser.add_argument("--src", required=True, help="root folder containing train/val/test")
    parser.add_argument("--dst", required=True, help="output root, e.g. data/dv")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), help="splits to convert (default: train val test)")
    parser.add_argument("--shift", type=float, default=None,
                        help="value added to every coordinate (default: -100 for DroneVehicle, 0 for VEDAI)")
    parser.add_argument("--raw_annotations", default=None,
                        help="VEDAI annotation folder (Annotations512/Annotations1024) with the original four corner labels")
    parser.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    args = parser.parse_args()

    shift = args.shift
    if shift is None:
        shift = DRONEVEHICLE_SHIFT if args.dataset == "dronevehicle" else 0.0
    if args.dataset == "dronevehicle" and shift == 0.0:
        print("WARNING: boxes are not shifted but the images are cropped, the labels will not match.")

    print("Dataset : {0}".format(args.dataset))
    print("Source  : {0}".format(osp.abspath(args.src)))
    print("Output  : {0}".format(osp.abspath(args.dst)))
    print("Shift   : {0}".format(shift))
    if args.raw_annotations:
        print("Raw anns: {0}".format(osp.abspath(args.raw_annotations)))

    for split in args.splits:
        written, skipped = prepare_split(args.src, args.dst, split, args.dataset, shift,
                                         args.raw_annotations, args.overwrite)
        print("{0:<6}{1:>8} label files, {2} images without target class".format(split, written, skipped))


if __name__ == "__main__":
    main()
