"""Prepare the visible / infrared image pairs of DroneVehicle and VEDAI.

The DMM model expects, for every split, two folders that share the file names::

    <dst>/<split>/image_vi/<id>.png    visible image
    <dst>/<split>/image_ir/<id>.png    infrared image

``tools/VOC2DOTA.py`` writes the matching ``annofiles_vi`` / ``annofiles_ir``.

DroneVehicle images are 840x712 and contain a 100 px white border that is
cropped away, which makes the pairs 640x512. VEDAI images are already 512x512,
so they are only re-linked (or copied) under the expected names.

Usage::

    python tools/data_process.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
    python tools/data_process.py --dataset vedai --src data/VEDAI --dst data/vedai
"""

import argparse
import os
import os.path as osp
import shutil

import cv2
import numpy as np
from tqdm import tqdm

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# (y0, y1, x0, x1): DroneVehicle adds a 100 px white border to every image
DRONEVEHICLE_CROP = (100, 612, 100, 740)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Read an image; cv2.imread rejects non ASCII paths on Windows."""
    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        raise OSError("Cannot read image: {0}".format(path))
    return cv2.imdecode(buffer, flags)


def imwrite_unicode(path, image):
    """Write an image; cv2.imwrite rejects non ASCII paths on Windows."""
    ext = osp.splitext(path)[1]
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise OSError("Cannot encode image: {0}".format(path))
    buffer.tofile(path)


def list_images(directory):
    if not osp.isdir(directory):
        return []
    return sorted(
        name
        for name in os.listdir(directory)
        if osp.splitext(name)[1].lower() in IMAGE_SUFFIXES
    )


def link_or_copy(source, destination):
    if osp.exists(destination):
        os.remove(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert(source, destination, crop=None, to_gray=False):
    if crop is None and not to_gray:
        link_or_copy(source, destination)
        return
    image = imread_unicode(source)
    if crop is not None:
        y0, y1, x0, x1 = crop
        image = image[y0:y1, x0:x1]
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imwrite_unicode(destination, image)


def prune(output_dir, keep_ids, suffix):
    """删掉本划分目录里不属于当前源划分的旧文件，避免上一次划分的残留混进来。"""
    removed = 0
    for name in os.listdir(output_dir):
        stem, ext = osp.splitext(name)
        if ext.lower() not in suffix:
            continue
        if stem not in keep_ids:
            os.remove(osp.join(output_dir, name))
            removed += 1
    return removed


def prepare_split(source_root, output_root, split, crop, ir_to_gray, overwrite):
    source_vi = osp.join(source_root, split, "{0}img".format(split))
    source_ir = osp.join(source_root, split, "{0}imgr".format(split))
    output_vi = osp.join(output_root, split, "image_vi")
    output_ir = osp.join(output_root, split, "image_ir")
    os.makedirs(output_vi, exist_ok=True)
    os.makedirs(output_ir, exist_ok=True)

    names = list_images(source_vi)
    if not names:
        print("Skip {0}: no image found in {1}".format(split, source_vi))
        return 0

    ids = {osp.splitext(name)[0] for name in names}
    if overwrite:
        for folder in (output_vi, output_ir):
            cleaned = prune(folder, ids, IMAGE_SUFFIXES)
            if cleaned:
                print("{0}: 清掉 {1} 张不属于本划分的旧图片".format(split, cleaned))

    infrared = set(list_images(source_ir))
    missing_ir = 0
    for name in tqdm(names, desc="{0:<5}".format(split)):
        png_name = osp.splitext(name)[0] + ".png"
        destination = osp.join(output_vi, png_name)
        if overwrite or not osp.exists(destination):
            convert(osp.join(source_vi, name), destination, crop, to_gray=False)

        if name not in infrared:
            missing_ir += 1
            continue
        destination_ir = osp.join(output_ir, png_name)
        if overwrite or not osp.exists(destination_ir):
            convert(osp.join(source_ir, name), destination_ir, crop, to_gray=ir_to_gray)

    if missing_ir:
        print("WARNING: {0} images of split {1} have no infrared pair.".format(missing_ir, split))
    return len(names)


def main():
    parser = argparse.ArgumentParser(description="Prepare RGB/IR image pairs for DMM.")
    parser.add_argument("--dataset", choices=["dronevehicle", "vedai"], required=True,
                        help="dataset that is being prepared")
    parser.add_argument("--src", required=True, help="root folder containing train/val/test")
    parser.add_argument("--dst", required=True, help="output root, e.g. data/dv")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        help="splits to prepare (default: train val test)")
    parser.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    parser.add_argument("--no-crop", action="store_true",
                        help="keep the original image size (the DroneVehicle border is kept)")
    parser.add_argument("--ir-rgb", action="store_true",
                        help="keep infrared images in colour instead of converting them to grayscale")
    args = parser.parse_args()

    crop = None if (args.dataset == "vedai" or args.no_crop) else DRONEVEHICLE_CROP
    ir_to_gray = args.dataset == "dronevehicle" and not args.ir_rgb

    print("Dataset : {0}".format(args.dataset))
    print("Source  : {0}".format(osp.abspath(args.src)))
    print("Output  : {0}".format(osp.abspath(args.dst)))
    print("Crop    : {0}".format("none" if crop is None else "y{0}:{1} x{2}:{3}".format(*crop)))

    totals = {}
    for split in args.splits:
        totals[split] = prepare_split(args.src, args.dst, split, crop, ir_to_gray, args.overwrite)

    print("")
    for split in args.splits:
        print("{0:<6}{1:>8} image pairs".format(split, totals[split]))


if __name__ == "__main__":
    main()
