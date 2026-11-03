# VEDAI

this is vehicle detection repository of Vehicle Detection on Aerial Imagery(VEDAI) based on Tensorflow object detection API.

annotations512 and annotations1024 folder contain annotations descriptions of each images labelled separately in 512 X 512 pixels and 1024 X 1024 pixels.

data folder contains the object_detection.pbtxt which indicates the class of item to be detected.

image_512 folder contains VEDAI images in 512 * 512 pixels downloaded from https://downloads.greyc.fr/vedai/.

image_1024 folder contains VEDAI images in 1024 * 1024 pixels downloaded from https://downloads.greyc.fr/vedai/.

main_vedai_512.py and main_vedai_1024.py use to generate XML files by extracting txt files from annotations512 and annotations1024. the XML files will be stored separately in image_512 and image_1024.

Only 6 classes are exported to XML: 1 car, 2 truck, 4 tractor, 5 camping car, 9 van, 11 pickup (see class.txt). Bus and the other classes are skipped; images without any of these classes do not get an XML file.

Use `--annotation_dir` and `--output_dir` to point the script at your annotation folder and output folder. Without arguments it looks for `annotations512`, `data/VEDAI/Annotations512` or `../data/VEDAI/Annotations512` relative to the working directory; the XML files then go to `image_512` (legacy layout) or to `AnnotationsXML512` next to the annotation folder. Example from the repository root:

```
python "code/main_vedai_512 -  Copy.py"
# Annotations: data\VEDAI\Annotations512
# Output dir : data\VEDAI\AnnotationsXML512
```

xml_to_csv_vedai.py converts the XML files to CSV files. please modify the codes to generate the different pixels image. CSV files will be recorded in data folder.

generate_tfrecord_vedai.py creates the train.record and test.record. files will be recorded in data folder.

split_vedai_dataset.py splits the generated XML files into train / val / test (default 0.7 / 0.1 / 0.2) with a DroneVehicle-like layout:

```
data/VEDAI/train/trainimg/{id}.png       color image
data/VEDAI/train/trainimgr/{id}.png      infrared image
data/VEDAI/train/trainlabel/{id}.xml     label of the color image
data/VEDAI/train/trainlabelr/{id}.xml    label of the infrared image (same boxes)
data/VEDAI/val/...  data/VEDAI/test/...
data/VEDAI/split_manifest.csv            image id, split and per-class object counts
```

The split is stratified by class (rarest class first) and every class that has at least 3 images is forced into each split. Images are hard linked by default (`--mode copy` copies them), so the split folders take almost no extra disk space on the same volume. Only images with an XML file are split, i.e. images containing at least one of the 6 target classes.

```
python "code/scripts/split_vedai_dataset.py"                     # 70/10/20, seed 42
python "code/scripts/split_vedai_dataset.py" --ratios 0.8 0.1 0.1 --mode copy --force
```

rebuild_vedai.py does the whole data pipeline in one command: re-split, rebuild the visible/infrared image pairs and convert the labels to DOTA format. It wipes `datasets/vedai` first so that files from an older split cannot leak into the new one, then prints a verification report (per-split counts, overlap between splits, class distribution).

```
python "code/scripts/rebuild_vedai.py"                           # 70/10/20, seed 42
python "code/scripts/rebuild_vedai.py" --ratios 0.8 0.1 0.1 --seed 2024
```

The older pipeline expected a manual split: once XML files are generated, take parts into two set (training set and testing set) and place them to directory: images/train and images/test separately. `split_vedai_dataset.py` replaces that manual step; the legacy `images/train` + `images/test` layout is only needed when you keep using xml_to_csv_vedai.py / generate_tfrecord_vedai.py unchanged.

basically refer to :https://github.com/mr8bit/vedai
