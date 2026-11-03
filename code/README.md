# 《DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing》

The official implementation of the [paper](https://arxiv.org/abs/2407.08132).

## Dataset

DMM takes visible / infrared pairs. Both datasets are converted into the same layout, ``data/dv`` for DroneVehicle and ``data/vedai`` for VEDAI:

```
data/dv
└── train
    ├── image_vi/00001.png      visible image
    ├── image_ir/00001.png      infrared image
    ├── annofiles_vi/00001.txt  DOTA style label of the visible image
    └── annofiles_ir/00001.txt  label of the infrared image (same boxes)
```
``val`` and ``test`` use the same structure. Every annotation line is ``x1 y1 x2 y2 x3 y3 x4 y4 class difficult``.

### DroneVehicle

1. Download the dataset from https://github.com/VisDrone/DroneVehicle and unzip it so that ``data/DroneVehicle/{train,val,test}/{split}img,{split}imgr,{split}label,{split}labelr`` exist, then crop the 100 px white border (840x712 -> 640x512):
```shell
python tools/data_process.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
```

2. Convert the labels. The boxes are shifted by -100 px to match the cropped images and the inconsistent ``freight-car`` spellings (``feright car``, ``*``, ...) are unified:
```shell
python tools/VOC2DOTA.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
```

### VEDAI

1. Split the dataset first. Run ``python code/split_vedai_dataset.py`` from the repository root (the folder that contains ``code/`` and ``data/``); it writes ``data/VEDAI/{train,val,test}/{split}img,{split}imgr,{split}label,{split}labelr``. Then prepare the image pairs. VEDAI images are already 512x512, so they are only hard linked under the expected names:
```shell
python tools/data_process.py --dataset vedai --src data/VEDAI --dst data/vedai
```

2. Convert the labels. ``--raw_annotations`` makes the converter read the original four corner annotations, which keeps the oriented boxes instead of falling back to axis-aligned ones:
```shell
python tools/VOC2DOTA.py --dataset vedai --src data/VEDAI --dst data/vedai --raw_annotations data/VEDAI/Annotations512
```
VEDAI's ``camping car`` is written as ``camping_car`` because the annotation files are space separated. VEDAI's ``bus`` class is dropped (it has only 3 instances), so images that contain nothing but a bus are not part of the dataset.

### Classes

| dataset | classes |
| --- | --- |
| DroneVehicle | car, truck, bus, van, freight-car (5) |
| DroneVehicle config | ``configs/dmm/dmm_s2anet_1x_dronevehicle.py`` |
| VEDAI | car, truck, tractor, camping_car, van, pickup (6) |
| VEDAI config | ``configs/dmm/dmm_s2anet_1x_vedai.py`` |

The class list lives in the dataset class (``mmrotate/datasets/dronevehicle.py``, ``mmrotate/datasets/vedai.py``) and ``num_classes`` is set in the config; both have to match the annotation files.

## Weights

The model is trained from scratch: no checkpoint is downloaded and both backbones use ``pretrained=None`` (see ``configs/dmm/_base_/dmm_s2anet_rgbt.py``). The ImageNet VMamba checkpoint used in the paper can be re-enabled there with ``pretrained="weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth"``.

## Envirenment
CUDA==11.8

Pytorch==2.1.2

mmcv==2.1.0

mmdet==3.3.0

mmengine==0.10.5

numpy==1.26.4

You can follow the steps below to create an virtual environment:

1. install all dependencies:
```
conda create -n dmm python=3.10
conda activate dmm

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U openmim
mim install mmdet
pip install numpy==1.26.4
```

- You might encounter the following, Downgrade the pip version to 24.0 (pip install pip==24.0)
```
Ignoring mmcv: markers 'extra == "mim"' don't match your environment
Ignoring mmengine: markers 'extra == "mim"' don't match your environment
```


2. Follow the https://github.com/MzeroMiko/VMamba Getting Started Step 2, install selective_scan==0.0.2


3. Clone the code and install:
```
git clone https://github.com/Another-0/DMM
cd DMM
pip install -v -e .
```

## Run

1. train
```
python ./tools/train.py configs/dmm/dmm_s2anet_1x_dronevehicle.py
python ./tools/train.py configs/dmm/dmm_s2anet_1x_vedai.py
```

2. test
```
python ./tools/test.py configs/dmm/dmm_s2anet_1x_dronevehicle.py ${CHECKPOINT}
python ./tools/test.py configs/dmm/dmm_s2anet_1x_vedai.py ${CHECKPOINT}
```

Both configs share ``configs/dmm/_base_/dmm_s2anet_rgbt.py`` (model) and ``configs/dmm/_base_/schedule_dmm_1x.py`` (12 epochs, AdamW 1e-4); they only differ in the dataset and in ``num_classes``.


For more command-line arguments, please refer to the code details.

## Acknowledgment
Our codes are mainly based on [MMRotate](https://github.com/open-mmlab/mmrotate) and [VMamba](https://github.com/MzeroMiko/VMamba). Many thanks to the authors!

## Citation
Please cite our work if you find our work and codes helpful for your research.
```
@article{zhou2024dmm,
  title={DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing},
  author={Zhou, Minghang and Li, Tianyu and Qiao, Chaofan and Xie, Dongyu and Wang, Guoqing and Ruan, Ningjuan and Mei, Lin and Yang, Yang},
  journal={arXiv preprint arXiv:2407.08132},
  year={2024}
}
```