# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS

from .dronevehicle import DVDataset


@DATASETS.register_module()
class VEDAIDataset(DVDataset):
    """VEDAI dataset in visible / infrared (co, ir) pair form.

    The annotation layout is identical to :class:`DVDataset`: one DOTA style
    txt file per image in ``annofiles_vi`` / ``annofiles_ir`` and one image per
    modality in ``image_vi`` / ``image_ir``. Only the classes differ.

    Class names may not contain spaces because the annotation files are space
    separated, hence ``camping_car`` instead of ``camping car``.
    """

    METAINFO = {
        "classes": (
            "car",
            "truck",
            "tractor",
            "camping_car",
            "van",
            "pickup",
        ),
        "palette": [
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (128, 0, 128),
            (0, 128, 255),
            (165, 42, 42),
        ],
    }
