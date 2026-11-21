# Copyright (c) OpenMMLab. All rights reserved.
"""兼容转发：实现位于 ``mmrotate.evaluation.metrics.dota_per_class``。

本文件原先自带一份 ``DOTAMetricPerClass``，与 mmrotate 内的同名类同时注册到
``mmrotate.registry.METRICS``，导入顺序无论先后都会触发
``KeyError: DOTAMetricPerClass is already registered``。这里改为转发导入，
保留旧路径（``custom_imports`` 与 ``training/evaluate.py``）可用。
"""
from mmrotate.evaluation.metrics.dota_per_class import (  # noqa: F401
    DOTAMetricPerClass,
)

__all__ = ['DOTAMetricPerClass']
