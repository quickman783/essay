# Copyright (c) OpenMMLab. All rights reserved.
"""在 DOTAMetric 的基础上，额外输出每个类别的 AP。"""
import os.path as osp
import tempfile
from collections import OrderedDict

from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS

from .dota_metric import DOTAMetric


@METRICS.register_module()
class DOTAMetricPerClass(DOTAMetric):
    """DOTAMetric 的增强版。

    除了父类给出的 ``mAP`` 与各 IoU 阈值的 ``APxx``，还会输出每个类别的 AP，
    键名为 ``AP_<类别名>``，取 IoU=0.5 时的每类 AP（论文里通常报这个）。

    用法与 DOTAMetric 完全一致：

        test_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
    """

    def compute_metrics(self, results: list) -> dict:
        # 结果合并、仅做格式转换或指标不是 mAP 时，没有逐类 AP 可算，交给父类
        if self.merge_patches or self.format_only or self.metric != 'mAP':
            return super().compute_metrics(results)

        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        # 转成 json 是父类的既有行为（中间结果，留在临时目录里）
        _ = self.results2json(preds, outfile_prefix)

        class_names = self.dataset_meta['classes']
        dets = [pred['pred_bbox_scores'] for pred in preds]

        eval_results = OrderedDict()
        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info('-' * 15 + 'iou_thr: ' + str(iou_thr) + '-' * 15)
            mean_ap, per_class = eval_rbbox_map(
                dets,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=self.use_07_metric,
                box_type=self.predict_box_type,
                dataset=class_names,
                logger=logger)
            mean_aps.append(mean_ap)
            eval_results['AP{0:02d}'.format(int(iou_thr * 100))] = round(mean_ap, 3)
            if abs(iou_thr - 0.5) < 1e-6:
                for name, item in zip(class_names, per_class):
                    eval_results['AP_{0}'.format(name)] = round(float(item['ap']), 4)

        eval_results['mAP'] = round(sum(mean_aps) / len(mean_aps), 4)
        eval_results.move_to_end('mAP', last=False)
        return eval_results
