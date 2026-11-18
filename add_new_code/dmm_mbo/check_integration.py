"""Verify real S2A losses, then a real-data DMM batch on CUDA (no training run)."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
EXTENSION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EXTENSION_ROOT.parent / 'code'
BASE_ROOT = Path(os.environ.get('DMM_BASE_ROOT', CODE_ROOT))
if not (BASE_ROOT / 'mmrotate/__init__.py').is_file():
    BASE_ROOT = CODE_ROOT / 'DMM-main'
os.environ['DMM_BASE_ROOT'] = str(BASE_ROOT)
sys.path.insert(0, str(BASE_ROOT))
sys.path.insert(0, str(EXTENSION_ROOT))

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner import set_random_seed
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules

from dmm_mbo.balancing import instance_weights
from dmm_mbo.head_loss import apply_loss, balanced_stage_losses, prepare_loss


def total(losses):
    return sum(sum(v) if isinstance(v, (list, tuple)) else v
               for k, v in losses.items() if 'loss' in k)


def compare_losses(left, right):
    for key in ('loss_cls', 'loss_bbox'):
        for a, b in zip(left[key], right[key]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def gradient(module, name):
    values = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert values, 'No gradients: ' + name
    assert all(torch.isfinite(x).all() for x in values), 'Non-finite gradients: ' + name
    value = sum(x.abs().sum().item() for x in values)
    assert value > 0, 'Zero gradients: ' + name
    return value


def check_balancing():
    costs = torch.tensor([[1., 3.], [2., 2.], [3., 1.]], device='cuda', requires_grad=True)
    weights = instance_weights(costs, torch.ones_like(costs, dtype=torch.bool))
    torch.testing.assert_close(weights, 3 * costs.detach().softmax(0))
    assert not weights.requires_grad
    assert weights[2, 0] > weights[1, 0] > weights[0, 0]
    torch.testing.assert_close(weights.sum(0), torch.full((2,), 3., device='cuda'))
    valid = torch.tensor([[True, False], [False, False], [True, False]], device='cuda')
    masked = instance_weights(costs, valid)
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked[:, 1], torch.ones(3, device='cuda'))
    torch.testing.assert_close(masked[valid[:, 0], 0].sum(), costs.new_tensor(2.))
    assert instance_weights(costs[:, :0], valid[:, :0]).shape == (3, 0)


def check_heads(cfg):
    initial_cfg = deepcopy(cfg.model.bbox_head_init)
    initial_cfg.update(train_cfg=cfg.model.train_cfg.init, test_cfg=cfg.model.test_cfg)
    head = MODELS.build(initial_cfg).cuda().train()
    head.init_weights()
    features = [torch.randn(2, 256, s, s, device='cuda', requires_grad=True)
                for s in (16, 8, 4, 2, 1)]
    boxes = [torch.tensor([[32., 32., 32., 32., .2], [80., 80., 32., 32., -.3]], device='cuda'),
             torch.tensor([[64., 64., 32., 32., .4]], device='cuda')]
    gt = [InstanceData(bboxes=RotatedBoxes(b), labels=torch.arange(len(b), device='cuda'))
          for b in boxes]
    metas = [dict(img_shape=(128, 128), pad_shape=(128, 128), scale_factor=(1., 1.)) for _ in gt]
    outputs = head(features)
    prepared = prepare_loss(head, outputs, gt, metas)
    reference = head.loss_by_feat(*outputs, gt, metas)
    equivalent = apply_loss(prepared, torch.ones(3, device='cuda'))
    compare_losses(reference, equivalent)
    ids = torch.cat([x.reshape(-1) for x in prepared.gt_ids])
    assert set(ids[ids >= 0].tolist()) == {0, 1, 2}, 'GT IDs were mixed between images'
    assert prepared.valid.all()
    # A higher cost for the same GT must change only that GT's positive loss.
    modified = replace(prepared, costs=prepared.costs + torch.tensor([0., 3., 0.], device='cuda'))
    parts, diagnostic = balanced_stage_losses([prepared, modified, prepared])
    assert diagnostic['weights'][1, 1] > diagnostic['weights'][0, 1]
    compare_losses(parts[0], apply_loss(prepared, diagnostic['weights'][0]))
    for out in outputs[1]:
        out.retain_grad()
    total(parts[0]).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in features)
    angle_grad = sum(x.grad[:, 4::5].abs().sum().item() for x in outputs[1])
    assert angle_grad > 0, 'Original angle regression did not receive gradient'
    # The same code must handle an empty image/batch without NaN or indexing errors.
    empty = [InstanceData(bboxes=RotatedBoxes(torch.empty(0, 5, device='cuda')),
                          labels=torch.empty(0, dtype=torch.long, device='cuda')) for _ in gt]
    detached = tuple([x.detach() for x in xs] for xs in outputs)
    e = prepare_loss(head, detached, empty, metas)
    eloss, ed = balanced_stage_losses([e, e, e])
    assert ed['weights'].shape == (3, 0) and torch.isfinite(total(eloss[0]))
    compare_losses(eloss[0], head.loss_by_feat(*detached, empty, metas))
    # Refinement uses each branch's own decoded ROIs and the original assigner.
    refine_cfg = deepcopy(cfg.model.bbox_head_refine[0])
    refine_cfg.update(train_cfg=cfg.model.train_cfg.refine[0], test_cfg=cfg.model.test_cfg)
    refine = MODELS.build(refine_cfg).cuda().train()
    refine.init_weights()
    rois = head.filter_bboxes(*detached)
    refined_outputs = refine(refine.feature_refine([x.detach() for x in features], rois))
    rp = prepare_loss(refine, refined_outputs, gt, metas, rois=rois)
    reference = refine.loss_by_feat(*refined_outputs, gt, metas, rois=rois)
    compare_losses(reference, apply_loss(rp, torch.ones(3, device='cuda')))
    result = dict(unit_weights_match_original=True, batch_gt_ids_correct=True,
                  softmax_matches_reference=True, empty_gt_passed=True,
                  init_angle_gradient=angle_grad, refine_unit_weights_match_original=True,
                  assigner=type(head.assigner).__name__, coder=type(head.bbox_coder).__name__,
                  cls_loss=type(head.loss_cls).__name__, bbox_loss=type(head.loss_bbox).__name__)
    del head, refine, prepared, modified, features, outputs, parts, refined_outputs, rp
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_batch(cfg, data_root, batch_size):
    loader_cfg = deepcopy(cfg.train_dataloader)
    loader_cfg.batch_size = batch_size
    loader_cfg.num_workers = 0
    loader_cfg.persistent_workers = False
    loader_cfg.sampler.shuffle = False
    loader_cfg.dataset.data_root = str(data_root)
    loader = Runner.build_dataloader(loader_cfg, seed=20261116)
    batch = next(iter(loader))
    model = MODELS.build(cfg.model).cuda()
    model.init_weights()
    for branch in ('rgb', 'ir'):
        torch.testing.assert_close(model.mbo_init_heads[branch].retina_cls.bias,
                                   model.bbox_head_init.retina_cls.bias)
        for auxiliary, fusion in zip(model.mbo_refine_heads[branch], model.bbox_head_refine):
            torch.testing.assert_close(auxiliary.retina_cls.bias, fusion.retina_cls.bias)
    model.train()
    batch = model.data_preprocessor(batch, training=True)
    shapes, handles = {}, []
    def capture(name):
        def hook(module, args, out):
            shapes[name] = [list(x.shape) for x in out]
        return hook
    for name in ('backbone_vi', 'backbone_ir', 'mtablock', 'fusblock', 'neck'):
        handles.append(getattr(model, name).register_forward_hook(capture(name)))
    for name in ('rgb', 'ir'):
        handles.append(model.mbo_necks[name].register_forward_hook(capture('fpn_' + name)))
    torch.cuda.reset_peak_memory_stats()
    losses = model(**batch, mode='loss')
    loss = total(losses)
    assert torch.isfinite(loss)
    loss.backward()
    grad = {name: gradient(getattr(model, name), name) for name in (
        'backbone_vi', 'backbone_ir', 'mtablock', 'fusblock', 'neck',
        'bbox_head_init', 'bbox_head_refine')}
    for branch in ('rgb', 'ir'):
        for name in ('mbo_necks', 'mbo_init_heads', 'mbo_refine_heads'):
            grad[name + '.' + branch] = gradient(getattr(model, name)[branch], name + '.' + branch)
    for handle in handles:
        handle.remove()
    report = dict(loss=loss.item(), auxiliary_init_matches_baseline=True,
                  shapes=shapes, gradient_abs_sums=grad,
                  input_shapes=[list(batch['inputs'].shape), list(batch['inputs2'].shape)],
                  gt_per_image=[len(x.gt_instances) for x in batch['data_samples']],
                  img_ids=[str(x.metainfo.get('img_id')) for x in batch['data_samples']],
                  losses={k: (sum(v).item() if isinstance(v, list) else v.item()) for k,v in losses.items()},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    # An optimizer step exercises the same original FP32 AdamW configuration.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report['optimizer_step_passed'] = True
    del optimizer, losses, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    # Fail if either normal tensor forward or predict ever calls an auxiliary module.
    def forbidden(*args):
        raise AssertionError('Inference called a training-only auxiliary branch')
    handles = [getattr(model, name)[branch].register_forward_pre_hook(forbidden)
               for name in ('mbo_necks', 'mbo_init_heads', 'mbo_refine_heads')
               for branch in ('rgb', 'ir')]
    with torch.no_grad():
        original = model(**batch, mode='tensor')
        prediction = model.predict(batch['inputs'], batch['inputs2'],
                                   batch['data_samples'], rescale=False)
    for handle in handles:
        handle.remove()
    model.remove_auxiliary_branches()
    with torch.no_grad():
        deployed = model(**batch, mode='tensor')
    for xs, ys in zip(original, deployed):
        for x, y in zip(xs, ys):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert not any(k.startswith('mbo_') for k in model.state_dict())
    assert len(prediction) == batch_size
    report.update(tensor_forward_passed=True, predict_passed=True,
                  inference_auxiliary_calls=0, remove_auxiliary_outputs_identical=True,
                  fusion_parameters=sum(p.numel() for p in model.parameters()),
                  no_auxiliary_state_dict_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT / 'configs/dmm_dcfl_mbo_train.py'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'The full DMM uses CUDA selective scan/AlignConv'
    register_all_modules()
    set_random_seed(20261116)
    cfg = Config.fromfile(args.config)
    assert cfg.model.bbox_head_init.type == 'DCFLS2AHead'
    assert cfg.model.bbox_head_refine[0].type == 'DCFLS2ARefineHead'
    assert cfg.model.train_cfg.init.assigner.type in ('C2FAssigner', 'VectorizedC2FAssigner')
    assert cfg.model.train_cfg.refine[0].assigner.type == cfg.model.train_cfg.init.assigner.type
    report = dict(status='running', config=str(Path(args.config).resolve()),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report['heads'] = check_heads(cfg)
        print('HEAD_CHECKS_PASSED', json.dumps(report['heads']), flush=True)
        report['batch'] = check_batch(cfg, args.data_root, args.batch_size)
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('FULL_BATCH_PASSED', json.dumps(report['batch']), flush=True)


if __name__ == '__main__':
    main()
