"""MS-DETR's soft instance reweighting, independent of any assigner.

Reference: YinghuiXing/MS-DETR, matcher.py, instance_reweight/plus_three.
Positive cost softmax emphasizes the branch with the larger instance cost.
The optional temperature is 1 by default, as in the reference implementation.
"""
import torch


@torch.no_grad()
def instance_weights(costs, valid, temperature=1.0, enabled=True):
    """Return detached [branches, GTs] weights; missing matches stay neutral.

    With three valid branches this is exactly 3 * softmax(costs, dim=0).
    If only k branches assigned a positive to a GT, normalize those to sum k;
    the other branches receive 1 (and have no positive loss to reweight).
    """
    if temperature <= 0:
        raise ValueError('MBO temperature must be positive')
    if costs.ndim != 2 or costs.shape != valid.shape:
        raise ValueError('Expected equally shaped [branches, GTs] inputs')
    if (valid & ~torch.isfinite(costs)).any():
        raise FloatingPointError('Non-finite matched MBO cost')
    result = torch.ones_like(costs, dtype=torch.float32)
    if not enabled or costs.shape[1] == 0:
        return result
    valid = valid.bool()
    count = valid.sum(dim=0)
    # Avoid softmax([-inf, -inf, -inf]) for completely unmatched GTs.
    logits = (costs.float() / temperature).masked_fill(~valid, -torch.inf)
    logits[:, count == 0] = 0
    balanced = logits.softmax(dim=0) * count.unsqueeze(0)
    return torch.where(valid, balanced, result)
