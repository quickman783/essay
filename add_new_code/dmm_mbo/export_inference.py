"""Strip RGB/IR training branches and optimizer state from a trusted MBO checkpoint."""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    prefixes = ('mbo_necks.', 'mbo_init_heads.', 'mbo_refine_heads.')
    fusion = {k: v for k, v in state.items() if not k.startswith(prefixes)}
    if len(fusion) == len(state):
        parser.error('Checkpoint does not contain MBO auxiliary branches')
    result = dict(state_dict=fusion, meta=checkpoint.get('meta', {}),
                  mbo_export=dict(source=str(args.checkpoint.resolve()),
                                  auxiliary_branches_removed=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print('Saved fusion-only checkpoint:', args.output)


if __name__ == '__main__':
    main()
