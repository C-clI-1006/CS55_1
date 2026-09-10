#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cs55_demo import load_config, run_full_pipeline, save_outputs


def main():
    parser = argparse.ArgumentParser(description='CS55_1_demo')
    parser.add_argument('--config', default='config.sample.json', help='Path to dataset config JSON')
    parser.add_argument('--no-cache', action='store_true', help='Recompute CLIP embeddings instead of using cache')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    cache_path = ROOT / 'cache' / f"{config['dataset_name']}_embeddings.pt"
    result, bundle = run_full_pipeline(
        config,
        use_cache=not args.no_cache,
        cache_path=cache_path,
    )

    result_path, bundle_path = save_outputs(result, bundle, ROOT / 'outputs')

    print('\n=== CS55_1_demo result ===')
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(f'\nSaved result: {result_path}')
    print(f'Saved evidence-chain bundle: {bundle_path}')


if __name__ == '__main__':
    main()
