"""
Run GEE-Direct WITHOUT dropout on Webb tasks — 5 seeds.
Replicates the original -36.6% catastrophic dependency finding.

Run: python -u run_webb_no_dropout.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

SEEDS = [42, 123, 456, 789, 1024]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'multi_seed', 'webb_no_dropout')


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_seed(seed):
    from dataset import DatasetFactory
    from gee_model import GEEConfig
    from gee_webb_comparison import (
        WebbGEEDirect, TASK_CONFIGS, prepare_data, train_model, evaluate_model,
    )

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=seed)
    set_seed(seed)
    train_data = prepare_data(factory, n_per_task=2000, split='train')
    test_data = prepare_data(factory, n_per_task=500, split='test')

    # Train GEE-Direct with NO dropout
    set_seed(seed)
    model = WebbGEEDirect(cfg).to(device)
    model.task_id_dropout = 0.0  # <-- NO DROPOUT
    log(f"    task_id_dropout = {model.task_id_dropout}")

    t0 = time.time()
    train_model(model, train_data, device, lr=cfg.lr, n_epochs=100)
    log(f"    Trained in {int(time.time()-t0)}s")

    results = {}
    for mode in ('normal', 'none', 'wrong'):
        overall, per_task = evaluate_model(model, test_data, device,
                                           task_id_mode=mode)
        results[mode] = {'overall': overall, 'per_task': per_task}
        task_str = "  ".join(f"{tc['short']}={per_task[tc['name']]:.1%}"
                             for tc in TASK_CONFIGS)
        log(f"    {mode:<8} overall={overall:.1%}  {task_str}")

    return results


def main():
    log("=" * 70)
    log("  GEE-Direct WITHOUT Dropout — Webb Tasks (5 seeds)")
    log("=" * 70)

    all_results = {}
    for seed in SEEDS:
        out_path = os.path.join(OUT_DIR, f'seed_{seed}', 'results.json')
        if os.path.exists(out_path):
            log(f"\n  [seed={seed}] SKIP (results exist)")
            with open(out_path) as f:
                all_results[seed] = json.load(f)
            continue

        log(f"\n  [seed={seed}] Starting...")
        t0 = time.time()
        results = run_seed(seed)
        all_results[seed] = results

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"  [seed={seed}] Done in {int(time.time()-t0)}s")

    # Aggregate
    log(f"\n{'='*70}")
    log("  AGGREGATED RESULTS: GEE-Direct NO DROPOUT")
    log(f"{'='*70}")

    for mode in ('normal', 'none', 'wrong'):
        accs = [all_results[s][mode]['overall'] for s in SEEDS if s in all_results]
        log(f"  {mode:<8} {np.mean(accs):.1%} ± {np.std(accs):.1%}  "
            f"(per seed: {', '.join(f'{a:.1%}' for a in accs)})")

    normal = [all_results[s]['normal']['overall'] for s in SEEDS if s in all_results]
    none_ = [all_results[s]['none']['overall'] for s in SEEDS if s in all_results]
    drops = [n - o for n, o in zip(normal, none_)]
    log(f"\n  Drop (with - without): {np.mean(drops):.1%} ± {np.std(drops):.1%}")
    log(f"  Per seed: {', '.join(f'{d:+.1%}' for d in drops)}")

    log(f"\n{'='*70}")
    log("  Done.")


if __name__ == '__main__':
    main()
