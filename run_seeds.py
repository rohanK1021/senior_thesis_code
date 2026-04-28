"""
Multi-Seed Experiment Runner for GEE Thesis
============================================
Runs all experiments with 5 seeds to produce mean ± std for every result.

Usage:
    python -u run_seeds.py [experiment]

Where experiment is one of:
    webb        — Three-way Webb-faithful comparison (ESBN / GEE-Direct / GEE-Separated)
    wcst        — Wisconsin Card Sorting Task (ESBN / GEE-Direct / GEE-Separated)
    unified     — Unified fixed-length tasks (ESBN / GEE / GEE-h0)
    noise       — Pathway-specific noise on unified GEE model
    wcst_noise  — Pathway-specific noise on unified GEE trained on WCST
    all         — Run all experiments in order

Results are saved to:
    results/multi_seed/{experiment}/seed_{seed}/results.json

Training checkpoints (for noise experiments) are saved to:
    results/multi_seed/{experiment}/seed_{seed}/model.pt

Resumable: if results.json already exists for a (experiment, seed) pair, it is
skipped — re-run to overwrite by deleting the file.
"""

import sys
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Global constants
# =============================================================================

SEEDS = [42, 123, 456, 789, 1024]
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'multi_seed')

NOISE_SIGMAS = [0.0, 0.5, 1.0, 2.0]
NOISE_CONDITIONS = ['global', 'h0_only', 'rcm_only']


# =============================================================================
# Utility
# =============================================================================

def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def result_path(experiment, seed):
    return os.path.join(BASE_DIR, experiment, f'seed_{seed}', 'results.json')


def checkpoint_path(experiment, seed):
    return os.path.join(BASE_DIR, experiment, f'seed_{seed}', 'model.pt')


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(data, path):
    ensure_dir(path)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    log(f"    Saved: {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def elapsed(t0):
    s = int(time.time() - t0)
    return f"{s // 60}m{s % 60:02d}s"


# =============================================================================
# Experiment A: Webb three-way comparison
# =============================================================================

def run_webb(seed):
    """Train ESBN / GEE-Direct / GEE-Separated on Webb tasks, evaluate 3 modes."""
    from dataset import DatasetFactory
    from gee_model import GEEConfig
    from gee_webb_comparison import (
        WebbESBN, WebbGEEDirect, WebbGEESeparated,
        TASK_CONFIGS, prepare_data, train_model, evaluate_model,
    )

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=seed)

    log(f"    Preparing data (seed={seed})...")
    set_seed(seed)
    train_data = prepare_data(factory, n_per_task=2000, split='train')
    test_data = prepare_data(factory, n_per_task=500, split='test')

    model_defs = [
        ('ESBN',          lambda: WebbESBN(cfg)),
        ('GEE-Direct',    lambda: WebbGEEDirect(cfg)),
        ('GEE-Separated', lambda: WebbGEESeparated(cfg)),
    ]

    results = {}
    for label, make_model in model_defs:
        log(f"    Training {label}...")
        set_seed(seed)
        model = make_model().to(device)
        t0 = time.time()
        train_model(model, train_data, device, lr=cfg.lr, n_epochs=100)
        log(f"    {label} trained in {elapsed(t0)}")

        model_results = {}
        for mode in ('normal', 'none', 'wrong'):
            overall, per_task = evaluate_model(model, test_data, device,
                                               task_id_mode=mode)
            model_results[mode] = {
                'overall': overall,
                'per_task': per_task,
            }
            log(f"      {mode:<8} overall={overall:.3f}  " +
                "  ".join(f"{tc['short']}={per_task[tc['name']]:.3f}"
                          for tc in TASK_CONFIGS))

        results[label] = model_results

    return results


# =============================================================================
# Experiment B: WCST
# =============================================================================

def run_wcst(seed):
    """Train WCST_ESBN / WCST_GEEDirect / WCST_GEESeparated, evaluate 3 modes."""
    from dataset import EntityPool
    from gee_model import GEEConfig
    from gee_wcst import (
        WCST_ESBN, WCST_GEEDirect, WCST_GEESeparated,
        train_wcst, run_evaluation,
    )

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    train_entities = pool.get('train')
    test_entities = pool.get('test')

    model_defs = [
        ('ESBN',          WCST_ESBN),
        ('GEE-Direct',    WCST_GEEDirect),
        ('GEE-Separated', WCST_GEESeparated),
    ]

    results = {}
    for label, ModelClass in model_defs:
        log(f"    Training {label}...")
        set_seed(seed)
        model = ModelClass(cfg).to(device)
        t0 = time.time()
        train_wcst(model, train_entities, device, lr=cfg.lr,
                   n_epochs=50, n_per_rule=1000, seed=seed)
        log(f"    {label} trained in {elapsed(t0)}")

        model_results = {}
        for mode in ('normal', 'none', 'wrong'):
            r = run_evaluation(model, test_entities, device,
                               n_sessions=50, task_id_mode=mode)
            model_results[mode] = {
                'accuracy_mean':         r['accuracy_mean'],
                'accuracy_std':          r['accuracy_std'],
                'persev_errors_mean':    r['persev_errors_mean'],
                'persev_errors_std':     r['persev_errors_std'],
                'non_persev_errors_mean': r['non_persev_errors_mean'],
                'non_persev_errors_std':  r['non_persev_errors_std'],
                'completions_mean':      r['completions_mean'],
                'completions_std':       r['completions_std'],
            }
            log(f"      {mode:<8} acc={r['accuracy_mean']:.3f}  "
                f"persev={r['persev_errors_mean']:.2f}  "
                f"completions={r['completions_mean']:.2f}")

        results[label] = model_results

    return results


# =============================================================================
# Experiment C: Unified fixed-length tasks
# =============================================================================

def run_unified(seed):
    """Train ESBN / GEE / GEE-h0 on unified tasks, evaluate 3 modes."""
    from dataset import EntityPool
    from gee_model import GEEConfig
    from gee_unified import GEE
    from unified_tasks import UnifiedTaskFactory, N_TASKS, TASK_NAMES
    from gee_main import (
        train_model as train_unified,
        evaluate_model as evaluate_unified,
    )

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    factory = UnifiedTaskFactory(pool.get('train'), pool.get('test'), seed=seed)

    model_configs = [
        ('ESBN',   {'use_h0': False, 'use_rcm': False, 'dropout_p': 0.0}),
        ('GEE',    {'use_h0': True,  'use_rcm': True,  'dropout_p': 0.3}),
        ('GEE-h0', {'use_h0': True,  'use_rcm': False, 'dropout_p': 0.3}),
    ]

    results = {}
    for name, kwargs in model_configs:
        log(f"    Training {name}...")
        set_seed(seed)
        model = GEE(cfg, n_tasks=N_TASKS, **kwargs).to(device)
        t0 = time.time()
        train_unified(model, factory, device, n_epochs=50,
                      batches_per_task=100, batch_size=64, lr=1e-4)
        log(f"    {name} trained in {elapsed(t0)}")

        model_results = {}
        for mode in ('normal', 'none', 'wrong'):
            overall, per_task = evaluate_unified(model, factory, device,
                                                 task_id_mode=mode)
            model_results[mode] = {
                'overall': overall,
                'per_task': per_task,
            }
            log(f"      {mode:<8} overall={overall:.3f}  " +
                "  ".join(f"{t}={per_task[t]:.3f}" for t in TASK_NAMES))

        results[name] = model_results

    return results


# =============================================================================
# Experiment D: Noise injection on unified GEE
# =============================================================================

def _get_or_train_unified_gee(seed, device, cfg, factory, ckpt_path):
    """Load checkpoint if present, otherwise train and save."""
    from gee_unified import GEE
    from unified_tasks import N_TASKS
    from gee_main import train_model as train_unified

    model = GEE(cfg, n_tasks=N_TASKS, use_h0=True, use_rcm=True,
                dropout_p=0.3).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log(f"    Loaded checkpoint: {ckpt_path}")
    else:
        log(f"    No checkpoint found — training from scratch...")
        set_seed(seed)
        t0 = time.time()
        train_unified(model, factory, device, n_epochs=50,
                      batches_per_task=100, batch_size=64, lr=1e-4)
        log(f"    Trained in {elapsed(t0)}")
        ensure_dir(ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        log(f"    Checkpoint saved: {ckpt_path}")

    return model


def run_noise(seed):
    """Evaluate unified GEE under pathway-specific noise."""
    from dataset import EntityPool
    from gee_model import GEEConfig
    from unified_tasks import UnifiedTaskFactory, N_TASKS
    from gee_noise_experiment import evaluate_with_noise, compute_consistency

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    factory = UnifiedTaskFactory(pool.get('train'), pool.get('test'), seed=seed)

    ckpt = checkpoint_path('noise', seed)
    model = _get_or_train_unified_gee(seed, device, cfg, factory, ckpt)
    model.eval()

    results = {}
    for noise_type in NOISE_CONDITIONS:
        results[noise_type] = {}
        for sigma in NOISE_SIGMAS:
            overall, per_task = evaluate_with_noise(model, factory, device,
                                                    sigma=sigma,
                                                    noise_type=noise_type)
            consistency = compute_consistency(model, factory, device,
                                             sigma=sigma,
                                             noise_type=noise_type)
            results[noise_type][str(sigma)] = {
                'overall': overall,
                'per_task': per_task,
                'consistency': consistency,
            }
            log(f"      {noise_type:<10} sigma={sigma:<4}  "
                f"overall={overall:.3f}  consistency={consistency:.3f}")

    return results


# =============================================================================
# Experiment E: WCST noise (unified GEE trained on WCST)
# =============================================================================

def _get_or_train_wcst_unified(seed, device, cfg, train_entities, ckpt_path):
    """Load checkpoint if present, otherwise train unified GEE on WCST."""
    from gee_unified import GEE
    from gee_wcst_noise import train_wcst_unified

    model = GEE(cfg, n_tasks=4, use_h0=True, use_rcm=True,
                dropout_p=0.3).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log(f"    Loaded checkpoint: {ckpt_path}")
    else:
        log(f"    No checkpoint found — training from scratch...")
        set_seed(seed)
        t0 = time.time()
        train_wcst_unified(model, train_entities, device, lr=5e-4,
                           n_epochs=50, n_per_rule=1000, seed=seed)
        log(f"    Trained in {elapsed(t0)}")
        ensure_dir(ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        log(f"    Checkpoint saved: {ckpt_path}")

    return model


def run_wcst_noise(seed):
    """Evaluate WCST unified GEE under pathway-specific noise."""
    from dataset import EntityPool
    from gee_model import GEEConfig
    from gee_wcst_noise import evaluate_wcst_session

    # Constants mirrored from gee_wcst_noise.py
    N_SESSIONS = 50
    MAX_COMPLETIONS = 6
    CRITERION_TRIALS = 6
    MAX_TRIALS = 200
    N_RULES = 4
    RULE_ATTR_IDX = [1, 0, 2, 3]

    set_seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    train_entities = pool.get('train')
    test_entities = pool.get('test')

    ckpt = checkpoint_path('wcst_noise', seed)
    model = _get_or_train_wcst_unified(seed, device, cfg, train_entities, ckpt)
    model.eval()

    results = {}
    for noise_type in NOISE_CONDITIONS:
        results[noise_type] = {}
        for sigma in NOISE_SIGMAS:
            # Run N_SESSIONS sessions and aggregate
            all_accs = []
            all_persev = []
            all_completions = []

            for s_idx in range(N_SESSIONS):
                sess = evaluate_wcst_session(
                    model, test_entities, device,
                    noise_type=noise_type,
                    noise_sigma=sigma,
                    seed=s_idx * 100 + seed,
                )
                n_correct = sum(t['correct'] for t in sess['trials'])
                n_total = max(len(sess['trials']), 1)
                n_persev = sum(t['is_perseverative'] for t in sess['trials'])

                all_accs.append(n_correct / n_total)
                all_persev.append(n_persev)
                all_completions.append(sess['rules_completed'])

            results[noise_type][str(sigma)] = {
                'accuracy_mean':      float(np.mean(all_accs)),
                'accuracy_std':       float(np.std(all_accs)),
                'persev_errors_mean': float(np.mean(all_persev)),
                'persev_errors_std':  float(np.std(all_persev)),
                'completions_mean':   float(np.mean(all_completions)),
                'completions_std':    float(np.std(all_completions)),
            }
            log(f"      {noise_type:<10} sigma={sigma:<4}  "
                f"acc={np.mean(all_accs):.3f}  "
                f"persev={np.mean(all_persev):.2f}  "
                f"completions={np.mean(all_completions):.2f}")

    return results


# =============================================================================
# Per-experiment dispatcher
# =============================================================================

EXPERIMENT_RUNNERS = {
    'webb':       run_webb,
    'wcst':       run_wcst,
    'unified':    run_unified,
    'noise':      run_noise,
    'wcst_noise': run_wcst_noise,
}

EXPERIMENT_ORDER = ['webb', 'wcst', 'unified', 'noise', 'wcst_noise']


def run_experiment(experiment):
    """Run a single experiment across all seeds, resuming completed ones."""
    runner = EXPERIMENT_RUNNERS[experiment]
    log()
    log("=" * 70)
    log(f"  Experiment: {experiment.upper()}")
    log(f"  Seeds: {SEEDS}")
    log("=" * 70)

    for seed in SEEDS:
        rpath = result_path(experiment, seed)

        if os.path.exists(rpath):
            log(f"\n  [seed={seed}] SKIP — results already exist: {rpath}")
            continue

        log(f"\n  {'─'*68}")
        log(f"  [seed={seed}] Starting...")
        log(f"  {'─'*68}")
        t0 = time.time()

        results = runner(seed)

        results['_meta'] = {
            'experiment': experiment,
            'seed': seed,
            'elapsed_seconds': int(time.time() - t0),
        }

        save_json(results, rpath)
        log(f"  [seed={seed}] Done in {elapsed(t0)}")

    log()
    log(f"  Experiment {experiment} complete for all seeds.")
    log()


# =============================================================================
# Aggregation helpers (printed after all seeds finish)
# =============================================================================

def aggregate_and_print(experiment):
    """Load all seed results for an experiment and print mean ± std tables."""
    log()
    log("=" * 70)
    log(f"  AGGREGATED RESULTS: {experiment.upper()}")
    log("=" * 70)

    all_results = {}
    missing = []
    for seed in SEEDS:
        rpath = result_path(experiment, seed)
        if os.path.exists(rpath):
            all_results[seed] = load_json(rpath)
        else:
            missing.append(seed)

    if missing:
        log(f"  WARNING: missing seeds {missing} — aggregation is partial")
    if not all_results:
        log("  No results to aggregate.")
        return

    if experiment == 'webb':
        _aggregate_webb(all_results)
    elif experiment == 'wcst':
        _aggregate_wcst(all_results)
    elif experiment == 'unified':
        _aggregate_unified(all_results)
    elif experiment == 'noise':
        _aggregate_noise(all_results)
    elif experiment == 'wcst_noise':
        _aggregate_wcst_noise(all_results)


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def _aggregate_webb(all_results):
    from gee_webb_comparison import TASK_CONFIGS
    model_names = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    modes = ['normal', 'none', 'wrong']

    log()
    log("  Overall accuracy: mean ± std")
    log(f"  {'Model':<18} {'Normal':>12} {'No ID':>12} {'Wrong':>12} {'Drop':>12}")
    log(f"  {'─'*60}")

    for model in model_names:
        row = f"  {model:<18}"
        norms = [all_results[s][model]['normal']['overall'] for s in all_results]
        noids = [all_results[s][model]['none']['overall']   for s in all_results]
        wrongs= [all_results[s][model]['wrong']['overall']  for s in all_results]
        drops = [n - i for n, i in zip(norms, noids)]

        for vals in (norms, noids, wrongs, drops):
            m, sd = _mean_std(vals)
            row += f"  {m:+.3f}±{sd:.3f}" if vals is drops else f"  {m:.3f}±{sd:.3f}"
        log(row)

    log()
    log("  Per-task accuracy (normal mode): mean ± std")
    log(f"  {'Task':<24}" + "".join(f" {n:>18}" for n in model_names))
    log(f"  {'─'*(24 + 19 * len(model_names))}")

    for tc in TASK_CONFIGS:
        tname = tc['name']
        row = f"  {tname:<24}"
        for model in model_names:
            vals = [all_results[s][model]['normal']['per_task'][tname]
                    for s in all_results]
            m, sd = _mean_std(vals)
            row += f"  {m:.3f}±{sd:.3f}      "
        log(row)


def _aggregate_wcst(all_results):
    model_names = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    modes = ['normal', 'none', 'wrong']

    for metric, label in [('accuracy_mean', 'Accuracy'),
                           ('persev_errors_mean', 'Persev errors'),
                           ('completions_mean', 'Completions')]:
        log()
        log(f"  {label}: mean ± std across seeds")
        log(f"  {'Model':<18} {'Normal':>14} {'No ID':>14} {'Wrong':>14}")
        log(f"  {'─'*62}")
        for model in model_names:
            row = f"  {model:<18}"
            for mode in modes:
                vals = [all_results[s][model][mode][metric] for s in all_results]
                m, sd = _mean_std(vals)
                row += f"  {m:.3f}±{sd:.3f}    "
            log(row)


def _aggregate_unified(all_results):
    from unified_tasks import TASK_NAMES
    model_names = ['ESBN', 'GEE', 'GEE-h0']
    modes = ['normal', 'none', 'wrong']

    log()
    log("  Overall accuracy: mean ± std")
    log(f"  {'Model':<12} {'Normal':>12} {'No ID':>12} {'Wrong':>12} {'Drop':>12}")
    log(f"  {'─'*56}")
    for model in model_names:
        norms = [all_results[s][model]['normal']['overall'] for s in all_results]
        noids = [all_results[s][model]['none']['overall']   for s in all_results]
        wrongs= [all_results[s][model]['wrong']['overall']  for s in all_results]
        drops = [n - i for n, i in zip(norms, noids)]
        row = f"  {model:<12}"
        for vals in (norms, noids, wrongs, drops):
            m, sd = _mean_std(vals)
            row += f"  {m:+.3f}±{sd:.3f}" if vals is drops else f"  {m:.3f}±{sd:.3f}"
        log(row)

    log()
    log("  Per-task accuracy (normal mode): mean ± std")
    log(f"  {'Task':<20}" + "".join(f" {n:>14}" for n in model_names))
    log(f"  {'─'*(20 + 15 * len(model_names))}")
    for tname in TASK_NAMES:
        row = f"  {tname:<20}"
        for model in model_names:
            vals = [all_results[s][model]['normal']['per_task'][tname]
                    for s in all_results]
            m, sd = _mean_std(vals)
            row += f"  {m:.3f}±{sd:.3f}  "
        log(row)


def _aggregate_noise(all_results):
    from unified_tasks import TASK_NAMES

    for noise_type in NOISE_CONDITIONS:
        log()
        log(f"  Noise type: {noise_type}")
        log(f"  {'Sigma':<8} {'Overall':>12} {'Consistency':>14}")
        log(f"  {'─'*36}")
        for sigma in NOISE_SIGMAS:
            key = str(sigma)
            ovals = [all_results[s][noise_type][key]['overall']     for s in all_results]
            cvals = [all_results[s][noise_type][key]['consistency'] for s in all_results]
            om, osd = _mean_std(ovals)
            cm, csd = _mean_std(cvals)
            log(f"  {sigma:<8}  {om:.3f}±{osd:.3f}    {cm:.3f}±{csd:.3f}")


def _aggregate_wcst_noise(all_results):
    for noise_type in NOISE_CONDITIONS:
        log()
        log(f"  Noise type: {noise_type}")
        log(f"  {'Sigma':<8} {'Accuracy':>12} {'PersevErr':>12} {'Completions':>14}")
        log(f"  {'─'*50}")
        for sigma in NOISE_SIGMAS:
            key = str(sigma)
            avals = [all_results[s][noise_type][key]['accuracy_mean']      for s in all_results]
            pvals = [all_results[s][noise_type][key]['persev_errors_mean'] for s in all_results]
            cvals = [all_results[s][noise_type][key]['completions_mean']   for s in all_results]
            am, asd = _mean_std(avals)
            pm, psd = _mean_std(pvals)
            cm, csd = _mean_std(cvals)
            log(f"  {sigma:<8}  {am:.3f}±{asd:.3f}    {pm:.2f}±{psd:.2f}    {cm:.2f}±{csd:.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        log("Usage: python -u run_seeds.py [experiment]")
        log("       experiment: webb | wcst | unified | noise | wcst_noise | all")
        sys.exit(1)

    experiment_arg = sys.argv[1].lower()

    if experiment_arg == 'all':
        experiments = EXPERIMENT_ORDER
    elif experiment_arg in EXPERIMENT_RUNNERS:
        experiments = [experiment_arg]
    else:
        log(f"Unknown experiment: {experiment_arg!r}")
        log(f"Valid choices: {list(EXPERIMENT_RUNNERS.keys()) + ['all']}")
        sys.exit(1)

    total_t0 = time.time()
    log()
    log("=" * 70)
    log("  GEE Multi-Seed Experiment Runner")
    log(f"  Seeds: {SEEDS}")
    log(f"  Experiments: {experiments}")
    log(f"  Results root: {BASE_DIR}")
    log("=" * 70)

    for experiment in experiments:
        run_experiment(experiment)
        aggregate_and_print(experiment)

    log()
    log("=" * 70)
    log(f"  All done. Total time: {elapsed(total_t0)}")
    log("=" * 70)


if __name__ == '__main__':
    main()
