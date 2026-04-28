"""
WCST Experiment for GEE Thesis
===============================
Wisconsin Card Sorting Task: tests context-dependent rule switching.

Every trial has the SAME structure (seq_len=2: reference + test entity).
The only way to know which attribute to sort by is from context (task_id).
This removes the structural disambiguation crutch of the Webb tasks
(where different sequence lengths partially identify the task).

Three phases:
  1. Fine-tune models on WCST (attribute-specific Same/Different)
  2. Evaluate with/without/noisy task_id + perseveration analysis
  3. Noise injection experiment (computational analogue of dopamine disruption)

Run: python -u gee_wcst.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import ALL_ENTITIES, Entity, ATTR_SIZES, EntityPool
from esbn_pure import ContextNorm, AttributeAwareEncoder
from gee_model import GEEConfig, RCMContextModule


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# WCST Configuration
# =============================================================================

WCST_RULES = ['color', 'shape', 'size', 'texture']
RULE_ATTR_IDX = [1, 0, 2, 3]  # maps rule to Entity attribute index (color=1, shape=0, etc.)
N_RULES = 4
CRITERION_TRIALS = 6     # consecutive correct to trigger rule switch
MAX_COMPLETIONS = 6       # max rule switches per session
MAX_TRIALS_PER_SESSION = 200  # safety limit


# =============================================================================
# WCST Trial Generation
# =============================================================================

class WCSTSession:
    """Generates and manages a WCST session."""

    def __init__(self, entities, seed=None):
        self.entities = entities
        self.rng = np.random.RandomState(seed)

    def generate_trial(self, active_rule, is_match):
        """Generate one WCST trial: (reference, test, is_match)."""
        attr_idx = RULE_ATTR_IDX[active_rule]
        ref = self.entities[self.rng.randint(len(self.entities))]

        if is_match:
            matches = [e for e in self.entities if e[attr_idx] == ref[attr_idx] and e != ref]
            if not matches:
                matches = [e for e in self.entities if e[attr_idx] == ref[attr_idx]]
            test = matches[self.rng.randint(len(matches))]
        else:
            non_matches = [e for e in self.entities if e[attr_idx] != ref[attr_idx]]
            test = non_matches[self.rng.randint(len(non_matches))]

        return ref, test, is_match

    def run_session_trials(self):
        """Generate a full WCST session with rule switches.

        Returns list of (ref, test, is_match, active_rule, trial_idx, is_post_switch).
        """
        active_rule = 0
        consecutive_correct = 0
        rules_completed = 0
        trials = []
        rules_sequence = list(range(N_RULES))
        self.rng.shuffle(rules_sequence)
        rule_order_idx = 0

        active_rule = rules_sequence[rule_order_idx]

        for trial_idx in range(MAX_TRIALS_PER_SESSION):
            if rules_completed >= MAX_COMPLETIONS:
                break

            is_match = self.rng.random() < 0.5
            ref, test, is_match = self.generate_trial(active_rule, is_match)

            # is_post_switch: True for first few trials after a rule change
            is_post_switch = (consecutive_correct == 0 and trial_idx > 0
                              and len(trials) > 0 and trials[-1][3] != active_rule)

            trials.append((ref, test, is_match, active_rule, trial_idx, is_post_switch))

            yield ref, test, is_match, active_rule, trial_idx

            # Placeholder: correct will be set by the caller via send()
            # For generation, we just produce trials. The evaluator handles correctness.

    def generate_all_trials(self):
        """Pre-generate a full session for stateless evaluation."""
        active_rule = 0
        consecutive_correct = 0
        rules_completed = 0
        rules_used = [0]
        rule_switch_points = []

        # Randomize first rule
        active_rule = self.rng.randint(N_RULES)
        rules_used = [active_rule]

        trials = []
        for trial_idx in range(MAX_TRIALS_PER_SESSION):
            if rules_completed >= MAX_COMPLETIONS:
                break

            is_match = self.rng.random() < 0.5
            ref, test, _ = self.generate_trial(active_rule, is_match)
            trials.append({
                'ref': ref, 'test': test, 'is_match': is_match,
                'active_rule': active_rule, 'trial_idx': trial_idx,
            })

        return trials, rules_used


# =============================================================================
# Models — same architecture as gee_webb_comparison.py but WCST-specific
# =============================================================================

class WCST_ESBN(nn.Module):
    """Webb-faithful ESBN for WCST. Ignores task_id."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, 2)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq, task_id=None, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)
        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
        for t in range(seq_len + 1):
            z_t = torch.zeros(batch, 1, cfg.value_dim, device=device) if t == seq_len else z_seq[:, t, :].unsqueeze(1)
            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred = self.y_out(gru_out).squeeze(1)
            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)
            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)
        return y_pred


class WCST_GEEDirect(nn.Module):
    """GEE-Direct for WCST. task_id in keys + RCM at output. Dropout p=0.5."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, 2)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))
        self.task_to_key = nn.Linear(N_RULES, cfg.key_dim, bias=False)
        rcm_input_dim = cfg.input_dim + N_RULES
        self.rcm = RCMContextModule(rcm_input_dim, 2, cfg.context_dim)
        self.ctx_to_out = nn.Linear(cfg.context_dim, 2, bias=False)
        self.task_id_dropout = 0.5

    def forward(self, x_seq, task_id=None, device=None, noise_sigma=0.0):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, N_RULES, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        if noise_sigma > 0:
            task_onehot = task_onehot + torch.randn_like(task_onehot) * noise_sigma
        task_key_bias = self.task_to_key(task_onehot)

        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, 2, device=device)

        for t in range(seq_len + 1):
            z_t = torch.zeros(batch, 1, cfg.value_dim, device=device) if t == seq_len else z_seq[:, t, :].unsqueeze(1)
            x_raw = torch.zeros(batch, cfg.input_dim, device=device) if t == seq_len else x_seq[:, t, :]
            rcm_in = torch.cat([x_raw, task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)
            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out)) + task_key_bias.unsqueeze(1)
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred = self.y_out(gru_out).squeeze(1) + self.ctx_to_out(c_t)
            y_prev = torch.softmax(y_pred.detach(), dim=-1)
            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)
            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)
        return y_pred


class WCST_GEESeparated(nn.Module):
    """GEE-Separated for WCST. Context at h_0 only. Dropout p=0.5."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, 2)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))
        self.h0_proj = nn.Linear(N_RULES, cfg.hidden_dim)
        self.task_id_dropout = 0.5

    def forward(self, x_seq, task_id=None, device=None, noise_sigma=0.0):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, N_RULES, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        if noise_sigma > 0:
            task_onehot = task_onehot + torch.randn_like(task_onehot) * noise_sigma
        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)

        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
        for t in range(seq_len + 1):
            z_t = torch.zeros(batch, 1, cfg.value_dim, device=device) if t == seq_len else z_seq[:, t, :].unsqueeze(1)
            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred = self.y_out(gru_out).squeeze(1)
            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)
            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)
        return y_pred


# =============================================================================
# WCST Data Generation for Training
# =============================================================================

def generate_wcst_data(entities, n_per_rule, rng):
    """Generate WCST training data: attribute-specific Same/Different."""
    inputs, labels, task_ids = [], [], []
    for rule in range(N_RULES):
        attr_idx = RULE_ATTR_IDX[rule]
        for _ in range(n_per_rule):
            is_match = rng.random() < 0.5
            ref = entities[rng.randint(len(entities))]
            if is_match:
                pool = [e for e in entities if e[attr_idx] == ref[attr_idx] and e != ref]
                if not pool:
                    pool = [e for e in entities if e[attr_idx] == ref[attr_idx]]
                test = pool[rng.randint(len(pool))]
                label = np.array([1., 0.], dtype=np.float32)
            else:
                pool = [e for e in entities if e[attr_idx] != ref[attr_idx]]
                test = pool[rng.randint(len(pool))]
                label = np.array([0., 1.], dtype=np.float32)

            seq = np.stack([ref.one_hot(), test.one_hot()])
            inputs.append(seq)
            labels.append(label)
            task_ids.append(rule)

    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    task_ids = np.array(task_ids, dtype=np.int64)
    perm = rng.permutation(len(inputs))
    return inputs[perm], labels[perm], task_ids[perm]


# =============================================================================
# Training
# =============================================================================

def train_wcst(model, train_entities, device, lr=5e-4, n_epochs=50,
               n_per_rule=1000, seed=42):
    rng = np.random.RandomState(seed)
    inputs, labels, task_ids = generate_wcst_data(train_entities, n_per_rule, rng)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(inputs), torch.tensor(labels), torch.tensor(task_ids))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b, tid_b in loader:
            x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)

            batch_logits = torch.zeros(x_b.shape[0], 2, device=device)
            for tid in tid_b.unique():
                mask = tid_b == tid
                batch_logits[mask] = model(x_b[mask], task_id=tid.item(), device=device)

            loss = F.cross_entropy(batch_logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x_b.shape[0]
            correct += (batch_logits.argmax(-1) == y_b.argmax(-1)).sum().item()
            total += x_b.shape[0]

        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")

    return model


# =============================================================================
# WCST Evaluation — Stateless (each trial independent)
# =============================================================================

@torch.no_grad()
def evaluate_wcst_session(model, test_entities, device, task_id_mode='normal',
                          noise_sigma=0.0, seed=None):
    """Run one WCST session. Returns detailed trial-by-trial results."""
    model.eval()
    rng = np.random.RandomState(seed)

    active_rule = rng.randint(N_RULES)
    consecutive_correct = 0
    rules_completed = 0
    trial_results = []
    prev_rule = active_rule
    switch_points = []

    for trial_idx in range(MAX_TRIALS_PER_SESSION):
        if rules_completed >= MAX_COMPLETIONS:
            break

        is_match = rng.random() < 0.5
        attr_idx = RULE_ATTR_IDX[active_rule]
        ref = test_entities[rng.randint(len(test_entities))]

        if is_match:
            pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx] and e != ref]
            if not pool:
                pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx]]
            test = pool[rng.randint(len(pool))]
        else:
            pool = [e for e in test_entities if e[attr_idx] != ref[attr_idx]]
            test = pool[rng.randint(len(pool))]

        x = torch.tensor(np.stack([ref.one_hot(), test.one_hot()]),
                         dtype=torch.float32, device=device).unsqueeze(0)

        # Get task_id based on mode
        if task_id_mode == 'normal':
            tid = active_rule
        elif task_id_mode == 'none':
            tid = None
        elif task_id_mode == 'wrong':
            tid = (active_rule + rng.randint(1, N_RULES)) % N_RULES
        else:
            tid = active_rule

        kwargs = {'task_id': tid, 'device': device}
        if noise_sigma > 0 and hasattr(model, 'task_id_dropout'):
            kwargs['noise_sigma'] = noise_sigma

        logits = model(x, **kwargs)
        pred = logits.argmax(-1).item()
        true_label = 0 if is_match else 1
        correct = (pred == true_label)

        # Check if response matches OLD rule (perseverative error)
        is_perseverative = False
        if not correct and prev_rule != active_rule:
            old_attr_idx = RULE_ATTR_IDX[prev_rule]
            old_match = (ref[old_attr_idx] == test[old_attr_idx])
            old_correct_pred = 0 if old_match else 1
            if pred == old_correct_pred:
                is_perseverative = True

        trial_results.append({
            'trial_idx': trial_idx,
            'active_rule': active_rule,
            'correct': correct,
            'is_perseverative': is_perseverative,
            'is_post_switch': (prev_rule != active_rule and consecutive_correct < CRITERION_TRIALS),
        })

        if correct:
            consecutive_correct += 1
            if consecutive_correct >= CRITERION_TRIALS:
                rules_completed += 1
                prev_rule = active_rule
                # Switch to a new rule
                new_rule = (active_rule + rng.randint(1, N_RULES)) % N_RULES
                switch_points.append(trial_idx + 1)
                active_rule = new_rule
                consecutive_correct = 0
        else:
            consecutive_correct = 0

    return {
        'trials': trial_results,
        'rules_completed': rules_completed,
        'switch_points': switch_points,
        'n_trials': len(trial_results),
    }


def run_evaluation(model, test_entities, device, n_sessions=50,
                   task_id_mode='normal', noise_sigma=0.0):
    """Run multiple WCST sessions and aggregate results."""
    all_sessions = []
    for s in range(n_sessions):
        result = evaluate_wcst_session(
            model, test_entities, device,
            task_id_mode=task_id_mode, noise_sigma=noise_sigma, seed=s * 100)
        all_sessions.append(result)

    # Aggregate
    accuracies = []
    persev_errors = []
    non_persev_errors = []
    completions = []
    trials_to_criterion = []

    for sess in all_sessions:
        n_correct = sum(t['correct'] for t in sess['trials'])
        n_total = len(sess['trials'])
        accuracies.append(n_correct / max(n_total, 1))
        completions.append(sess['rules_completed'])
        persev = sum(t['is_perseverative'] for t in sess['trials'])
        non_persev = sum(not t['correct'] and not t['is_perseverative'] for t in sess['trials'])
        persev_errors.append(persev)
        non_persev_errors.append(non_persev)

    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'persev_errors_mean': np.mean(persev_errors),
        'persev_errors_std': np.std(persev_errors),
        'non_persev_errors_mean': np.mean(non_persev_errors),
        'non_persev_errors_std': np.std(non_persev_errors),
        'completions_mean': np.mean(completions),
        'completions_std': np.std(completions),
        'all_sessions': all_sessions,
    }


def compute_adaptation_curve(all_sessions, window=25):
    """Compute accuracy relative to rule switch points."""
    aligned = {i: [] for i in range(-5, window + 1)}
    for sess in all_sessions:
        for sp in sess['switch_points']:
            for t in sess['trials']:
                rel = t['trial_idx'] - sp
                if -5 <= rel <= window:
                    aligned[rel].append(1.0 if t['correct'] else 0.0)
    positions = sorted(aligned.keys())
    means = [np.mean(aligned[p]) if aligned[p] else np.nan for p in positions]
    sems = [np.std(aligned[p]) / max(np.sqrt(len(aligned[p])), 1) if aligned[p] else np.nan
            for p in positions]
    return positions, means, sems


# =============================================================================
# Noise Injection Experiment
# =============================================================================

def run_noise_experiment(models, model_names, test_entities, device,
                         sigma_values, n_sessions=20):
    """Run noise sweep for all models."""
    results = {name: {} for name in model_names}
    for name, model in zip(model_names, models):
        log(f"  Noise sweep: {name}")
        for sigma in sigma_values:
            r = run_evaluation(model, test_entities, device,
                               n_sessions=n_sessions,
                               task_id_mode='normal', noise_sigma=sigma)
            results[name][sigma] = r
            log(f"    σ={sigma:.1f}: acc={r['accuracy_mean']:.1%} "
                f"persev={r['persev_errors_mean']:.1f}")
    return results


# =============================================================================
# Figures
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.dpi': 200,
})
MODEL_COLORS = ['#5B9BD5', '#9B59B6', '#2ECC71']


def plot_overview(eval_results, model_names, save_dir='figures/wcst'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Accuracy
    ax = axes[0]
    accs = [eval_results[n]['accuracy_mean'] for n in model_names]
    errs = [eval_results[n]['accuracy_std'] for n in model_names]
    bars = ax.bar(range(len(model_names)), accs, yerr=errs, capsize=4,
                  color=MODEL_COLORS, edgecolor='white', linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('A.  Overall accuracy', fontweight='bold', loc='left')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel B: Perseveration
    ax = axes[1]
    persev = [eval_results[n]['persev_errors_mean'] for n in model_names]
    errs = [eval_results[n]['persev_errors_std'] for n in model_names]
    bars = ax.bar(range(len(model_names)), persev, yerr=errs, capsize=4,
                  color=MODEL_COLORS, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, persev):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Perseverative errors')
    ax.set_title('B.  Perseveration (lower = better)', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel C: Rule completions
    ax = axes[2]
    comps = [eval_results[n]['completions_mean'] for n in model_names]
    errs = [eval_results[n]['completions_std'] for n in model_names]
    bars = ax.bar(range(len(model_names)), comps, yerr=errs, capsize=4,
                  color=MODEL_COLORS, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Rules completed')
    ax.set_title('C.  Rule completion rate', fontweight='bold', loc='left')
    ax.set_ylim(0, MAX_COMPLETIONS + 1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.suptitle('WCST Performance', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig_wcst_overview.png', bbox_inches='tight')
    log(f"  Saved: {save_dir}/fig_wcst_overview.png")
    plt.close()


def plot_adaptation(eval_results, model_names, save_dir='figures/wcst'):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(model_names):
        sessions = eval_results[name]['all_sessions']
        positions, means, sems = compute_adaptation_curve(sessions)
        means = np.array(means); sems = np.array(sems)
        ax.plot(positions, means, '-o', color=MODEL_COLORS[i], label=name,
                markersize=3, linewidth=2)
        ax.fill_between(positions, means - sems, means + sems,
                        color=MODEL_COLORS[i], alpha=0.15)
    ax.axvline(0, color='red', ls='--', alpha=0.5, label='Rule switch')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Trials relative to rule switch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Adaptation After Rule Switch', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig_wcst_adaptation.png', bbox_inches='tight')
    log(f"  Saved: {save_dir}/fig_wcst_adaptation.png")
    plt.close()


def plot_noise(noise_results, model_names, sigma_values, save_dir='figures/wcst'):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        persevs = [noise_results[name][s]['persev_errors_mean'] for s in sigma_values]
        ax.plot(sigma_values, persevs, '-o', color=MODEL_COLORS[i], label=name,
                markersize=5, linewidth=2.5)
    ax.set_xlabel('Noise σ (context signal degradation)')
    ax.set_ylabel('Perseverative errors (higher = worse)')
    ax.set_title('Context Signal Degradation Mimics Dopamine Disruption',
                 fontsize=13, fontweight='bold')
    ax.annotate('clean\ncontext', xy=(0, ax.get_ylim()[0]), fontsize=8,
                ha='center', color='#555')
    ax.annotate('context\nabolished', xy=(2.0, ax.get_ylim()[0]), fontsize=8,
                ha='center', color='#555')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig_wcst_noise.png', bbox_inches='tight')
    log(f"  Saved: {save_dir}/fig_wcst_noise.png")
    plt.close()


def plot_error_types(eval_results, model_names, save_dir='figures/wcst'):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.35
    persev = [eval_results[n]['persev_errors_mean'] for n in model_names]
    non_persev = [eval_results[n]['non_persev_errors_mean'] for n in model_names]
    ax.bar(x, persev, width, label='Perseverative errors', color='#E74C3C', alpha=0.85)
    ax.bar(x, non_persev, width, bottom=persev, label='Non-perseverative errors',
           color='#F39C12', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Number of errors')
    ax.set_title('Error Type Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig_wcst_errors.png', bbox_inches='tight')
    log(f"  Saved: {save_dir}/fig_wcst_errors.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    train_entities = pool.get('train')
    test_entities = pool.get('test')

    log("=" * 70)
    log("  WCST Experiment — Wisconsin Card Sorting Task")
    log("=" * 70)
    log(f"  Train entities: {len(train_entities)}  Test entities: {len(test_entities)}")
    log(f"  Rules: {WCST_RULES}")
    log(f"  Criterion: {CRITERION_TRIALS} consecutive correct")
    log(f"  Max completions: {MAX_COMPLETIONS}")
    log()

    # ── Phase 1: Train models on WCST ──
    model_names = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    models = []

    for name, ModelClass in [('ESBN', WCST_ESBN),
                              ('GEE-Direct', WCST_GEEDirect),
                              ('GEE-Separated', WCST_GEESeparated)]:
        log(f"{'─'*70}")
        log(f"  Training: {name}")
        log(f"{'─'*70}")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = ModelClass(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Params: {n_params:,}")
        t0 = time.time()
        train_wcst(model, train_entities, device, lr=cfg.lr, n_epochs=50,
                   n_per_rule=1000, seed=seed)
        log(f"  Training: {time.time()-t0:.0f}s")
        models.append(model)
        log()

    # ── Phase 2: Evaluate ──
    log("=" * 70)
    log("  WCST Evaluation (50 sessions per condition)")
    log("=" * 70)

    eval_results = {}
    for name, model in zip(model_names, models):
        log(f"\n  {name}:")
        r = run_evaluation(model, test_entities, device, n_sessions=50,
                           task_id_mode='normal')
        eval_results[name] = r
        log(f"    With task_id:    acc={r['accuracy_mean']:.1%} ± {r['accuracy_std']:.1%}  "
            f"persev={r['persev_errors_mean']:.1f}  completions={r['completions_mean']:.1f}")

        r_none = run_evaluation(model, test_entities, device, n_sessions=50,
                                task_id_mode='none')
        log(f"    No task_id:      acc={r_none['accuracy_mean']:.1%} ± {r_none['accuracy_std']:.1%}  "
            f"persev={r_none['persev_errors_mean']:.1f}")

        r_wrong = run_evaluation(model, test_entities, device, n_sessions=50,
                                 task_id_mode='wrong')
        log(f"    Wrong task_id:   acc={r_wrong['accuracy_mean']:.1%} ± {r_wrong['accuracy_std']:.1%}  "
            f"persev={r_wrong['persev_errors_mean']:.1f}")

    # ── Phase 3: Noise Injection ──
    log(f"\n{'='*70}")
    log("  Noise Injection Experiment")
    log(f"{'='*70}")
    sigma_values = [0, 0.1, 0.25, 0.5, 1.0, 2.0]
    noise_results = run_noise_experiment(models, model_names, test_entities, device,
                                         sigma_values, n_sessions=20)

    # ── Save Results ──
    results_summary = {
        'eval': {name: {
            'accuracy': eval_results[name]['accuracy_mean'],
            'persev_errors': eval_results[name]['persev_errors_mean'],
            'completions': eval_results[name]['completions_mean'],
        } for name in model_names},
        'noise': {name: {str(s): {
            'accuracy': noise_results[name][s]['accuracy_mean'],
            'persev_errors': noise_results[name][s]['persev_errors_mean'],
        } for s in sigma_values} for name in model_names},
    }
    with open('results/wcst_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    log("\n  Saved: results/wcst_results.json")

    # ── Figures ──
    log("\n  Generating figures...")
    plot_overview(eval_results, model_names)
    plot_adaptation(eval_results, model_names)
    plot_noise(noise_results, model_names, sigma_values)
    plot_error_types(eval_results, model_names)

    # ── Summary ──
    log(f"\n{'='*70}")
    log("  WCST Results Summary")
    log(f"{'='*70}")
    log(f"  {'Model':<18} {'Accuracy':>10} {'Persev Err':>12} {'Completions':>13}")
    log(f"  {'─'*55}")
    for name in model_names:
        r = eval_results[name]
        log(f"  {name:<18} {r['accuracy_mean']:>9.1%} {r['persev_errors_mean']:>11.1f} "
            f"{r['completions_mean']:>12.1f}")
    log(f"{'='*70}")
    log("\n  Done.")


if __name__ == '__main__':
    main()
