"""
WCST Feedback Experiment — Experiment 1: WCST without oracle signals
====================================================================
The model does NOT receive oracle task_id. Instead, it must infer the sorting
rule from feedback (correct/incorrect) on previous trials. The RCM maintains
cross-trial state, accumulating evidence about which rule is currently active.

Three-way comparison:
  - Oracle  (GEE-Direct with task_id)  — hardcoded from gee_wcst.py results
  - Feedback (FeedbackGEE)             — learns rule from feedback signal
  - ESBN    (no context)               — hardcoded from gee_wcst.py results

Run: python -u gee_wcst_feedback.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import EntityPool, ATTR_SIZES
from esbn_pure import ContextNorm, AttributeAwareEncoder
from gee_model import GEEConfig, RCMContextModule


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# WCST Configuration
# =============================================================================

WCST_RULES = ['color', 'shape', 'size', 'texture']
RULE_ATTR_IDX = [1, 0, 2, 3]  # color=1, shape=0, size=2, texture=3
N_RULES = 4
CRITERION_TRIALS = 6
MAX_COMPLETIONS = 6
MAX_TRIALS_PER_SESSION = 200

# Hardcoded oracle (GEE-Direct) and ESBN baselines from gee_wcst.py
ORACLE_RESULTS = {
    'accuracy': 0.949,
    'persev_errors': 1.1,
    'non_persev_errors': 5.0,
    'completions': 6.0,
}
ESBN_RESULTS = {
    'accuracy': 0.729,
    'persev_errors': 24.3,
    'non_persev_errors': 15.0,
    'completions': 5.9,
}


# =============================================================================
# FeedbackGEE Model
# =============================================================================

class FeedbackGEE(nn.Module):
    """
    Feedback-based GEE for WCST. No oracle task_id.

    Architecture:
      - ESBN core (same as WCST models) resets per trial
      - Cross-trial RCM persists across trials, receiving:
        ref(12) + test(12) + prev_feedback_onehot(2) = 26-dim input
      - RCM context biases the output logits
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ESBN core (same as WCST models)
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # Cross-trial RCM (replaces oracle task_id)
        # Input: ref(12) + test(12) + prev_feedback_onehot(2) = 26
        rcm_input_dim = cfg.input_dim * 2 + 2
        self.rcm = RCMContextModule(rcm_input_dim, 2, cfg.context_dim)

        # Output
        self.y_out = nn.Linear(cfg.hidden_dim, 2)
        self.ctx_to_out = nn.Linear(cfg.context_dim, 2, bias=False)

    def forward_trial(self, ref, test, prev_feedback, prev_pred, c_rcm, device):
        """
        Process one WCST trial. ESBN resets per trial, RCM persists.

        Args:
            ref:           (batch, 12) reference entity one-hot
            test:          (batch, 12) test entity one-hot
            prev_feedback: (batch, 2) one-hot feedback from previous trial
            prev_pred:     (batch, 2) softmax prediction from previous trial
            c_rcm:         (batch, context_dim) RCM hidden state
            device:        torch device

        Returns:
            logits:  (batch, 2)
            c_rcm:   (batch, context_dim) updated RCM state
        """
        batch = ref.shape[0]
        cfg = self.cfg

        # Update RCM with cross-trial info
        rcm_in = torch.cat([ref, test, prev_feedback], dim=-1)  # (batch, 26)
        c_rcm = self.rcm(rcm_in, prev_pred, c_rcm)

        # ESBN core: process (ref, test) as seq_len=2
        x_seq = torch.stack([ref, test], dim=1)  # (batch, 2, 12)
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, 2, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(3):  # seq_len(2) + 1
            z_t = (torch.zeros(batch, 1, cfg.value_dim, device=device)
                   if t == 2 else z_seq[:, t, :].unsqueeze(1))

            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2)
                             * w_k.unsqueeze(2)).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w
                M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1)
                M_v = torch.cat([M_v, z_t], dim=1)

        logits = self.y_out(gru_out.squeeze(1)) + self.ctx_to_out(c_rcm)
        return logits, c_rcm


# =============================================================================
# Training Session Generation
# =============================================================================

def generate_training_sessions(entities, batch_size, trials_per_session, rng):
    """Pre-generate batch of WCST sessions with fixed rule switches."""
    all_refs = []
    all_tests = []
    all_targets = []
    all_rules = []

    for b in range(batch_size):
        active_rule = rng.randint(N_RULES)
        switch_every = rng.randint(6, 11)  # switch every 6-10 trials
        trials_since_switch = 0

        refs, tests, targets, rules = [], [], [], []
        for t in range(trials_per_session):
            # Check for rule switch
            if trials_since_switch >= switch_every:
                active_rule = (active_rule + rng.randint(1, 4)) % 4
                switch_every = rng.randint(6, 11)
                trials_since_switch = 0

            # Generate trial
            attr_idx = RULE_ATTR_IDX[active_rule]
            is_match = rng.random() < 0.5
            ref = entities[rng.randint(len(entities))]
            if is_match:
                pool = [e for e in entities if e[attr_idx] == ref[attr_idx] and e != ref]
                if not pool:
                    pool = [e for e in entities if e[attr_idx] == ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]
            else:
                pool = [e for e in entities if e[attr_idx] != ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]

            refs.append(ref.one_hot())
            tests.append(test_e.one_hot())
            targets.append([1., 0.] if is_match else [0., 1.])
            rules.append(active_rule)
            trials_since_switch += 1

        all_refs.append(refs)
        all_tests.append(tests)
        all_targets.append(targets)
        all_rules.append(rules)

    # Shape: (trials_per_session, batch_size, dim)
    return (np.array(all_refs, dtype=np.float32).transpose(1, 0, 2),
            np.array(all_tests, dtype=np.float32).transpose(1, 0, 2),
            np.array(all_targets, dtype=np.float32).transpose(1, 0, 2),
            np.array(all_rules, dtype=np.int64).T)


# =============================================================================
# Training
# =============================================================================

def train_feedback_gee(model, train_entities, device, n_epochs=50,
                       n_batches_per_epoch=100, batch_size=32,
                       trials_per_session=30, lr=1e-4, seed=42):
    """Train FeedbackGEE on batched WCST sessions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.RandomState(seed)
    cfg = model.cfg

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in range(n_batches_per_epoch):
            # Generate a batch of sessions
            refs, tests, targets, rules = generate_training_sessions(
                train_entities, batch_size, trials_per_session, rng)

            refs_t = torch.tensor(refs, dtype=torch.float32, device=device)
            tests_t = torch.tensor(tests, dtype=torch.float32, device=device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=device)

            # Initialize cross-trial state
            c_rcm = torch.zeros(batch_size, cfg.context_dim, device=device)
            prev_feedback = torch.zeros(batch_size, 2, device=device)
            prev_pred = torch.zeros(batch_size, 2, device=device)

            total_loss = torch.tensor(0.0, device=device)

            for t in range(trials_per_session):
                logits, c_rcm = model.forward_trial(
                    refs_t[t], tests_t[t], prev_feedback, prev_pred, c_rcm, device)

                loss_t = F.cross_entropy(logits, targets_t[t])
                total_loss = total_loss + loss_t

                # Feedback based on model's OWN predictions (not teacher-forced)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    true_label = targets_t[t].argmax(dim=-1)
                    correct_mask = (pred == true_label).float()
                    # prev_feedback: [correct, incorrect] one-hot
                    prev_feedback = torch.stack([correct_mask, 1.0 - correct_mask], dim=-1)
                    prev_pred = torch.softmax(logits.detach(), dim=-1)
                    # Detach RCM state to avoid BPTT across entire session
                    # (truncated BPTT per trial)
                    c_rcm = c_rcm.detach()

                    epoch_correct += correct_mask.sum().item()
                    epoch_total += batch_size

            avg_loss = total_loss / trials_per_session
            optimizer.zero_grad()
            avg_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += avg_loss.item()

        avg_epoch_loss = epoch_loss / n_batches_per_epoch
        avg_epoch_acc = epoch_correct / max(epoch_total, 1)

        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={avg_epoch_loss:.4f} acc={avg_epoch_acc:.1%}")

    return model


# =============================================================================
# WCST Evaluation — Performance-dependent rule switching
# =============================================================================

@torch.no_grad()
def evaluate_wcst_session(model, test_entities, device, seed=None):
    """Run one WCST session with performance-dependent rule switching."""
    model.eval()
    rng = np.random.RandomState(seed)
    cfg = model.cfg

    active_rule = rng.randint(N_RULES)
    consecutive_correct = 0
    rules_completed = 0
    prev_rule = active_rule
    switch_points = []
    trial_results = []

    # Cross-trial state
    c_rcm = torch.zeros(1, cfg.context_dim, device=device)
    prev_feedback = torch.zeros(1, 2, device=device)
    prev_pred = torch.zeros(1, 2, device=device)

    for trial_idx in range(MAX_TRIALS_PER_SESSION):
        if rules_completed >= MAX_COMPLETIONS:
            break

        # Generate trial
        is_match = rng.random() < 0.5
        attr_idx = RULE_ATTR_IDX[active_rule]
        ref = test_entities[rng.randint(len(test_entities))]

        if is_match:
            pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx] and e != ref]
            if not pool:
                pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx]]
            test_e = pool[rng.randint(len(pool))]
        else:
            pool = [e for e in test_entities if e[attr_idx] != ref[attr_idx]]
            test_e = pool[rng.randint(len(pool))]

        ref_t = torch.tensor(ref.one_hot(), dtype=torch.float32, device=device).unsqueeze(0)
        test_t = torch.tensor(test_e.one_hot(), dtype=torch.float32, device=device).unsqueeze(0)

        logits, c_rcm = model.forward_trial(
            ref_t, test_t, prev_feedback, prev_pred, c_rcm, device)

        pred = logits.argmax(-1).item()
        true_label = 0 if is_match else 1
        correct = (pred == true_label)

        # Check perseverative error (matches previous rule, not current)
        is_perseverative = False
        if not correct and prev_rule != active_rule:
            old_attr_idx = RULE_ATTR_IDX[prev_rule]
            old_match = (ref[old_attr_idx] == test_e[old_attr_idx])
            old_correct_pred = 0 if old_match else 1
            if pred == old_correct_pred:
                is_perseverative = True

        trial_results.append({
            'trial_idx': trial_idx,
            'active_rule': active_rule,
            'correct': correct,
            'is_perseverative': is_perseverative,
            'is_post_switch': (prev_rule != active_rule
                               and consecutive_correct < CRITERION_TRIALS),
        })

        # Update feedback for next trial
        correct_f = 1.0 if correct else 0.0
        prev_feedback = torch.tensor([[correct_f, 1.0 - correct_f]],
                                     dtype=torch.float32, device=device)
        prev_pred = torch.softmax(logits.detach(), dim=-1)

        # Performance-dependent rule switching
        if correct:
            consecutive_correct += 1
            if consecutive_correct >= CRITERION_TRIALS:
                rules_completed += 1
                prev_rule = active_rule
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


def run_evaluation(model, test_entities, device, n_sessions=50):
    """Run multiple WCST sessions and aggregate results."""
    all_sessions = []
    for s in range(n_sessions):
        result = evaluate_wcst_session(model, test_entities, device, seed=s * 100)
        all_sessions.append(result)

    accuracies = []
    persev_errors = []
    non_persev_errors = []
    completions = []

    for sess in all_sessions:
        n_correct = sum(t['correct'] for t in sess['trials'])
        n_total = len(sess['trials'])
        accuracies.append(n_correct / max(n_total, 1))
        completions.append(sess['rules_completed'])
        persev = sum(t['is_perseverative'] for t in sess['trials'])
        non_persev = sum(not t['correct'] and not t['is_perseverative']
                         for t in sess['trials'])
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
    sems = [np.std(aligned[p]) / max(np.sqrt(len(aligned[p])), 1)
            if aligned[p] else np.nan for p in positions]
    return positions, means, sems


# =============================================================================
# Figures
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.dpi': 200,
})

# Colors: purple=Oracle, orange=Feedback, blue=ESBN
COLORS = {'Oracle': '#9B59B6', 'Feedback': '#E67E22', 'ESBN': '#5B9BD5'}


def plot_overview(feedback_results):
    """Fig 1: Three-panel overview comparing Oracle, Feedback, ESBN."""
    model_names = ['Oracle', 'Feedback', 'ESBN']
    colors = [COLORS[n] for n in model_names]

    accs = [ORACLE_RESULTS['accuracy'],
            feedback_results['accuracy_mean'],
            ESBN_RESULTS['accuracy']]
    persevs = [ORACLE_RESULTS['persev_errors'],
               feedback_results['persev_errors_mean'],
               ESBN_RESULTS['persev_errors']]
    comps = [ORACLE_RESULTS['completions'],
             feedback_results['completions_mean'],
             ESBN_RESULTS['completions']]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Accuracy
    ax = axes[0]
    bars = ax.bar(range(3), accs, color=colors, edgecolor='white', linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('A.  Overall accuracy', fontweight='bold', loc='left')
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Perseverative errors
    ax = axes[1]
    bars = ax.bar(range(3), persevs, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, persevs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Perseverative errors')
    ax.set_title('B.  Perseveration (lower = better)', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Rule completions
    ax = axes[2]
    bars = ax.bar(range(3), comps, color=colors, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, comps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Rules completed')
    ax.set_title('C.  Rule completion rate', fontweight='bold', loc='left')
    ax.set_ylim(0, MAX_COMPLETIONS + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('WCST Performance: Oracle vs Feedback vs ESBN',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_wcst_feedback_overview.png', bbox_inches='tight')
    log("  Saved: fig_wcst_feedback_overview.png")
    plt.close()


def plot_adaptation(feedback_results):
    """Fig 2: Post-switch recovery curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Feedback model adaptation curve (computed from actual sessions)
    positions, means, sems = compute_adaptation_curve(
        feedback_results['all_sessions'])
    means = np.array(means)
    sems = np.array(sems)
    ax.plot(positions, means, '-o', color=COLORS['Feedback'], label='Feedback',
            markersize=3, linewidth=2)
    ax.fill_between(positions, means - sems, means + sems,
                    color=COLORS['Feedback'], alpha=0.15)

    # Oracle and ESBN: plot flat reference lines from known results
    # (We don't have session-level data for these, so show mean accuracy)
    ax.axhline(ORACLE_RESULTS['accuracy'], color=COLORS['Oracle'], ls='--',
               alpha=0.7, linewidth=2, label=f"Oracle (mean={ORACLE_RESULTS['accuracy']:.0%})")
    ax.axhline(ESBN_RESULTS['accuracy'], color=COLORS['ESBN'], ls='--',
               alpha=0.7, linewidth=2, label=f"ESBN (mean={ESBN_RESULTS['accuracy']:.0%})")

    ax.axvline(0, color='red', ls='--', alpha=0.5, label='Rule switch')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Trials relative to rule switch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Adaptation After Rule Switch (Feedback Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_wcst_feedback_adaptation.png', bbox_inches='tight')
    log("  Saved: fig_wcst_feedback_adaptation.png")
    plt.close()


def plot_errors(feedback_results):
    """Fig 3: Error type breakdown (stacked bars)."""
    model_names = ['Oracle', 'Feedback', 'ESBN']
    colors_bar = ['#E74C3C', '#F39C12', '#95A5A6']

    persevs = [ORACLE_RESULTS['persev_errors'],
               feedback_results['persev_errors_mean'],
               ESBN_RESULTS['persev_errors']]
    non_persevs = [ORACLE_RESULTS['non_persev_errors'],
                   feedback_results['non_persev_errors_mean'],
                   ESBN_RESULTS['non_persev_errors']]
    # "Other" errors = total errors - persev - non_persev (should be ~0)
    # Total errors = total_trials * (1 - accuracy)
    # For Oracle and ESBN we approximate from known results
    # For Feedback we compute from actual data

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.5

    ax.bar(x, persevs, width, label='Perseverative errors',
           color=colors_bar[0], alpha=0.85)
    ax.bar(x, non_persevs, width, bottom=persevs,
           label='Non-perseverative errors', color=colors_bar[1], alpha=0.85)

    # Add totals on top
    for i in range(len(model_names)):
        total = persevs[i] + non_persevs[i]
        ax.text(i, total + 0.5, f'{total:.1f}', ha='center', fontsize=9,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Number of errors')
    ax.set_title('Error Type Analysis: Perseverative vs Non-perseverative',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_wcst_feedback_errors.png', bbox_inches='tight')
    log("  Saved: fig_wcst_feedback_errors.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    train_entities = pool.get('train')
    test_entities = pool.get('test')

    log("=" * 70)
    log("  WCST Feedback Experiment")
    log("  Model infers sorting rule from feedback (no oracle task_id)")
    log("=" * 70)
    log(f"  Train entities: {len(train_entities)}  Test entities: {len(test_entities)}")
    log(f"  Rules: {WCST_RULES}")
    log(f"  Criterion: {CRITERION_TRIALS} consecutive correct")
    log(f"  Max completions: {MAX_COMPLETIONS}")
    log(f"  Training: 50 epochs x 100 batches x 32 sessions x 30 trials")
    log()

    # ── Train FeedbackGEE ──
    log(f"{'='*70}")
    log("  Training: FeedbackGEE")
    log(f"{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = FeedbackGEE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Params: {n_params:,}")

    t0 = time.time()
    train_feedback_gee(
        model, train_entities, device,
        n_epochs=50, n_batches_per_epoch=100, batch_size=32,
        trials_per_session=30, lr=1e-4, seed=seed)
    train_time = time.time() - t0
    log(f"  Training: {train_time:.0f}s")
    log()

    # ── Evaluate ──
    log(f"{'='*70}")
    log("  WCST Evaluation (50 sessions)")
    log(f"{'='*70}")

    feedback_results = run_evaluation(model, test_entities, device, n_sessions=50)
    log(f"  Feedback GEE:")
    log(f"    Accuracy:       {feedback_results['accuracy_mean']:.1%} "
        f"+/- {feedback_results['accuracy_std']:.1%}")
    log(f"    Persev errors:  {feedback_results['persev_errors_mean']:.1f} "
        f"+/- {feedback_results['persev_errors_std']:.1f}")
    log(f"    Non-persev err: {feedback_results['non_persev_errors_mean']:.1f} "
        f"+/- {feedback_results['non_persev_errors_std']:.1f}")
    log(f"    Completions:    {feedback_results['completions_mean']:.1f} "
        f"+/- {feedback_results['completions_std']:.1f}")
    log()

    # ── Summary table ──
    log(f"{'='*70}")
    log("  Results Summary")
    log(f"{'='*70}")
    log(f"  {'Model':<18} {'Accuracy':>10} {'Persev Err':>12} {'Completions':>13}")
    log(f"  {'─'*55}")
    log(f"  {'Oracle (GEE-D)':<18} {ORACLE_RESULTS['accuracy']:>9.1%} "
        f"{ORACLE_RESULTS['persev_errors']:>11.1f} "
        f"{ORACLE_RESULTS['completions']:>12.1f}")
    log(f"  {'Feedback GEE':<18} {feedback_results['accuracy_mean']:>9.1%} "
        f"{feedback_results['persev_errors_mean']:>11.1f} "
        f"{feedback_results['completions_mean']:>12.1f}")
    log(f"  {'ESBN (no ctx)':<18} {ESBN_RESULTS['accuracy']:>9.1%} "
        f"{ESBN_RESULTS['persev_errors']:>11.1f} "
        f"{ESBN_RESULTS['completions']:>12.1f}")
    log(f"{'='*70}")

    # ── Save results ──
    results_json = {
        'feedback': {
            'accuracy': float(feedback_results['accuracy_mean']),
            'accuracy_std': float(feedback_results['accuracy_std']),
            'persev_errors': float(feedback_results['persev_errors_mean']),
            'persev_errors_std': float(feedback_results['persev_errors_std']),
            'non_persev_errors': float(feedback_results['non_persev_errors_mean']),
            'non_persev_errors_std': float(feedback_results['non_persev_errors_std']),
            'completions': float(feedback_results['completions_mean']),
            'completions_std': float(feedback_results['completions_std']),
        },
        'oracle': ORACLE_RESULTS,
        'esbn': ESBN_RESULTS,
        'training_time_s': train_time,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/wcst_feedback_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    log("\n  Saved: results/wcst_feedback_results.json")

    # Save to /private/tmp
    summary_lines = [
        "WCST Feedback Experiment Results",
        "=" * 50,
        f"Model: FeedbackGEE (no oracle task_id)",
        f"Training: 50 epochs x 100 batches x 32 sessions x 30 trials",
        f"Training time: {train_time:.0f}s",
        "",
        f"{'Model':<18} {'Accuracy':>10} {'Persev Err':>12} {'Completions':>13}",
        f"{'─'*55}",
        f"{'Oracle (GEE-D)':<18} {ORACLE_RESULTS['accuracy']:>9.1%} "
        f"{ORACLE_RESULTS['persev_errors']:>11.1f} "
        f"{ORACLE_RESULTS['completions']:>12.1f}",
        f"{'Feedback GEE':<18} {feedback_results['accuracy_mean']:>9.1%} "
        f"{feedback_results['persev_errors_mean']:>11.1f} "
        f"{feedback_results['completions_mean']:>12.1f}",
        f"{'ESBN (no ctx)':<18} {ESBN_RESULTS['accuracy']:>9.1%} "
        f"{ESBN_RESULTS['persev_errors']:>11.1f} "
        f"{ESBN_RESULTS['completions']:>12.1f}",
        "=" * 50,
    ]
    with open('/private/tmp/wcst_feedback.txt', 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    log("  Saved: /private/tmp/wcst_feedback.txt")

    # ── Figures ──
    log("\n  Generating figures...")
    plot_overview(feedback_results)
    plot_adaptation(feedback_results)
    plot_errors(feedback_results)

    log("\n  Done.")


if __name__ == '__main__':
    main()
