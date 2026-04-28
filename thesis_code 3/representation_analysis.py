"""
Representation Analysis
========================
Extract and visualize internal representations from the trained unified GEE.

Analyses:
  1. t-SNE / PCA of GRU final hidden states, colored by task
  2. t-SNE / PCA of write keys, colored by task and by timestep
  3. Comparison under h₀ noise (σ=2.0) and RCM noise (σ=2.0)

Figures:
  fig_repr_hidden_tsne.png  — t-SNE of final hidden states, colored by task
  fig_repr_keys_tsne.png    — t-SNE of write keys, colored by task
  fig_repr_noise_comparison.png — 3 panels (clean / h₀ noise / RCM noise) hidden-state t-SNE

Run: python -u representation_analysis.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# sklearn — required for t-SNE and PCA
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from dataset import EntityPool
from gee_model import GEEConfig
from gee_unified import GEE
from unified_tasks import UnifiedTaskFactory, SEQ_LEN, N_TASKS, TASK_NAMES, TASK_SHORTS


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


N_EVAL = 500   # trials per task

# Per-task colors (same as noise experiment)
TASK_COLORS = ['#5B9BD5', '#E74C3C', '#F39C12', '#2ECC71']

# Timestep colormap
TIMESTEP_CMAP = 'plasma'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'figure.dpi': 200,
})


# =============================================================================
# Model loading
# =============================================================================

def load_model(cfg, device, path='results/gee_unified.pt'):
    model = GEE(cfg, n_tasks=N_TASKS, use_h0=True, use_rcm=True, dropout_p=0.3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    log(f"  Loaded: {path}")
    return model


# =============================================================================
# Representation extraction via hooks
# =============================================================================

@torch.no_grad()
def extract_representations(model, factory, device, n_eval=N_EVAL,
                             noise_sigma=0.0, h0_noise_sigma=0.0,
                             rcm_noise_sigma=0.0,
                             h0_gain=1.0, rcm_gain=1.0):
    """
    Run model on test data for all tasks and extract:
      - final_hidden: GRU hidden state at the last timestep (batch, hidden_dim)
      - write_keys:   write keys at each timestep (batch * (seq_len+1), key_dim)
      - task_labels:  integer task id for each sample in final_hidden
      - key_task_labels:   task label for each write key
      - key_timestep_labels: timestep index for each write key

    Uses PyTorch hooks on model.gru and model.key_w_out.
    """
    model.eval()

    all_hidden    = []  # one per forward call — the LAST step's hidden state
    all_write_keys = []  # write keys from each forward call, all timesteps flattened
    all_task_labels = []
    all_key_task_labels = []
    all_key_timestep_labels = []

    # ── Hook: capture GRU hidden state ──
    # nn.GRU with batch_first=True: output is (output_seq, hidden)
    # output_seq shape: (batch, seq_len_call, hidden_dim)
    # hidden shape:     (num_layers, batch, hidden_dim)
    # We call gru with seq=1 at each step, so output_seq is (batch, 1, hidden_dim).
    # We want the hidden at the LAST step of the full sequence (t=seq_len).
    # Strategy: capture all and keep track of call count per forward pass.

    step_count = []  # mutable container to track step index in current pass
    gru_hidden_buffer = []

    def gru_hook(module, inp, output):
        # output = (gru_out, hidden); hidden shape: (1, batch, hidden_dim)
        gru_hidden_buffer.append(output[1].detach().squeeze(0))  # (batch, hidden_dim)

    # ── Hook: capture write keys ──
    key_w_buffer = []  # list of (batch, 1, key_dim) tensors, one per step

    def key_hook(module, inp, output):
        key_w_buffer.append(F.relu(output).detach().squeeze(1))  # (batch, key_dim)

    handle_gru = model.gru.register_forward_hook(gru_hook)
    handle_key = model.key_w_out.register_forward_hook(key_hook)

    try:
        for tid in range(N_TASKS):
            inp, tgt = factory.generate_batch(tid, n_eval, split='test')
            x = torch.tensor(inp, dtype=torch.float32, device=device)

            # Clear buffers before each forward pass
            gru_hidden_buffer.clear()
            key_w_buffer.clear()

            kwargs = dict(
                task_id=tid, output_dim=2, device=device,
                noise_sigma=noise_sigma,
                h0_noise_sigma=h0_noise_sigma,
                rcm_noise_sigma=rcm_noise_sigma,
                h0_gain=h0_gain,
                rcm_gain=rcm_gain,
            )
            _ = model(x, **kwargs)

            # gru_hidden_buffer now has (seq_len+1) entries, each (batch, hidden_dim)
            # The last entry corresponds to t=seq_len (the extra timestep)
            if gru_hidden_buffer:
                final_h = gru_hidden_buffer[-1]           # (batch, hidden_dim)
                all_hidden.append(final_h.cpu().numpy())
                all_task_labels.extend([tid] * n_eval)

            # key_w_buffer has (seq_len+1) entries
            actual_steps = len(key_w_buffer)
            for step_idx, kw in enumerate(key_w_buffer):  # (batch, key_dim)
                all_write_keys.append(kw.cpu().numpy())
                all_key_task_labels.extend([tid] * n_eval)
                all_key_timestep_labels.extend([step_idx] * n_eval)

    finally:
        handle_gru.remove()
        handle_key.remove()

    hidden_arr   = np.concatenate(all_hidden,    axis=0)  # (N_tasks*n_eval, hidden_dim)
    keys_arr     = np.concatenate(all_write_keys, axis=0)  # (N_tasks*(seq_len+1)*n_eval, key_dim)
    task_labels  = np.array(all_task_labels)               # (N_tasks*n_eval,)
    key_task_labels     = np.array(all_key_task_labels)
    key_timestep_labels = np.array(all_key_timestep_labels)

    return {
        'hidden':     hidden_arr,
        'keys':       keys_arr,
        'task':       task_labels,
        'key_task':   key_task_labels,
        'key_tstep':  key_timestep_labels,
    }


# =============================================================================
# Dimensionality reduction helpers
# =============================================================================

def run_tsne(data, perplexity=30, random_state=42, n_components=2):
    """Run t-SNE. Subsample if very large."""
    max_pts = 4000
    if len(data) > max_pts:
        idx = np.random.RandomState(random_state).choice(len(data), max_pts, replace=False)
        data = data[idx]
        return TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=random_state, max_iter=1000).fit_transform(data), idx
    return TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, max_iter=1000).fit_transform(data), None


def run_pca(data, n_components=2):
    return PCA(n_components=n_components).fit_transform(data)


# =============================================================================
# Figure 1: Hidden state t-SNE colored by task
# =============================================================================

def plot_hidden_tsne(reps, title_suffix='', save_path='fig_repr_hidden_tsne.png'):
    hidden = reps['hidden']
    task   = reps['task']

    log(f"  t-SNE on hidden states ({hidden.shape[0]} pts, {hidden.shape[1]} dims)...")
    emb, idx = run_tsne(hidden)
    if idx is not None:
        task = task[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: t-SNE
    ax = axes[0]
    for tid in range(N_TASKS):
        mask = task == tid
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=TASK_COLORS[tid], s=6, alpha=0.5,
                   label=TASK_SHORTS[tid])
    ax.set_title(f'A.  t-SNE — GRU Hidden States{title_suffix}',
                 fontweight='bold', loc='left')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2, framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: PCA
    log(f"  PCA on hidden states...")
    pca_emb = run_pca(hidden)
    if idx is not None:
        task_pca = reps['task'][idx]
    else:
        task_pca = reps['task']

    ax = axes[1]
    for tid in range(N_TASKS):
        mask = task_pca == tid
        ax.scatter(pca_emb[mask, 0], pca_emb[mask, 1],
                   c=TASK_COLORS[tid], s=6, alpha=0.5,
                   label=TASK_SHORTS[tid])
    ax.set_title(f'B.  PCA — GRU Hidden States{title_suffix}',
                 fontweight='bold', loc='left')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(markerscale=2, framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    log(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# Figure 2: Write key t-SNE colored by task and by timestep
# =============================================================================

def plot_keys_tsne(reps):
    keys      = reps['keys']
    key_task  = reps['key_task']
    key_tstep = reps['key_tstep']

    # Subsample keys for tractable t-SNE (keys has N_tasks*(seq_len+1)*n_eval rows)
    rng = np.random.RandomState(42)
    max_pts = 4000
    if len(keys) > max_pts:
        idx = rng.choice(len(keys), max_pts, replace=False)
        keys_sub      = keys[idx]
        key_task_sub  = key_task[idx]
        key_tstep_sub = key_tstep[idx]
    else:
        keys_sub      = keys
        key_task_sub  = key_task
        key_tstep_sub = key_tstep

    log(f"  t-SNE on write keys ({len(keys_sub)} pts, {keys_sub.shape[1]} dims)...")
    emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(keys_sub)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: colored by task
    ax = axes[0]
    for tid in range(N_TASKS):
        mask = key_task_sub == tid
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=TASK_COLORS[tid], s=5, alpha=0.4,
                   label=TASK_SHORTS[tid])
    ax.set_title('A.  Write Keys — Colored by Task', fontweight='bold', loc='left')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2, framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: colored by timestep
    ax = axes[1]
    n_steps = key_tstep_sub.max() + 1
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=key_tstep_sub, cmap=TIMESTEP_CMAP,
                    s=5, alpha=0.4, vmin=0, vmax=n_steps - 1)
    plt.colorbar(sc, ax=ax, label='Timestep')
    ax.set_title('B.  Write Keys — Colored by Timestep', fontweight='bold', loc='left')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_repr_keys_tsne.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_repr_keys_tsne.png")
    plt.close()


# =============================================================================
# Figure 3: Noise comparison — 3 panels (clean / h0 noise / RCM noise)
# =============================================================================

def plot_noise_comparison(reps_clean, reps_h0, reps_rcm):
    """3-panel t-SNE of hidden states under clean / h₀ noise / RCM noise."""
    conditions = [
        (reps_clean, 'Clean',         '#000000'),
        (reps_h0,   'h\u2080 noise (σ=2)', '#9B59B6'),
        (reps_rcm,  'RCM noise (σ=2)', '#E67E22'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (reps, title, _) in zip(axes, conditions):
        hidden = reps['hidden']
        task   = reps['task']

        log(f"  t-SNE for '{title}' ({hidden.shape[0]} pts)...")
        emb, idx = run_tsne(hidden)
        if idx is not None:
            task = task[idx]

        for tid in range(N_TASKS):
            mask = task == tid
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=TASK_COLORS[tid], s=6, alpha=0.45,
                       label=TASK_SHORTS[tid])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.08)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend on last axis
    legend_handles = [
        Patch(color=TASK_COLORS[tid], label=TASK_SHORTS[tid])
        for tid in range(N_TASKS)
    ]
    axes[-1].legend(handles=legend_handles, markerscale=2, framealpha=0.8, fontsize=9,
                    loc='best')

    plt.suptitle('GRU Hidden State Geometry Under Context Noise',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_repr_noise_comparison.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_repr_noise_comparison.png")
    plt.close()


# =============================================================================
# Separation metric: between-task / within-task distance ratio
# =============================================================================

def compute_separation(hidden, task_labels, n_tasks=N_TASKS):
    """Silhouette-like measure: mean between-class dist / mean within-class dist."""
    centroids = np.array([hidden[task_labels == t].mean(0) for t in range(n_tasks)])

    # within-class mean pairwise distance
    within = []
    for t in range(n_tasks):
        pts = hidden[task_labels == t]
        if len(pts) > 1:
            # mean L2 distance from centroid
            dists = np.linalg.norm(pts - centroids[t], axis=1)
            within.append(dists.mean())
    within_mean = float(np.mean(within))

    # between-class mean pairwise distance of centroids
    between = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            between.append(np.linalg.norm(centroids[i] - centroids[j]))
    between_mean = float(np.mean(between))

    ratio = between_mean / max(within_mean, 1e-8)
    return {'between': between_mean, 'within': within_mean, 'ratio': ratio}


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn is required for this script.\n"
            "Install with: pip install scikit-learn"
        )

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    factory = UnifiedTaskFactory(pool.get('train'), pool.get('test'), seed=seed)

    log("=" * 70)
    log("  Representation Analysis — Unified GEE")
    log("=" * 70)
    log()

    # ── Load model ──
    model = load_model(cfg, device, 'results/gee_unified.pt')
    log()

    # ── Extract clean representations ──
    log(f"{'─'*70}")
    log("  Extracting clean representations...")
    log(f"{'─'*70}")
    reps_clean = extract_representations(model, factory, device, n_eval=N_EVAL)
    sep_clean = compute_separation(reps_clean['hidden'], reps_clean['task'])
    log(f"  Hidden state separation: between={sep_clean['between']:.3f}  "
        f"within={sep_clean['within']:.3f}  ratio={sep_clean['ratio']:.3f}")
    log()

    # ── Extract h₀-noise representations ──
    log(f"{'─'*70}")
    log("  Extracting h\u2080 noise representations (σ=2.0)...")
    log(f"{'─'*70}")
    reps_h0 = extract_representations(model, factory, device, n_eval=N_EVAL,
                                       h0_noise_sigma=2.0)
    sep_h0 = compute_separation(reps_h0['hidden'], reps_h0['task'])
    log(f"  Hidden state separation: between={sep_h0['between']:.3f}  "
        f"within={sep_h0['within']:.3f}  ratio={sep_h0['ratio']:.3f}")
    log()

    # ── Extract RCM-noise representations ──
    log(f"{'─'*70}")
    log("  Extracting RCM noise representations (σ=2.0)...")
    log(f"{'─'*70}")
    reps_rcm = extract_representations(model, factory, device, n_eval=N_EVAL,
                                        rcm_noise_sigma=2.0)
    sep_rcm = compute_separation(reps_rcm['hidden'], reps_rcm['task'])
    log(f"  Hidden state separation: between={sep_rcm['between']:.3f}  "
        f"within={sep_rcm['within']:.3f}  ratio={sep_rcm['ratio']:.3f}")
    log()

    # ── Save separation metrics ──
    os.makedirs('results', exist_ok=True)
    sep_data = {
        'clean': sep_clean,
        'h0_noise_2': sep_h0,
        'rcm_noise_2': sep_rcm,
        'n_eval_per_task': N_EVAL,
        'hidden_dim': int(reps_clean['hidden'].shape[1]),
        'key_dim':    int(reps_clean['keys'].shape[1]),
    }
    with open('results/representation_analysis.json', 'w') as f:
        json.dump(sep_data, f, indent=2)
    log("  Saved: results/representation_analysis.json")
    log()

    # ── Figures ──
    log(f"{'─'*70}")
    log("  Generating figures...")
    log(f"{'─'*70}")

    # Figure 1: Hidden state t-SNE (clean)
    plot_hidden_tsne(reps_clean, title_suffix=' (clean)',
                     save_path='fig_repr_hidden_tsne.png')

    # Figure 2: Write key t-SNE (clean)
    plot_keys_tsne(reps_clean)

    # Figure 3: Noise comparison
    plot_noise_comparison(reps_clean, reps_h0, reps_rcm)

    # ── Summary ──
    log()
    log(f"{'='*70}")
    log("  Summary — Hidden state separation (between/within ratio)")
    log(f"{'='*70}")
    log(f"  Clean:          ratio={sep_clean['ratio']:.3f}")
    log(f"  h\u2080 noise σ=2:  ratio={sep_h0['ratio']:.3f}  "
        f"(change: {sep_h0['ratio']-sep_clean['ratio']:+.3f})")
    log(f"  RCM noise σ=2:  ratio={sep_rcm['ratio']:.3f}  "
        f"(change: {sep_rcm['ratio']-sep_clean['ratio']:+.3f})")
    log()
    log("  Done.")


if __name__ == '__main__':
    main()
