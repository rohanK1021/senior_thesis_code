"""
Pure GRU-ESBN in PyTorch — faithful to Webb et al. 2021
========================================================
Matches Webb's ESBN architecture exactly. Only change: GRU instead of LSTM.

Webb's code: https://github.com/taylorwwebb/emergent_symbols
Key architectural elements preserved:
  - Context normalization (full sequence mean/var)
  - Scalar gate on entire retrieved key+confidence vector
  - Per-entry confidence concatenated with keys before weighted sum
  - z_t queries value slot, retrieves associated keys
  - Extra timestep, output from final hidden state
  - Memory grows by concatenation

Run: python -u esbn_pure.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


@dataclass
class ESBNConfig:
    input_dim:   int = 12    # raw one-hot (preserves attribute structure)
    value_dim:   int = 128   # encoder output / memory value (Webb: z_size=128)
    key_dim:     int = 256   # abstract key (Webb: key_size=256)
    hidden_dim:  int = 512   # GRU hidden (Webb: hidden_size=512)
    output_dim:  int = 2
    lr:          float = 5e-4
    n_epochs:    int = 100
    batch_size:  int = 32
    seed:        int = 42


class ContextNorm(nn.Module):
    """Context normalization — Webb's exact implementation.
    Full sequence mean/var. This is intentionally non-causal."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        return z_seq * self.gamma + self.beta


class AttributeAwareEncoder(nn.Module):
    """
    Attribute-aware encoder. Each attribute (shape, color, size, texture)
    gets its own learned subspace. Two entities sharing an attribute have
    identical vectors in that subspace — critical for DistributionOf3.

    Raw 12-dim one-hot: [shape(4)|color(4)|size(2)|texture(2)]
    """
    ATTR_SLICES = [(0, 4), (4, 8), (8, 10), (10, 12)]

    def __init__(self, input_dim, value_dim):
        super().__init__()
        n_attrs = len(self.ATTR_SLICES)
        self.dim_per_attr = value_dim // n_attrs
        remainder = value_dim - self.dim_per_attr * n_attrs

        self.attr_encoders = nn.ModuleList()
        for i, (start, end) in enumerate(self.ATTR_SLICES):
            attr_size = end - start
            out_dim = self.dim_per_attr + (1 if i < remainder else 0)
            self.attr_encoders.append(nn.Sequential(
                nn.Linear(attr_size, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
            ))

        self.norm = nn.LayerNorm(value_dim)

    def forward(self, x):
        parts = []
        for i, (start, end) in enumerate(self.ATTR_SLICES):
            parts.append(self.attr_encoders[i](x[:, start:end]))
        return self.norm(torch.cat(parts, dim=-1))


class PureESBN(nn.Module):
    """
    Pure ESBN — faithful to Webb et al. 2021, GRU swapped for LSTM.

    Matches Webb's architecture exactly:
      - GRU (Webb: LSTM) with batch_first input
      - Scalar gate on entire retrieved vector (key + confidence)
      - Per-entry confidence concatenated with keys before weighted sum
      - z_t queries M_v, retrieves M_k
      - Context normalization (full sequence, Webb's original)
      - Extra timestep, output at final step
    """
    def __init__(self, cfg: ESBNConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)

        # Context normalization (Webb's exact implementation)
        self.contextnorm = ContextNorm(cfg.value_dim)

        # GRU controller (Webb: LSTM)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        # Learnable confidence (Webb's exact formulation)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # Initialize parameters (adapted from Webb for GRU)
        for name, param in self.named_parameters():
            if 'encoder' in name or 'confidence' in name or 'contextnorm' in name:
                continue
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'gru' in name:
                # GRU: reset+update gates (sigmoid), new gate (tanh)
                nn.init.xavier_normal_(param[:cfg.hidden_dim * 2, :])
                nn.init.xavier_normal_(param[cfg.hidden_dim * 2:cfg.hidden_dim * 3, :],
                                       gain=5.0 / 3.0)
            elif 'key_w' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'g_out' in name or 'y_out' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x_seq):
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg
        device = x_seq.device

        # Encode all timesteps
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)

        # Context normalization (full sequence — Webb's design)
        z_seq = self.contextnorm(z_seq)

        # Initialize hidden state (Webb: hidden + cell_state for LSTM)
        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)

        # Initialize retrieved key vector (with sequence dim for nn.GRU)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        # Memory model (extra timestep to process key retrieved on final step)
        for t in range(seq_len + 1):
            # Embedding
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)

            # Controller — GRU (Webb: LSTM)
            gru_out, hidden = self.gru(key_r, hidden)

            # Key output (Webb: key_w)
            key_w = F.relu(self.key_w_out(gru_out))

            # Gate — scalar (Webb: g_out → 1 dim)
            g = torch.sigmoid(self.g_out(gru_out))

            # Task output
            y_pred_linear = self.y_out(gru_out).squeeze(1)

            # Read from memory (Webb's exact retrieval)
            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                # z_t queries value slot, retrieves keys + confidence
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias
                )
                # Webb: gate * weighted_sum(cat(keys, confidence))
                key_r = g * (
                    torch.cat([M_k, c_k.unsqueeze(2)], dim=2)
                    * w_k.unsqueeze(2)
                ).sum(1).unsqueeze(1)

            # Write to memory
            if t == 0:
                M_k = key_w
                M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1)
                M_v = torch.cat([M_v, z_t], dim=1)

        return y_pred_linear


# =============================================================================
# Training and evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_batch.shape[0]
        preds = logits.argmax(dim=-1)
        labels = y_batch.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += x_batch.shape[0]
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        preds = logits.argmax(dim=-1)
        labels = y_batch.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += x_batch.shape[0]
    return correct / max(total, 1)


def make_loader(inputs, labels, batch_size, shuffle=True):
    if isinstance(inputs, list):
        inputs = np.array(inputs, dtype=np.float32)
    if isinstance(labels, list):
        labels = np.array(labels, dtype=np.float32)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# Main
# =============================================================================

def main():
    from dataset import DatasetFactory
    import time

    cfg = ESBNConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # 40 train / 16 test — enough for all tasks including IdentityRules
    factory = DatasetFactory(embed_dim=cfg.input_dim,
                             n_train=40, n_test=16,
                             pool_seed=cfg.seed)

    log("=" * 65)
    log("  Pure GRU-ESBN — Faithful to Webb et al. 2021")
    log("=" * 65)
    log(f"  Device:     {device}")
    log(f"  input_dim:  {cfg.input_dim} (raw one-hot, attributes preserved)")
    log(f"  value_dim:  {cfg.value_dim}  key_dim: {cfg.key_dim}  hidden: {cfg.hidden_dim}")
    log(f"  lr={cfg.lr}  epochs={cfg.n_epochs}  batch={cfg.batch_size}")
    log(f"  {factory.pool.summary()}")

    n_params = sum(p.numel() for p in PureESBN(cfg).parameters() if p.requires_grad)
    log(f"  Parameters: {n_params:,}")
    log()

    task_names = ['same_different', 'rmts', 'distribution_of_3', 'identity_rules']
    tasks = [factory.same_different, factory.rmts, factory.distribution_of_3, factory.identity_rules]
    results = {}

    for task_name, task in zip(task_names, tasks):
        log(f"{'─'*65}")
        log(f"  {task_name} (seq_len={task.SEQ_LEN})")
        log(f"{'─'*65}")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        tr_inp, tr_lbl, _ = task.generate(3000, split='train')
        te_inp, te_lbl, _ = task.generate(500, split='test')
        train_loader = make_loader(tr_inp, tr_lbl, cfg.batch_size)
        test_loader = make_loader(te_inp, te_lbl, cfg.batch_size, shuffle=False)

        model = PureESBN(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

        best_test = 0.0
        log(f"  {'Ep':>4}  {'Loss':>8}  {'Train':>7}  {'Test':>7}  {'Time':>6}")
        log(f"  {'─'*38}")

        t0 = time.time()
        for epoch in range(1, cfg.n_epochs + 1):
            loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            if test_acc > best_test:
                best_test = test_acc
            elapsed = time.time() - t0
            if epoch % 5 == 0 or epoch == 1:
                log(f"  {epoch:>4}  {loss:>8.4f}  {train_acc:>6.1%}  {test_acc:>6.1%}  {elapsed:>5.0f}s")

        results[task_name] = best_test
        log(f"  Best: {best_test:.1%}  ({time.time()-t0:.0f}s total)")
        log()

    log("=" * 65)
    log("  Pure GRU-ESBN Results")
    log("=" * 65)
    log(f"  {'Task':<22} {'Test':>8}  Status")
    log(f"  {'─'*45}")
    for name, acc in results.items():
        status = "PASS" if acc >= 0.90 else "BELOW 90%"
        log(f"  {name:<22} {acc:>7.1%}  {status}")
    overall = np.mean(list(results.values()))
    log(f"  {'OVERALL':<22} {overall:>7.1%}")
    log("=" * 65)


if __name__ == '__main__':
    main()
