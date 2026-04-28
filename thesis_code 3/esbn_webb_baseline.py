"""
GRU-ESBN Baseline — All 5 Tasks (Webb-faithful + Original Dist3)
================================================================
Runs the Webb-faithful ESBN (LSTM→GRU only change) on:
  1. SameDifferent       (seq=2, binary)    — shared between original and Webb
  2. Webb RMTS           (seq=6, binary)    — 2 target pairs
  3. Webb Dist3          (seq=9, 4-way)     — set completion
  4. Webb IdentityRules  (seq=9, 4-way)     — ABA/ABB/AAA, 4 choices
  5. Original Dist3      (seq=4, binary)    — shared-attribute detection (our task)

Each task gets its own model instance with the appropriate output_dim.

Run: python -u esbn_webb_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from dataclasses import dataclass
from dataset import DatasetFactory
from esbn_pure import ContextNorm, AttributeAwareEncoder


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


@dataclass
class BaselineConfig:
    input_dim:   int = 12
    value_dim:   int = 128   # Webb: z_size=128
    key_dim:     int = 256   # Webb: key_size=256
    hidden_dim:  int = 512   # Webb: hidden_size=512
    lr:          float = 5e-4
    n_epochs:    int = 100
    batch_size:  int = 32
    seed:        int = 42


class PureESBN(nn.Module):
    """Webb-faithful ESBN with configurable output_dim."""
    def __init__(self, cfg: BaselineConfig, output_dim: int = 2):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim

        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, output_dim)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq):
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg
        device = x_seq.device

        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)

            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred_linear = self.y_out(gru_out).squeeze(1)

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias
                )
                key_r = g * (
                    torch.cat([M_k, c_k.unsqueeze(2)], dim=2)
                    * w_k.unsqueeze(2)
                ).sum(1).unsqueeze(1)

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
    cfg = BaselineConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=cfg.seed)

    log("=" * 70)
    log("  GRU-ESBN Baseline — 5 Tasks (Webb-faithful + Original Dist3)")
    log("=" * 70)
    log(f"  Device:     {device}")
    log(f"  value_dim:  {cfg.value_dim}  key_dim: {cfg.key_dim}  hidden: {cfg.hidden_dim}")
    log(f"  lr={cfg.lr}  epochs={cfg.n_epochs}  batch={cfg.batch_size}")
    log(f"  {factory.pool.summary()}")
    log()

    # 5 tasks: name, task object, output_dim, chance level
    tasks = [
        ("SameDifferent",       factory.same_different,          2, 0.50),
        ("Webb RMTS",           factory.webb_rmts,               2, 0.50),
        ("Webb Dist3",          factory.webb_distribution_of_3,  4, 0.25),
        ("Webb IdentityRules",  factory.webb_identity_rules,     4, 0.25),
        ("Original Dist3",      factory.distribution_of_3,       2, 0.50),
    ]

    results = {}

    for task_name, task, output_dim, chance in tasks:
        log(f"{'─'*70}")
        log(f"  {task_name} (seq_len={task.SEQ_LEN}, output_dim={output_dim}, chance={chance:.0%})")
        log(f"{'─'*70}")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        tr_inp, tr_lbl, _ = task.generate(3000, split='train')
        te_inp, te_lbl, _ = task.generate(500, split='test')
        train_loader = make_loader(tr_inp, tr_lbl, cfg.batch_size)
        test_loader = make_loader(te_inp, te_lbl, cfg.batch_size, shuffle=False)

        model = PureESBN(cfg, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Params: {n_params:,}")

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

        results[task_name] = (best_test, chance)
        log(f"  Best: {best_test:.1%}  ({time.time()-t0:.0f}s total)")
        log()

    log("=" * 70)
    log("  GRU-ESBN Baseline Results (5 Tasks)")
    log("=" * 70)
    log(f"  {'Task':<25} {'Test':>8} {'Chance':>8}  Status")
    log(f"  {'─'*55}")
    for name, (acc, chance) in results.items():
        thresh = max(chance + 0.40, 0.90)  # above chance + 40pp or 90%
        status = "PASS" if acc >= thresh else "BELOW"
        log(f"  {name:<25} {acc:>7.1%} {chance:>7.0%}  {status}")
    log("=" * 70)


if __name__ == '__main__':
    main()
