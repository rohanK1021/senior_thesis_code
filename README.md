# Two Signatures of Context Failure

Code accompanying the senior thesis **"Two Signatures of Context Failure: A Computational Framework for Disambiguating WCST Perseveration"** (Princeton University, 2026).

**Author:** Rohan Kumar
**Advisor:** Jonathan D. Cohen

This repository contains the PyTorch implementation of the GEE (GRU-ESBN-EGO) hybrid architecture, the four reasoning tasks (Webb et al. 2021), the Wisconsin Card Sorting Task (WCST), and all experimental scripts used to generate the figures and tables in the thesis.

---

## Quick start

```bash
# Clone the repository
git clone https://github.com/rohanK1021/senior_thesis_code.git
cd senior_thesis_code

# Install dependencies (Python 3.10 recommended)
pip install -r requirements.txt

# Reproduce the central crossover result (Figure 10)
python gee_noise_experiment.py
python gee_wcst_noise.py
python plot_multiseed.py
```

All experiments use random seeds **42, 123, 456, 789, 1024** for multi-seed runs.

---

## Architectures (§3.3)

| File | Architecture | Thesis reference |
|---|---|---|
| `esbn_pure.py` | ESBN baseline (no context); GRU controller, Webb-faithful indirection | §3.3 ESBN |
| `true_gee.py` | GEE-Direct: task ID enters write keys + RCM biases output | §3.3 GEE-Direct |
| `gee_separated.py` | GEE-Separated: task ID sets `h₀` only; task-ID dropout p=0.5 | §3.3 GEE-Separated |
| `gee_unified.py` | GEE-Combined: `h₀` + RCM, no key modulation (used for noise/gain experiments) | §3.3 GEE-Combined |
| `ego_webb.py` | EGO-Baseline: same context pathways as GEE-Combined but **no indirection** | §3.3 EGO-Baseline, §5.5 |
| `gee_model.py` | Shared core: `GEEConfig`, `RCMContextModule`, encoders | §3.3, §2.4 |
| `ego_pytorch.py` | Standalone PyTorch port of Giallanza et al. 2024 EGO (reference only) | §2.4.3 |

## Tasks (§3.4)

| File | Contents |
|---|---|
| `dataset.py` | 64-entity system (4 shapes × 4 colors × 2 sizes × 2 textures); Webb-faithful task generators |
| `unified_tasks.py` | Fixed-length (seq=6) binary-output variants used in §4.2 |

The WCST is implemented inside the experiment scripts (see `gee_wcst.py`, `gee_wcst_noise.py`, `gee_wcst_feedback.py`).

---

## Mapping: scripts → thesis figures and tables

### Figures

| Figure | Thesis section | Script(s) |
|---|---|---|
| **Figure 1** — Architecture diagrams | §3.3 | `generate_architecture_diagrams.py` |
| **Figure 2** — Four reasoning tasks | §3.4 | `generate_task_diagrams.py` |
| **Figure 3** — WCST task structure | §3.4 | `generate_task_diagrams.py` |
| **Figure 4** — Standard vs. Unified task conditions | §3.4 | `generate_task_diagrams.py` |
| **Figure 5** — Multi-seed Webb reasoning task results | §4.1 | `run_seeds.py` → `plot_multiseed.py` (`fig_webb`) |
| **Figure 6** — Dropout training eliminates GEE-Direct's reliance on task label | §4.1 | `run_webb_no_dropout.py`, `plot_dropout_effect.py` |
| **Figure 7** — Unified tasks (multi-seed) | §4.2 | `gee_main.py`, `run_seeds.py` → `plot_multiseed.py` (`fig_unified`) |
| **Figure 8** — WCST multi-seed results | §4.2 | `gee_wcst.py`, `run_seeds.py` → `plot_multiseed.py` (`fig_wcst`) |
| **Figure 9** — Post-switch adaptation in WCST | §4.2 | `gee_wcst.py` |
| **Figure 10** — Pathway-specific noise (the crossover) | §4.3 | `gee_noise_experiment.py`, `gee_wcst_noise.py` → `plot_multiseed.py` (`fig_figure11_noise`) |
| **Figure 11** — WCST perseveration under pathway-specific noise | §4.3.3 | `gee_wcst_noise.py` |
| **Figure 12** — Crossover interaction (central finding) | §4.3.3 | `plot_multiseed.py` (`fig_double_dissociation`) |
| **Figure 13** — t-SNE of GRU hidden states | §4.4.1 | `representation_analysis.py` |
| **Figure 14** — Confusion matrices at σ=1.0 | §4.4.2 | `error_analysis.py` |
| **Figure 15** — Nonlinear interaction between context pathways | §4.4.2 | `nonlinearity_analysis.py` |
| **Figure 16** — Crossover replicates under gain modulation | §5.3 | `gee_gain_experiment.py` |
| **Figure 17** — Response consistency under gain vs. additive noise | §5.3 | `gee_gain_experiment.py` |

### Tables

| Table | Thesis section | Script(s) |
|---|---|---|
| **Table 1** — Robustness to task-ID removal on reasoning tasks | §4.1 | `gee_webb_comparison.py`, `run_webb_no_dropout.py` |
| **Table 2** — Noise effects on fixed-rule tasks | §4.3.1 | `gee_noise_experiment.py` |
| **Table 3** — Noise effects on WCST | §4.3.2 | `gee_wcst_noise.py` |
| **Table 4** — Three signatures of context failure | §4.3.3.3 | `gee_noise_experiment.py`, `gee_wcst_noise.py` |
| **Table 5** — Indirection tradeoff (EGO vs. ESBN vs. GEE-Combined) | §5.5 | `ego_webb.py`, `gee_webb_comparison.py` |

---

## Experiment scripts by thesis stage

### Stage 1 — Boundary condition (§4.1)
- `gee_webb_comparison.py` — three-way ESBN / GEE-Direct / GEE-Separated training on Webb tasks
- `run_webb_no_dropout.py` — GEE-Direct without dropout (the −40.5% drop)
- `esbn_webb_baseline.py` — single-task ESBN replication of Webb 2021

### Stage 2 — When context matters (§4.2)
- `gee_main.py` — unified-task training (ESBN / GEE / GEE-h₀)
- `gee_wcst.py` — WCST training, post-switch adaptation curves

### Stage 3 — Crossover (§4.3)
- `gee_noise_experiment.py` — pathway-specific noise on fixed-rule tasks
- `gee_wcst_noise.py` — pathway-specific noise on WCST + perseveration analysis

### Deeper analysis (§4.4)
- `representation_analysis.py` — t-SNE / silhouette of GRU hidden states
- `error_analysis.py` — confusion matrices, recovery trajectories
- `nonlinearity_analysis.py` — superadditive vs. subadditive damage

### Discussion experiments (§5)
- `gee_wcst_feedback.py` — feedback-only WCST (§5.2.3)
- `gee_gain_experiment.py` — gain modulation, Cohen & Servan-Schreiber 1992 (§5.3)
- `gee_wcst_dropout_ablation.py` — dropout-rate sweep (§5.4)

### Plotting and orchestration
- `run_seeds.py` — multi-seed driver (calls each experiment with seeds 42, 123, 456, 789, 1024)
- `plot_multiseed.py` — assembles main figures from multi-seed JSON outputs
- `plot_dropout_effect.py` — Figure 6

---

## Reproducing a specific figure

To regenerate, e.g., **Figure 10** (the crossover) end-to-end:

```bash
python run_seeds.py --experiment noise --seeds 42,123,456,789,1024
python run_seeds.py --experiment wcst_noise --seeds 42,123,456,789,1024
python plot_multiseed.py --figure figure11_noise
```

Each experiment script can also be run standalone for a single seed (defaults to seed 42). Check the `--help` output of any script for options.

---

## Documentation

- `PROJECT_LOG.md` — dated experiment log with raw numerical results from each run, including the multi-seed validation.

---

## Citation

If you use this code or build on this work, please cite the thesis:

```
Kumar, R. (2026). Two Signatures of Context Failure: A Computational Framework
for Disambiguating WCST Perseveration. Senior thesis, Princeton University.
```

---

## License

MIT — see `LICENSE`.

## Acknowledgments

This work builds directly on the ESBN architecture of Webb, Sinha, and Cohen (2021) and the EGO architecture of Giallanza, Campbell, and Cohen (2024). Thanks to Jon Cohen for advising this thesis.
