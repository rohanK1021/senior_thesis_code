"""
Generate architecture diagrams for ESBN and GEE models.
Run: python generate_architecture_diagrams.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colors ──
BLUE = '#4A90D9'
DARK_BLUE = '#2C6EA4'
PURPLE = '#9B59B6'
ORANGE = '#E67E22'
GREEN = '#27AE60'
RED = '#E74C3C'
GRAY = '#7F8C8D'
LIGHT_GRAY = '#F0F3F5'
TEAL = '#1ABC9C'
DARK = '#2C3E50'
MEMORY_COLOR = '#D5E8D4'
GATE_COLOR = '#FFF2CC'
GRU_COLOR = '#DAE8FC'
ENCODER_COLOR = '#E1D5E7'


def box(ax, cx, cy, w, h, text, color=BLUE, fontsize=10,
        text_color='white', ls='-', lw=1.5, multiline=False):
    """Draw a rounded box with centered text."""
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle="round,pad=0.08", facecolor=color,
                        edgecolor=DARK, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(b)
    va = 'center'
    fs = fontsize
    if multiline:
        fs = fontsize - 1
    ax.text(cx, cy, text, ha='center', va=va, fontsize=fs,
            fontweight='bold', color=text_color, zorder=3,
            linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.8, style='-|>',
          rad=0.0, ls='-', zorder=1):
    """Draw an arrow between two points."""
    cs = f'arc3,rad={rad}' if rad != 0 else 'arc3,rad=0'
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                         connectionstyle=cs, mutation_scale=18,
                         lw=lw, color=color, linestyle=ls, zorder=zorder)
    ax.add_patch(a)


def label(ax, x, y, text, fontsize=9, color=DARK, style='italic',
          ha='center', va='center', weight='normal', rotation=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontstyle=style, fontweight=weight, zorder=4, rotation=rotation)


# =====================================================================
# ESBN Architecture Diagram
# =====================================================================

def draw_esbn():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Title
    ax.text(6.5, 10.0, 'ESBN: Emergent Symbol Binding Network',
            ha='center', va='center', fontsize=18, fontweight='bold', color=DARK)
    ax.text(6.5, 9.5, 'Webb et al. 2021 (GRU variant) — One timestep shown',
            ha='center', va='center', fontsize=11, color=GRAY, fontstyle='italic')

    # ── INPUT SECTION (top) ──
    box(ax, 1.5, 8.3, 1.6, 0.7, 'Input\nx_t', color='#BDC3C7', text_color=DARK, fontsize=9, multiline=True)
    box(ax, 4.5, 8.3, 2.2, 0.7, 'Encoder\n(per-attribute)', color=ENCODER_COLOR, text_color=DARK, fontsize=9, multiline=True)
    box(ax, 8.0, 8.3, 2.4, 0.7, 'Context Norm\n(full-sequence \u03bc/\u03c3)', color=ENCODER_COLOR, text_color=DARK, fontsize=9, multiline=True)

    arrow(ax, 2.3, 8.3, 3.4, 8.3, color=DARK)
    arrow(ax, 5.6, 8.3, 6.8, 8.3, color=DARK)
    label(ax, 3.0, 8.6, '12-dim\none-hot', fontsize=7, color=GRAY)
    label(ax, 6.2, 8.6, '128-dim', fontsize=7, color=GRAY)

    # z_t output label
    box(ax, 10.8, 8.3, 1.0, 0.6, 'z_t', color=TEAL, text_color='white', fontsize=11)
    arrow(ax, 9.2, 8.3, 10.3, 8.3, color=DARK)
    label(ax, 9.7, 8.6, '128-dim', fontsize=7, color=GRAY)

    # ── MEMORY SECTION (middle right) ──
    # Memory background
    mem_bg = FancyBboxPatch((8.8, 4.2), 3.5, 3.3, boxstyle="round,pad=0.15",
                             facecolor=MEMORY_COLOR, edgecolor=GREEN,
                             linewidth=2, linestyle='-', zorder=1, alpha=0.5)
    ax.add_patch(mem_bg)
    ax.text(10.55, 7.25, 'External Memory', ha='center', fontsize=11,
            fontweight='bold', color=GREEN)

    box(ax, 10.0, 6.4, 1.6, 0.7, 'M_v\n(values)', color='#82C46C',
        text_color=DARK, fontsize=9, multiline=True)
    box(ax, 10.0, 5.2, 1.6, 0.7, 'M_k\n(keys)', color='#5BAD3E',
        text_color='white', fontsize=9, multiline=True)

    # z_t → write M_v
    arrow(ax, 10.8, 7.7, 10.0, 6.75, color=TEAL, lw=2)
    label(ax, 11.1, 7.2, 'write\nvalue', fontsize=7, color=TEAL, ha='left')

    # z_t → query M_v (similarity)
    arrow(ax, 10.8, 7.7, 10.8, 6.75, color=RED, lw=2, rad=-0.1)
    label(ax, 11.7, 7.1, 'query\n(z_t \u00b7 M_v)', fontsize=7, color=RED, ha='left')

    # Attention weights
    box(ax, 11.5, 5.8, 1.3, 0.5, 'softmax\n(w_k)', color=GATE_COLOR,
        text_color=DARK, fontsize=8, multiline=True)

    # Retrieved key
    arrow(ax, 10.0, 4.85, 10.0, 4.0, color='#5BAD3E', lw=2)
    label(ax, 9.2, 4.4, 'retrieve\nweighted\nM_k', fontsize=7, color='#5BAD3E', ha='center')

    # ── CONFIDENCE + GATE (middle) ──
    box(ax, 7.0, 5.8, 1.8, 0.6, 'Confidence\n(per-entry \u03c3)', color=GATE_COLOR,
        text_color=DARK, fontsize=8, multiline=True)
    box(ax, 7.0, 4.8, 1.8, 0.6, 'Scalar Gate\ng = \u03c3(W\u00b7h)',
        color=GATE_COLOR, text_color=DARK, fontsize=8, multiline=True)

    # Arrows: M_k → confidence → gate → key_r
    arrow(ax, 9.2, 5.2, 7.9, 5.8, color=DARK)
    arrow(ax, 7.0, 5.5, 7.0, 5.1, color=DARK)

    # key_r output
    box(ax, 5.0, 4.0, 1.4, 0.6, 'key_r', color='#F39C12',
        text_color='white', fontsize=11)
    arrow(ax, 7.0, 4.5, 5.7, 4.0, color=DARK, lw=2)
    label(ax, 6.5, 4.5, 'gated\nretrieval', fontsize=7, color=DARK)

    # ── GRU CONTROLLER (bottom) ──
    gru_bg = FancyBboxPatch((1.5, 1.0), 6.0, 2.2, boxstyle="round,pad=0.15",
                             facecolor=GRU_COLOR, edgecolor=BLUE,
                             linewidth=2, alpha=0.4, zorder=1)
    ax.add_patch(gru_bg)
    ax.text(4.5, 3.0, 'GRU Controller', ha='center', fontsize=11,
            fontweight='bold', color=DARK_BLUE)

    box(ax, 4.5, 2.2, 2.4, 0.8, 'GRU\n(hidden=512)', color=BLUE,
        text_color='white', fontsize=10, multiline=True)

    # key_r → GRU input
    arrow(ax, 5.0, 3.7, 5.0, 2.6, color='#F39C12', lw=2.5)
    label(ax, 5.6, 3.3, 'key_r\n(257-dim)', fontsize=8, color='#F39C12', ha='left')

    # GRU → key_w (write key)
    box(ax, 8.5, 2.2, 1.6, 0.6, 'key_w\n(ReLU)', color='#5BAD3E',
        text_color='white', fontsize=9, multiline=True)
    arrow(ax, 5.7, 2.2, 7.7, 2.2, color=DARK)
    label(ax, 6.7, 2.5, '256-dim', fontsize=7, color=GRAY)

    # key_w → write M_k
    arrow(ax, 8.5, 2.8, 10.0, 4.85, color='#5BAD3E', lw=2, rad=-0.2)
    label(ax, 8.6, 3.8, 'write\nkey', fontsize=7, color='#5BAD3E', ha='left')

    # GRU → output
    box(ax, 2.5, 0.5, 1.8, 0.6, 'Output y\n(linear)', color=DARK_BLUE,
        text_color='white', fontsize=9, multiline=True)
    arrow(ax, 4.5, 1.8, 2.5, 1.1, color=DARK)

    # Previous key_r input (from left)
    arrow(ax, 0.5, 2.2, 3.3, 2.2, color='#F39C12', lw=2, style='-|>')
    label(ax, 0.8, 2.6, 'key_r\n(from t-1)', fontsize=8, color='#F39C12', ha='left')

    # ── INDIRECTION CALLOUT ──
    # Big red X between z_t and GRU
    ax.plot([9.5, 10.5], [3.3, 2.7], 'x', color=RED, markersize=0)

    # Barrier line
    barrier_y = [3.5, 3.5]
    barrier_x = [1.0, 8.0]
    # ax.plot(barrier_x, barrier_y, '--', color=RED, lw=1.5, alpha=0.5)

    # Indirection annotation
    indirection_box = FancyBboxPatch((0.0, -0.3), 7.5, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#FDEDEC', edgecolor=RED,
                                      linewidth=2, zorder=2, alpha=0.9)
    ax.add_patch(indirection_box)
    ax.text(3.75, 0.1, 'INDIRECTION: GRU receives abstract keys (key_r), '
            'NEVER raw encoded input (z_t)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=RED, zorder=3)

    # Step numbers
    for i, (x, y, txt) in enumerate([
        (1.5, 9.0, '1. Encode'),
        (8.0, 9.0, '2. Normalize'),
        (10.8, 7.0, '3. Write z_t to M_v'),
        (12.2, 5.4, '4. Query M_v with z_t'),
        (9.0, 3.8, '5. Retrieve keys from M_k'),
        (5.8, 3.3, '6. Gate retrieval'),
        (2.5, 3.0, '7. Feed key_r to GRU'),
        (8.5, 1.6, '8. Write key_w to M_k'),
    ], 1):
        ax.text(x, y, txt, fontsize=7, color=GRAY, ha='left',
                fontstyle='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig('fig_diagram_esbn.png', bbox_inches='tight', facecolor='white', dpi=200)
    print("Saved: fig_diagram_esbn.png")
    plt.close()


# =====================================================================
# GEE Architecture Diagram
# =====================================================================

def draw_gee():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1.5, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7.5, 10.5, 'GEE: GRU-ESBN-EGO Unified Architecture',
            ha='center', fontsize=18, fontweight='bold', color=DARK)
    ax.text(7.5, 10.0, 'Context enters via h\u2080 and RCM only — '
            'keys and memory are NEVER touched',
            ha='center', fontsize=11, color=GRAY, fontstyle='italic')

    # ── ESBN CORE (center, as dashed box) ──
    esbn_bg = FancyBboxPatch((3.5, 1.5), 8.5, 7.0, boxstyle="round,pad=0.2",
                              facecolor='#F8F9FA', edgecolor=BLUE,
                              linewidth=2.5, linestyle='--', zorder=0, alpha=0.6)
    ax.add_patch(esbn_bg)
    ax.text(7.75, 8.2, 'ESBN Core (Webb-faithful, untouched)',
            ha='center', fontsize=12, fontweight='bold', color=BLUE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=BLUE, alpha=0.9))

    # Input
    box(ax, 2.0, 7.2, 1.6, 0.7, 'Input\nx_t', color='#BDC3C7',
        text_color=DARK, fontsize=9, multiline=True)

    # Encoder + ContextNorm
    box(ax, 5.0, 7.2, 2.2, 0.7, 'Encoder +\nContextNorm', color=ENCODER_COLOR,
        text_color=DARK, fontsize=9, multiline=True)
    arrow(ax, 2.8, 7.2, 3.9, 7.2, color=DARK)

    # z_t
    box(ax, 7.5, 7.2, 0.9, 0.6, 'z_t', color=TEAL, text_color='white', fontsize=11)
    arrow(ax, 6.1, 7.2, 7.05, 7.2, color=DARK)

    # Memory
    mem_bg = FancyBboxPatch((8.8, 4.5), 2.8, 2.8, boxstyle="round,pad=0.12",
                             facecolor=MEMORY_COLOR, edgecolor=GREEN,
                             linewidth=2, zorder=1, alpha=0.5)
    ax.add_patch(mem_bg)
    ax.text(10.2, 7.05, 'Memory', ha='center', fontsize=10,
            fontweight='bold', color=GREEN)

    box(ax, 10.2, 6.3, 1.4, 0.6, 'M_v', color='#82C46C',
        text_color=DARK, fontsize=10)
    box(ax, 10.2, 5.3, 1.4, 0.6, 'M_k', color='#5BAD3E',
        text_color='white', fontsize=10)

    # z_t → M_v (write + query)
    arrow(ax, 7.95, 7.2, 9.5, 6.6, color=TEAL, lw=2)
    label(ax, 8.3, 7.0, 'write/query', fontsize=7, color=TEAL)

    # Retrieval path
    box(ax, 8.0, 5.3, 1.2, 0.5, 'Gate', color=GATE_COLOR,
        text_color=DARK, fontsize=9)
    arrow(ax, 9.5, 5.3, 8.6, 5.3, color=DARK)
    label(ax, 9.0, 5.6, 'retrieve', fontsize=7, color=DARK)

    # key_r
    box(ax, 6.5, 4.5, 1.2, 0.5, 'key_r', color='#F39C12',
        text_color='white', fontsize=10)
    arrow(ax, 8.0, 5.0, 7.1, 4.7, color=DARK, lw=2)

    # GRU Controller
    box(ax, 6.5, 3.2, 2.4, 0.9, 'GRU\nController', color=BLUE,
        text_color='white', fontsize=11, multiline=True)
    arrow(ax, 6.5, 4.25, 6.5, 3.65, color='#F39C12', lw=2.5)

    # GRU → key_w → M_k
    box(ax, 9.5, 3.2, 1.2, 0.5, 'key_w', color='#5BAD3E',
        text_color='white', fontsize=9)
    arrow(ax, 7.7, 3.2, 8.9, 3.2, color=DARK)
    arrow(ax, 9.5, 3.45, 10.2, 5.0, color='#5BAD3E', lw=1.5, rad=-0.2)
    label(ax, 10.8, 4.2, 'write\nkey', fontsize=7, color='#5BAD3E')

    # GRU → output (before RCM addition)
    box(ax, 6.5, 2.0, 1.8, 0.6, 'Output\nlogits', color=DARK_BLUE,
        text_color='white', fontsize=9, multiline=True)
    arrow(ax, 6.5, 2.75, 6.5, 2.3, color=DARK)

    # ══════════════════════════════════════════════
    # h_0 PATHWAY (left side, purple)
    # ══════════════════════════════════════════════
    h0_bg = FancyBboxPatch((-0.5, 2.0), 3.2, 4.5, boxstyle="round,pad=0.15",
                            facecolor='#F4ECF7', edgecolor=PURPLE,
                            linewidth=2.5, zorder=0, alpha=0.6)
    ax.add_patch(h0_bg)
    ax.text(1.1, 6.2, 'h\u2080 Pathway', ha='center', fontsize=12,
            fontweight='bold', color=PURPLE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=PURPLE, alpha=0.9))

    # Task ID input
    box(ax, 1.1, 5.3, 1.8, 0.6, 'task_id\n(one-hot)', color='#D2B4DE',
        text_color=DARK, fontsize=9, multiline=True)

    # h0_proj
    box(ax, 1.1, 4.3, 2.0, 0.6, 'Linear + tanh', color=PURPLE,
        text_color='white', fontsize=9)
    arrow(ax, 1.1, 5.0, 1.1, 4.6, color=PURPLE, lw=2)

    # h_0 output
    box(ax, 1.1, 3.2, 1.4, 0.6, 'h\u2080', color=PURPLE,
        text_color='white', fontsize=13)
    arrow(ax, 1.1, 4.0, 1.1, 3.5, color=PURPLE, lw=2)

    # h_0 → GRU initial state
    arrow(ax, 1.8, 3.2, 5.3, 3.2, color=PURPLE, lw=2.5)
    label(ax, 3.5, 3.5, 'initial hidden state', fontsize=8,
          color=PURPLE, style='italic')

    # h0 description
    ax.text(1.1, 2.5, 'Sets cognitive\n"mindset" for\nthe sequence',
            ha='center', fontsize=8, color=PURPLE, fontstyle='italic',
            linespacing=1.3)

    # ══════════════════════════════════════════════
    # RCM PATHWAY (right side, orange)
    # ══════════════════════════════════════════════
    rcm_bg = FancyBboxPatch((12.5, 1.0), 3.0, 6.8, boxstyle="round,pad=0.15",
                             facecolor='#FDF2E9', edgecolor=ORANGE,
                             linewidth=2.5, zorder=0, alpha=0.6)
    ax.add_patch(rcm_bg)
    ax.text(14.0, 7.5, 'RCM Pathway', ha='center', fontsize=12,
            fontweight='bold', color=ORANGE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=ORANGE, alpha=0.9))

    # RCM inputs
    box(ax, 14.0, 6.5, 1.8, 0.6, 'x_raw +\ntask_id', color='#F5CBA7',
        text_color=DARK, fontsize=9, multiline=True)

    # y_prev
    box(ax, 14.0, 5.5, 1.6, 0.5, 'y_prev', color='#F5CBA7',
        text_color=DARK, fontsize=9)

    # RCM GRU
    box(ax, 14.0, 4.5, 2.0, 0.7, 'RCM\n(GRUCell)', color=ORANGE,
        text_color='white', fontsize=10, multiline=True)
    arrow(ax, 14.0, 6.2, 14.0, 4.85, color=ORANGE, lw=2)
    arrow(ax, 14.0, 5.25, 14.0, 4.85, color=ORANGE, lw=1.5)

    # c_t
    box(ax, 14.0, 3.5, 1.2, 0.5, 'c_t', color=ORANGE,
        text_color='white', fontsize=11)
    arrow(ax, 14.0, 4.15, 14.0, 3.75, color=ORANGE, lw=2)

    # c_t recurrence
    arrow(ax, 14.8, 3.5, 15.0, 4.5, color=ORANGE, lw=1.5, rad=-0.5)
    label(ax, 15.2, 4.0, 'recurrent', fontsize=7, color=ORANGE)

    # ctx_to_out
    box(ax, 14.0, 2.5, 2.0, 0.5, 'Linear\n(ctx\u2192out)', color=ORANGE,
        text_color='white', fontsize=8, multiline=True)
    arrow(ax, 14.0, 3.25, 14.0, 2.75, color=ORANGE, lw=2)

    # RCM bias → output
    box(ax, 10.5, 2.0, 1.6, 0.6, '+  bias', color='#FDEBD0',
        text_color=DARK, fontsize=10)
    arrow(ax, 13.0, 2.5, 11.3, 2.0, color=ORANGE, lw=2.5)
    arrow(ax, 7.4, 2.0, 9.7, 2.0, color=DARK, lw=1.5)

    # Final output
    box(ax, 10.5, 0.8, 1.8, 0.6, 'Final\nOutput y', color=DARK_BLUE,
        text_color='white', fontsize=10, multiline=True)
    arrow(ax, 10.5, 1.7, 10.5, 1.1, color=DARK, lw=2)

    # RCM description
    ax.text(14.0, 1.5, 'Tracks context\ndynamically\nacross trials',
            ha='center', fontsize=8, color=ORANGE, fontstyle='italic',
            linespacing=1.3)

    # ══════════════════════════════════════════════
    # ANNOTATIONS
    # ══════════════════════════════════════════════

    # Task_id dropout annotation
    box(ax, 1.1, 7.2, 2.2, 0.7, 'Task ID\n(or dropped)', color='#D2B4DE',
        text_color=DARK, fontsize=9, multiline=True)
    arrow(ax, 2.2, 7.2, 3.0, 7.2, color=PURPLE, lw=1.5, ls='--')
    label(ax, 2.6, 6.7, 'dropout\np=0.3', fontsize=7, color=PURPLE)

    # Input arrow to RCM
    arrow(ax, 2.8, 7.5, 13.5, 7.5, color=GRAY, lw=1, ls='--')
    arrow(ax, 13.5, 7.5, 14.0, 6.8, color=ORANGE, lw=1.5)

    # Key constraint annotation
    constraint_box = FancyBboxPatch((3.0, -1.2), 9.5, 0.9,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FDEDEC', edgecolor=RED,
                                     linewidth=2, zorder=2, alpha=0.9)
    ax.add_patch(constraint_box)
    ax.text(7.75, -0.75, 'CONSTRAINT: Context (h\u2080 + RCM) NEVER enters '
            'keys or memory. The indirection pathway is pure ESBN.',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=RED, zorder=3)

    plt.tight_layout()
    plt.savefig('fig_diagram_gee.png', bbox_inches='tight', facecolor='white', dpi=200)
    print("Saved: fig_diagram_gee.png")
    plt.close()


# =====================================================================
# Comparison / Summary Diagram
# =====================================================================

def draw_comparison():
    """Show the three architectures side by side: ESBN, GEE-Direct, GEE-Separated."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax in axes:
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 8.5)
        ax.axis('off')
        ax.set_aspect('equal')

    titles = ['ESBN (No Context)', 'GEE-Direct\n(Context in Keys)', 'GEE-Separated\n(Context at h\u2080 + Output)']
    colors = [BLUE, RED, GREEN]
    results = ['98.4% with ID\n98.4% without\n0.0% drop',
               '97.5% with ID\n61.0% without\n-36.6% drop',
               '98.1% with ID\n98.2% without\n-0.1% drop']
    verdicts = ['Baseline', 'CATASTROPHIC\nDEPENDENCY', 'BEST OF\nBOTH WORLDS']
    verdict_colors = [GRAY, RED, GREEN]

    for idx, ax in enumerate(axes):
        c = colors[idx]

        # Title
        ax.text(3.0, 8.0, titles[idx], ha='center', fontsize=13,
                fontweight='bold', color=c, linespacing=1.3)

        # Common components
        # Input
        box(ax, 1.5, 6.5, 1.4, 0.5, 'Input x_t', color='#BDC3C7',
            text_color=DARK, fontsize=8)
        # Encoder
        box(ax, 1.5, 5.5, 1.4, 0.5, 'Encoder', color=ENCODER_COLOR,
            text_color=DARK, fontsize=8)
        arrow(ax, 1.5, 6.25, 1.5, 5.75, color=DARK, lw=1.2)

        # z_t
        box(ax, 1.5, 4.5, 0.8, 0.4, 'z_t', color=TEAL,
            text_color='white', fontsize=9)
        arrow(ax, 1.5, 5.25, 1.5, 4.7, color=DARK, lw=1.2)

        # Memory
        box(ax, 4.5, 5.5, 1.2, 0.5, 'M_v', color='#82C46C',
            text_color=DARK, fontsize=9)
        box(ax, 4.5, 4.5, 1.2, 0.5, 'M_k', color='#5BAD3E',
            text_color='white', fontsize=9)
        arrow(ax, 1.9, 4.5, 3.9, 5.5, color=TEAL, lw=1.2)
        arrow(ax, 4.5, 4.25, 3.5, 3.8, color=DARK, lw=1.2)

        # Gate
        box(ax, 3.0, 3.5, 1.0, 0.4, 'Gate', color=GATE_COLOR,
            text_color=DARK, fontsize=8)

        # GRU
        box(ax, 3.0, 2.5, 1.6, 0.6, 'GRU', color=BLUE,
            text_color='white', fontsize=10)
        arrow(ax, 3.0, 3.3, 3.0, 2.8, color=DARK, lw=1.5)

        # Output
        box(ax, 3.0, 1.3, 1.4, 0.5, 'Output y', color=DARK_BLUE,
            text_color='white', fontsize=9)
        arrow(ax, 3.0, 2.2, 3.0, 1.55, color=DARK, lw=1.2)

        # GRU → key_w → M_k
        arrow(ax, 3.8, 2.5, 4.5, 4.25, color='#5BAD3E', lw=1.2, rad=-0.2)

        # ── Model-specific additions ──
        if idx == 0:
            # ESBN: nothing extra
            ax.text(3.0, 0.3, 'No context mechanism.\nPure indirection.',
                    ha='center', fontsize=9, color=GRAY, fontstyle='italic',
                    linespacing=1.3)

        elif idx == 1:
            # GEE-Direct: task_id in keys + RCM at output
            # Task ID
            box(ax, 5.5, 6.5, 1.2, 0.4, 'task_id', color='#F5B7B1',
                text_color=DARK, fontsize=8)

            # task_id → keys (RED — bad!)
            arrow(ax, 5.5, 6.3, 4.5, 4.75, color=RED, lw=2.5)
            ax.text(5.6, 5.5, 'IN KEYS!', fontsize=9, color=RED,
                    fontweight='bold', rotation=-50)

            # RCM at output
            box(ax, 5.5, 1.3, 1.0, 0.4, 'RCM', color=ORANGE,
                text_color='white', fontsize=8)
            arrow(ax, 5.5, 6.1, 5.5, 1.5, color=ORANGE, lw=1.5, rad=0.3)
            arrow(ax, 5.0, 1.3, 3.7, 1.3, color=ORANGE, lw=1.5)
            label(ax, 4.3, 1.0, '+bias', fontsize=7, color=ORANGE)

        elif idx == 2:
            # GEE-Separated: h_0 + RCM at output
            # Task ID
            box(ax, 5.5, 6.5, 1.2, 0.4, 'task_id', color='#D5F5E3',
                text_color=DARK, fontsize=8)

            # task_id → h_0 (GREEN — good!)
            box(ax, 5.5, 3.5, 0.8, 0.4, 'h\u2080', color=PURPLE,
                text_color='white', fontsize=9)
            arrow(ax, 5.5, 6.3, 5.5, 3.7, color=PURPLE, lw=2, rad=0.2)
            arrow(ax, 5.1, 3.5, 3.8, 2.7, color=PURPLE, lw=2.5)
            label(ax, 4.7, 3.1, 'init\nstate', fontsize=7, color=PURPLE)

            # RCM at output
            box(ax, 5.5, 1.3, 1.0, 0.4, 'RCM', color=ORANGE,
                text_color='white', fontsize=8)
            arrow(ax, 5.5, 6.1, 5.5, 1.5, color=ORANGE, lw=1.2,
                  rad=0.5)
            arrow(ax, 5.0, 1.3, 3.7, 1.3, color=ORANGE, lw=1.5)
            label(ax, 4.3, 1.0, '+bias', fontsize=7, color=ORANGE)

            # Highlight: keys untouched
            ax.text(4.5, 3.9, 'keys\nuntouched', fontsize=7, color=GREEN,
                    fontweight='bold', ha='center', fontstyle='italic')

        # Results box
        result_bg = FancyBboxPatch((0.3, -0.4), 5.4, 0.9,
                                    boxstyle="round,pad=0.1",
                                    facecolor='white',
                                    edgecolor=verdict_colors[idx],
                                    linewidth=2, zorder=2)
        ax.add_patch(result_bg)
        ax.text(1.8, 0.05, results[idx], fontsize=8, color=DARK,
                ha='center', va='center', fontfamily='monospace', linespacing=1.2)
        ax.text(4.5, 0.05, verdicts[idx], fontsize=9, color=verdict_colors[idx],
                ha='center', va='center', fontweight='bold', linespacing=1.2)

    fig.suptitle('Three Architectures: Where Should Context Enter?',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fig_diagram_comparison.png', bbox_inches='tight', facecolor='white', dpi=200)
    print("Saved: fig_diagram_comparison.png")
    plt.close()


if __name__ == '__main__':
    draw_esbn()
    draw_gee()
    draw_comparison()
    print("\nAll architecture diagrams generated.")
