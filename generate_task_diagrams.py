"""
Generate intuitive task visualizations and GEE Unified architecture diagram.
Run: python generate_task_diagrams.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

# ── Colors ──
COLORS = {
    'red': '#E74C3C', 'blue': '#3498DB', 'green': '#27AE60',
    'yellow': '#F1C40F', 'purple': '#9B59B6', 'orange': '#E67E22',
    'gray': '#BDC3C7', 'pink': '#FF69B4',
}
DARK = '#2C3E50'
LIGHT_BG = '#F8F9FA'
MATCH_COLOR = '#27AE60'
NOMATCH_COLOR = '#E74C3C'


def draw_entity(ax, x, y, shape, color, size='large', label_below=None):
    """Draw an entity as a colored geometric shape."""
    r = 0.28 if size == 'large' else 0.19
    c = COLORS.get(color, COLORS['gray'])

    if shape == 'circle':
        patch = Circle((x, y), r, facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    elif shape == 'square':
        patch = Rectangle((x - r, y - r), 2 * r, 2 * r,
                           facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    elif shape == 'triangle':
        h = r * 1.6
        pts = [(x, y + h * 0.55), (x - r, y - h * 0.45), (x + r, y - h * 0.45)]
        patch = Polygon(pts, facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    elif shape == 'diamond':
        pts = [(x, y + r * 1.1), (x - r * 0.75, y), (x, y - r * 1.1), (x + r * 0.75, y)]
        patch = Polygon(pts, facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    elif shape == 'pentagon':
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
        pts = [(x + r * np.cos(a), y + r * np.sin(a)) for a in angles]
        patch = Polygon(pts, facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    elif shape == 'hexagon':
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        pts = [(x + r * np.cos(a), y + r * np.sin(a)) for a in angles]
        patch = Polygon(pts, facecolor=c, edgecolor=DARK, lw=2, zorder=5)
    else:
        patch = Circle((x, y), r, facecolor=c, edgecolor=DARK, lw=2, zorder=5)

    ax.add_patch(patch)

    if label_below:
        ax.text(x, y - r - 0.18, label_below, ha='center', va='top',
                fontsize=7, color='#555', zorder=6)


def bracket(ax, x1, x2, y, label, color=DARK, direction='below', offset=0.4):
    """Draw a horizontal bracket with a label."""
    dy = -offset if direction == 'below' else offset
    lw = 1.5
    ax.plot([x1, x1], [y, y + dy * 0.3], color=color, lw=lw, zorder=3)
    ax.plot([x2, x2], [y, y + dy * 0.3], color=color, lw=lw, zorder=3)
    ax.plot([x1, x2], [y + dy * 0.3, y + dy * 0.3], color=color, lw=lw, zorder=3)
    label_y = y + dy * 0.3 + (dy * 0.25)
    ax.text((x1 + x2) / 2, label_y, label, ha='center', va='center' if direction == 'below' else 'center',
            fontsize=9, color=color, fontweight='bold', zorder=4)


def group_box(ax, x1, y1, x2, y2, color='#3498DB', alpha=0.12, label=None, label_pos='top'):
    """Draw a light background box around a group of entities."""
    w, h = x2 - x1, y2 - y1
    rect = FancyBboxPatch((x1, y1), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor=color,
                           alpha=alpha, lw=1.5, zorder=1)
    ax.add_patch(rect)
    if label:
        ly = y2 + 0.15 if label_pos == 'top' else y1 - 0.2
        ax.text((x1 + x2) / 2, ly, label, ha='center', fontsize=8,
                color=color, fontweight='bold', zorder=4, alpha=0.9)


def result_badge(ax, x, y, text, is_match=True):
    """Draw a match/non-match result badge."""
    c = MATCH_COLOR if is_match else NOMATCH_COLOR
    bg = '#D5F5E3' if is_match else '#FADBD8'
    box = FancyBboxPatch((x - 0.6, y - 0.2), 1.2, 0.4,
                          boxstyle="round,pad=0.06", facecolor=bg,
                          edgecolor=c, lw=2, zorder=5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            fontweight='bold', color=c, zorder=6)


# =====================================================================
# Task 1: Same/Different
# =====================================================================
def draw_same_different(ax):
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')

    # Title
    ax.text(5.25, 3.1, 'Same / Different', fontsize=16, fontweight='bold',
            ha='center', color=DARK)
    ax.text(5.25, 2.6, 'Are these two entities identical?  (seq_len = 2, binary output)',
            ha='center', fontsize=10, color='#666', fontstyle='italic')

    # Match example
    ax.text(0.0, 1.7, 'MATCH:', fontsize=10, fontweight='bold', color=MATCH_COLOR)
    draw_entity(ax, 1.8, 1.5, 'circle', 'red', 'large')
    draw_entity(ax, 3.0, 1.5, 'circle', 'red', 'large')
    group_box(ax, 1.3, 0.9, 3.5, 2.1, color=MATCH_COLOR, alpha=0.08)
    ax.annotate('', xy=(3.8, 1.5), xytext=(3.6, 1.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    result_badge(ax, 4.7, 1.5, 'SAME', is_match=True)
    ax.text(2.4, 0.7, '= ?', fontsize=14, ha='center', color='#555', fontweight='bold')

    # Non-match example
    ax.text(6.0, 1.7, 'NON-MATCH:', fontsize=10, fontweight='bold', color=NOMATCH_COLOR)
    draw_entity(ax, 7.8, 1.5, 'circle', 'red', 'large')
    draw_entity(ax, 9.0, 1.5, 'square', 'blue', 'large')
    group_box(ax, 7.3, 0.9, 9.5, 2.1, color=NOMATCH_COLOR, alpha=0.08)
    ax.annotate('', xy=(9.8, 1.5), xytext=(9.6, 1.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    result_badge(ax, 10.5, 1.5, 'DIFF', is_match=False)
    ax.text(8.4, 0.7, '\u2260 ?', fontsize=14, ha='center', color='#555', fontweight='bold')

    # Position labels
    for x, lbl in [(1.8, 'pos 0'), (3.0, 'pos 1')]:
        ax.text(x, 0.95, lbl, ha='center', fontsize=7, color='#999')
    for x, lbl in [(7.8, 'pos 0'), (9.0, 'pos 1')]:
        ax.text(x, 0.95, lbl, ha='center', fontsize=7, color='#999')


# =====================================================================
# Task 2: RMTS (Relational Match-to-Sample)
# =====================================================================
def draw_rmts(ax):
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2.0, 3.5)
    ax.axis('off')

    ax.text(7.0, 3.1, 'Relational Match-to-Sample (RMTS)', fontsize=16,
            fontweight='bold', ha='center', color=DARK)
    ax.text(7.0, 2.6, 'Which target pair has the same relation (same/different) as the source?  '
            '(seq_len = 6, binary)',
            ha='center', fontsize=10, color='#666', fontstyle='italic')

    # Source pair (SAME relation)
    ax.text(0.0, 1.8, 'Source:', fontsize=10, fontweight='bold', color='#555')
    draw_entity(ax, 1.5, 1.5, 'circle', 'red', 'large')
    draw_entity(ax, 2.7, 1.5, 'circle', 'red', 'large')
    group_box(ax, 1.0, 0.85, 3.2, 2.15, color='#3498DB', alpha=0.1, label='SAME relation')
    bracket(ax, 1.5, 2.7, 0.85, 'same', color='#3498DB', direction='below', offset=0.35)

    # Arrow
    ax.annotate('', xy=(4.5, 1.5), xytext=(3.5, 1.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(4.0, 1.85, 'which\nmatches?', ha='center', fontsize=8, color='#555')

    # Target 1 (SAME - correct)
    ax.text(5.0, 1.8, 'Target 1:', fontsize=9, fontweight='bold', color=MATCH_COLOR)
    draw_entity(ax, 6.2, 1.5, 'square', 'blue', 'large')
    draw_entity(ax, 7.4, 1.5, 'square', 'blue', 'large')
    group_box(ax, 5.7, 0.85, 7.9, 2.15, color=MATCH_COLOR, alpha=0.1)
    bracket(ax, 6.2, 7.4, 0.85, 'same \u2713', color=MATCH_COLOR, direction='below', offset=0.35)

    # Target 2 (DIFFERENT - wrong)
    ax.text(8.5, 1.8, 'Target 2:', fontsize=9, fontweight='bold', color='#999')
    draw_entity(ax, 9.7, 1.5, 'triangle', 'green', 'large')
    draw_entity(ax, 10.9, 1.5, 'diamond', 'yellow', 'large')
    group_box(ax, 9.2, 0.85, 11.4, 2.15, color='#999', alpha=0.06)
    bracket(ax, 9.7, 10.9, 0.85, 'diff \u2717', color='#999', direction='below', offset=0.35)

    # Answer
    result_badge(ax, 12.8, 1.5, 'Target 1', is_match=True)
    ax.annotate('', xy=(12.1, 1.5), xytext=(11.6, 1.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))

    # Key insight
    ax.text(7.0, -1.2, 'The model must compare relations, not entities.\n'
            'Different entities, same abstract structure.',
            ha='center', fontsize=9, color='#888', fontstyle='italic', linespacing=1.4)


# =====================================================================
# Task 3: Distribution-of-3
# =====================================================================
def draw_dist3(ax):
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2.3, 3.8)
    ax.axis('off')

    ax.text(7.0, 3.5, 'Distribution of Three', fontsize=16,
            fontweight='bold', ha='center', color=DARK)
    ax.text(7.0, 3.0, 'Find the shared attribute in Row 1, then complete Row 2.  '
            '(seq_len = 9, 4-way output)',
            ha='center', fontsize=10, color='#666', fontstyle='italic')

    # Row 1: all RED (different shapes)
    ax.text(0.0, 2.2, 'Row 1:', fontsize=10, fontweight='bold', color='#555')
    ax.text(0.0, 1.85, '(all share\nan attribute)', fontsize=7, color='#888',
            fontstyle='italic', linespacing=1.2)
    draw_entity(ax, 1.8, 2.1, 'circle', 'red', 'large')
    draw_entity(ax, 3.0, 2.1, 'square', 'red', 'large')
    draw_entity(ax, 4.2, 2.1, 'triangle', 'red', 'large')
    group_box(ax, 1.3, 1.45, 4.7, 2.75, color=COLORS['red'], alpha=0.1,
              label='all RED', label_pos='top')

    # Row 2: two red, one missing
    ax.text(0.0, 0.7, 'Row 2:', fontsize=10, fontweight='bold', color='#555')
    ax.text(0.0, 0.35, '(incomplete)', fontsize=7, color='#888', fontstyle='italic')
    draw_entity(ax, 1.8, 0.6, 'circle', 'red', 'large')
    draw_entity(ax, 3.0, 0.6, 'square', 'red', 'large')
    # Question mark for missing
    ax.text(4.2, 0.6, '?', fontsize=24, ha='center', va='center',
            color=COLORS['red'], fontweight='bold', zorder=5)
    question_circle = Circle((4.2, 0.6), 0.35, facecolor='#FADBD8',
                              edgecolor=COLORS['red'], lw=2, ls='--', zorder=4)
    ax.add_patch(question_circle)

    # Arrow to choices
    ax.annotate('', xy=(5.8, 0.6), xytext=(4.7, 0.6),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    # 4 choices
    ax.text(8.0, 1.6, 'Choices:', fontsize=10, fontweight='bold', color='#555')
    choices_x = [6.5, 8.0, 9.5, 11.0]
    choices = [
        ('diamond', 'blue', False),
        ('triangle', 'red', True),    # correct
        ('square', 'green', False),
        ('diamond', 'yellow', False),
    ]
    for i, (s, c, correct) in enumerate(choices):
        draw_entity(ax, choices_x[i], 0.6, s, c, 'large')
        if correct:
            # Highlight correct
            highlight = Circle((choices_x[i], 0.6), 0.42, facecolor='none',
                               edgecolor=MATCH_COLOR, lw=3, ls='-', zorder=6)
            ax.add_patch(highlight)
            ax.text(choices_x[i], -0.15, '\u2713 RED', fontsize=9, ha='center',
                    color=MATCH_COLOR, fontweight='bold')
        else:
            ax.text(choices_x[i], -0.15, '\u2717', fontsize=9, ha='center', color='#CCC')

    # Answer labels
    for i, lbl in enumerate(['0', '1', '2', '3']):
        ax.text(choices_x[i], 1.2, f'[{lbl}]', ha='center', fontsize=8, color='#AAA')

    # Key insight
    ax.text(7.0, -1.5, 'Must discover which attribute (color/shape/size/texture) is shared,\n'
            'then find the entity that has that same attribute value.',
            ha='center', fontsize=9, color='#888', fontstyle='italic', linespacing=1.4)


# =====================================================================
# Task 4: Identity Rules
# =====================================================================
def draw_identity_rules(ax):
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2.3, 3.8)
    ax.axis('off')

    ax.text(7.0, 3.5, 'Identity Rules', fontsize=16,
            fontweight='bold', ha='center', color=DARK)
    ax.text(7.0, 3.0, 'Infer the pattern rule (ABA or ABB) from the example, '
            'then apply it.  (seq_len = 9, 4-way)',
            ha='center', fontsize=10, color='#666', fontstyle='italic')

    # Example triple: ABA pattern
    ax.text(0.0, 2.2, 'Example:', fontsize=10, fontweight='bold', color='#555')
    draw_entity(ax, 1.5, 2.1, 'circle', 'red', 'large')
    draw_entity(ax, 2.7, 2.1, 'square', 'blue', 'large')
    draw_entity(ax, 3.9, 2.1, 'circle', 'red', 'large')

    # ABA labels
    for x, lbl in [(1.5, 'A'), (2.7, 'B'), (3.9, 'A')]:
        ax.text(x, 1.45, lbl, ha='center', fontsize=12, fontweight='bold',
                color='#9B59B6')
    group_box(ax, 1.0, 1.45, 4.4, 2.75, color='#9B59B6', alpha=0.1,
              label='ABA pattern', label_pos='top')

    # Arrow: infer rule
    ax.annotate('', xy=(5.3, 2.1), xytext=(4.6, 2.1),
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2))
    ax.text(4.95, 2.5, 'infer\nrule', ha='center', fontsize=8, color='#9B59B6')

    # Query pair
    ax.text(5.5, 2.2, 'Query:', fontsize=10, fontweight='bold', color='#555')
    draw_entity(ax, 6.5, 2.1, 'triangle', 'green', 'large')
    draw_entity(ax, 7.7, 2.1, 'diamond', 'yellow', 'large')
    ax.text(8.9, 2.1, '?', fontsize=24, ha='center', va='center',
            color='#9B59B6', fontweight='bold', zorder=5)
    question_circle = Circle((8.9, 2.1), 0.35, facecolor='#F4ECF7',
                              edgecolor='#9B59B6', lw=2, ls='--', zorder=4)
    ax.add_patch(question_circle)

    # Query labels
    for x, lbl in [(6.5, 'A'), (7.7, 'B'), (8.9, '?')]:
        ax.text(x, 1.45, lbl, ha='center', fontsize=12, fontweight='bold',
                color='#9B59B6')

    # Arrow to choices
    ax.annotate('', xy=(6.5, 0.8), xytext=(7.5, 1.3),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(7.5, 0.85, 'apply ABA:\n3rd = 1st', ha='center', fontsize=8, color='#555')

    # 4 choices
    ax.text(5.0, 0.8, 'Choices:', fontsize=10, fontweight='bold', color='#555')
    choices_x = [6.0, 7.5, 9.0, 10.5]
    choices = [
        ('triangle', 'green', True),     # correct (ABA → repeat first)
        ('diamond', 'yellow', False),     # ABB answer
        ('pentagon', 'purple', False),
        ('hexagon', 'orange', False),
    ]
    for i, (s, c, correct) in enumerate(choices):
        draw_entity(ax, choices_x[i], 0.0, s, c, 'large')
        if correct:
            highlight = Circle((choices_x[i], 0.0), 0.42, facecolor='none',
                               edgecolor=MATCH_COLOR, lw=3, zorder=6)
            ax.add_patch(highlight)
            ax.text(choices_x[i], -0.65, '\u2713 = A', fontsize=9, ha='center',
                    color=MATCH_COLOR, fontweight='bold')
        else:
            ax.text(choices_x[i], -0.65, '\u2717', fontsize=9, ha='center', color='#CCC')

    # Key insight
    ax.text(7.0, -1.5, 'Must learn abstract pattern rules (ABA, ABB, AAA)\nand apply them '
            'to completely novel entities.',
            ha='center', fontsize=9, color='#888', fontstyle='italic', linespacing=1.4)


# =====================================================================
# Task 5: WCST
# =====================================================================
def draw_wcst(ax):
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2.5, 3.8)
    ax.axis('off')

    ax.text(7.0, 3.5, 'Wisconsin Card Sorting Task (WCST)', fontsize=16,
            fontweight='bold', ha='center', color=DARK)
    ax.text(7.0, 3.0, 'Sort by the active rule. Same cards, different answers!  '
            '(seq_len = 2, binary, rules switch)',
            ha='center', fontsize=10, color='#666', fontstyle='italic')

    # The two cards
    ax.text(0.2, 2.1, 'Reference:', fontsize=10, fontweight='bold', color='#555')
    draw_entity(ax, 2.0, 2.1, 'circle', 'red', 'large')
    ax.text(2.0, 1.45, 'large red circle', ha='center', fontsize=7, color='#888')

    ax.text(3.5, 2.1, 'Test:', fontsize=10, fontweight='bold', color='#555')
    draw_entity(ax, 4.8, 2.1, 'square', 'red', 'large')
    ax.text(4.8, 1.45, 'large red square', ha='center', fontsize=7, color='#888')

    # Four rules with answers
    rules_data = [
        ('COLOR',   'red = red',       True,  COLORS['red']),
        ('SHAPE',   'circle \u2260 square', False, COLORS['blue']),
        ('SIZE',    'large = large',   True,  COLORS['green']),
        ('TEXTURE', '(same texture)',  True,  COLORS['purple']),
    ]

    ax.text(7.0, 2.3, 'Active Rule:', fontsize=11, fontweight='bold', color='#555')

    for i, (rule, comparison, is_match, c) in enumerate(rules_data):
        y = 1.8 - i * 0.7
        x_start = 6.5

        # Rule box
        rule_box = FancyBboxPatch((x_start, y - 0.22), 1.6, 0.44,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor=DARK,
                                   lw=1.5, alpha=0.85, zorder=3)
        ax.add_patch(rule_box)
        ax.text(x_start + 0.8, y, rule, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=4)

        # Arrow
        ax.annotate('', xy=(8.5, y), xytext=(8.2, y),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2))

        # Comparison
        ax.text(9.5, y, comparison, ha='center', va='center', fontsize=9, color='#555')

        # Arrow to result
        ax.annotate('', xy=(10.8, y), xytext=(10.5, y),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2))

        # Result
        txt = 'MATCH' if is_match else 'DIFF'
        result_badge(ax, 11.8, y, txt, is_match=is_match)

    # Rule switching annotation
    switch_box = FancyBboxPatch((0.5, -1.5), 13.0, 0.9,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FDF2E9', edgecolor=COLORS['orange'],
                                 lw=2, zorder=2)
    ax.add_patch(switch_box)
    ax.text(7.0, -1.05, 'Rule switches after 6 consecutive correct responses. '
            'The model must detect the switch\n'
            'and adapt. Perseverating on the old rule = perseverative error '
            '(hallmark of prefrontal damage).',
            ha='center', va='center', fontsize=9, color=DARK, zorder=3,
            linespacing=1.4)


# =====================================================================
# GEE Unified Architecture
# =====================================================================
def draw_gee_unified():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

    BLUE = '#4A90D9'
    PURPLE = '#9B59B6'
    ORANGE = '#E67E22'
    GREEN = '#27AE60'
    TEAL = '#1ABC9C'
    ENCODER_COLOR = '#E1D5E7'
    GATE_COLOR = '#FFF2CC'
    GRU_COLOR = '#DAE8FC'
    MEMORY_COLOR = '#D5E8D4'

    def ubox(cx, cy, w, h, text, color=BLUE, fontsize=10, text_color='white', ls='-'):
        b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                            boxstyle="round,pad=0.08", facecolor=color,
                            edgecolor=DARK, linewidth=1.5, linestyle=ls, zorder=2)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, zorder=3, linespacing=1.3)

    def uarrow(x1, y1, x2, y2, color=DARK, lw=1.8, rad=0.0, ls='-'):
        cs = f'arc3,rad={rad}'
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                             connectionstyle=cs, mutation_scale=16,
                             lw=lw, color=color, linestyle=ls, zorder=1)
        ax.add_patch(a)

    def ulabel(x, y, text, fontsize=9, color=DARK, **kw):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=color, fontstyle='italic', zorder=4, **kw)

    # Title
    ax.text(7.0, 10.0, 'GEE Unified: Single Model for All Tasks',
            ha='center', fontsize=18, fontweight='bold', color=DARK)
    ax.text(7.0, 9.4, 'h\u2080 sets cognitive mode  |  RCM tracks dynamic context  |  '
            'Keys & memory untouched by context',
            ha='center', fontsize=10, color='#777', fontstyle='italic')

    # ── ESBN CORE ──
    esbn_bg = FancyBboxPatch((3.0, 2.0), 8.5, 6.5, boxstyle="round,pad=0.2",
                              facecolor='#F8F9FA', edgecolor=BLUE,
                              linewidth=2.5, linestyle='--', zorder=0, alpha=0.5)
    ax.add_patch(esbn_bg)
    ax.text(7.25, 8.2, 'ESBN Core (pure indirection, unchanged)',
            ha='center', fontsize=11, fontweight='bold', color=BLUE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=BLUE, alpha=0.9))

    # Input → Encoder → z_t
    ubox(4.5, 7.2, 2.2, 0.65, 'Encoder +\nContext Norm', color=ENCODER_COLOR,
         text_color=DARK, fontsize=9)
    ubox(7.2, 7.2, 0.9, 0.55, 'z_t', color=TEAL, text_color='white', fontsize=11)
    uarrow(5.6, 7.2, 6.75, 7.2)

    # Memory
    mem_bg = FancyBboxPatch((8.5, 4.6), 2.6, 2.4, boxstyle="round,pad=0.1",
                             facecolor=MEMORY_COLOR, edgecolor=GREEN,
                             linewidth=2, zorder=1, alpha=0.4)
    ax.add_patch(mem_bg)
    ax.text(9.8, 6.8, 'Memory', ha='center', fontsize=10,
            fontweight='bold', color=GREEN)
    ubox(9.8, 6.2, 1.3, 0.5, 'M_v', color='#82C46C', text_color=DARK, fontsize=10)
    ubox(9.8, 5.3, 1.3, 0.5, 'M_k', color='#5BAD3E', text_color='white', fontsize=10)

    # z_t → Memory
    uarrow(7.65, 7.2, 9.15, 6.45, color=TEAL, lw=2)

    # Retrieval path
    ubox(7.8, 5.3, 1.0, 0.45, 'Gate', color=GATE_COLOR, text_color=DARK, fontsize=9)
    uarrow(9.15, 5.3, 8.3, 5.3)

    # key_r
    ubox(6.2, 4.5, 1.1, 0.5, 'key_r', color='#F39C12', text_color='white', fontsize=10)
    uarrow(7.8, 5.05, 6.75, 4.7, lw=2)

    # GRU
    ubox(6.2, 3.3, 2.2, 0.8, 'GRU\nController', color=BLUE,
         text_color='white', fontsize=11)
    uarrow(6.2, 4.25, 6.2, 3.7, color='#F39C12', lw=2.5)

    # GRU → key_w → M_k
    ubox(9.3, 3.3, 1.1, 0.45, 'key_w', color='#5BAD3E', text_color='white', fontsize=9)
    uarrow(7.3, 3.3, 8.75, 3.3)
    uarrow(9.3, 3.52, 9.8, 5.05, color='#5BAD3E', lw=1.5, rad=-0.2)

    # GRU → base output
    ubox(6.2, 2.3, 1.5, 0.5, 'logits', color='#2C6EA4', text_color='white', fontsize=10)
    uarrow(6.2, 2.9, 6.2, 2.55)

    # ── h₀ PATHWAY ──
    h0_bg = FancyBboxPatch((-0.5, 2.5), 2.8, 5.5, boxstyle="round,pad=0.15",
                            facecolor='#F4ECF7', edgecolor=PURPLE,
                            linewidth=2.5, zorder=0, alpha=0.5)
    ax.add_patch(h0_bg)
    ax.text(0.9, 7.7, 'h\u2080 Pathway', ha='center', fontsize=12,
            fontweight='bold', color=PURPLE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=PURPLE, alpha=0.9))

    ubox(0.9, 6.8, 1.8, 0.55, 'task_id\n(one-hot)', color='#D2B4DE',
         text_color=DARK, fontsize=8)
    ubox(0.9, 5.8, 1.8, 0.55, 'Linear + tanh', color=PURPLE,
         text_color='white', fontsize=9)
    uarrow(0.9, 6.52, 0.9, 6.08, color=PURPLE, lw=2)

    ubox(0.9, 4.7, 1.2, 0.55, 'h\u2080', color=PURPLE, text_color='white', fontsize=13)
    uarrow(0.9, 5.52, 0.9, 4.97, color=PURPLE, lw=2)

    # h₀ → GRU
    uarrow(1.5, 4.0, 5.1, 3.5, color=PURPLE, lw=2.5)
    ulabel(3.0, 3.9, 'initial hidden\nstate', fontsize=8, color=PURPLE)

    # h₀ function label
    ax.text(0.9, 3.3, 'Sets cognitive\n"mode" at start\nof each trial',
            ha='center', fontsize=8, color=PURPLE, fontstyle='italic', linespacing=1.3)

    # Noise injection point
    ubox(0.9, 4.0, 1.6, 0.4, '\u26a1 h\u2080 noise', color='#F4ECF7',
         text_color=PURPLE, fontsize=8, ls='--')

    # ── RCM PATHWAY ──
    rcm_bg = FancyBboxPatch((12.0, 1.5), 2.8, 6.5, boxstyle="round,pad=0.15",
                             facecolor='#FDF2E9', edgecolor=ORANGE,
                             linewidth=2.5, zorder=0, alpha=0.5)
    ax.add_patch(rcm_bg)
    ax.text(13.4, 7.7, 'RCM Pathway', ha='center', fontsize=12,
            fontweight='bold', color=ORANGE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=ORANGE, alpha=0.9))

    ubox(13.4, 6.8, 1.8, 0.55, 'x_raw +\ntask_id', color='#F5CBA7',
         text_color=DARK, fontsize=8)
    ubox(13.4, 5.9, 1.6, 0.45, 'y_prev', color='#F5CBA7',
         text_color=DARK, fontsize=8)
    ubox(13.4, 4.9, 1.8, 0.6, 'RCM\n(GRUCell)', color=ORANGE,
         text_color='white', fontsize=10)
    uarrow(13.4, 6.52, 13.4, 5.2, color=ORANGE, lw=2)
    uarrow(13.4, 5.67, 13.4, 5.2, color=ORANGE, lw=1.5)

    ubox(13.4, 3.9, 1.0, 0.45, 'c_t', color=ORANGE, text_color='white', fontsize=11)
    uarrow(13.4, 4.6, 13.4, 4.12, color=ORANGE, lw=2)

    # Recurrence
    uarrow(14.0, 3.9, 14.4, 4.9, color=ORANGE, lw=1.2, rad=-0.5)

    ubox(13.4, 3.0, 1.8, 0.45, 'Linear\n(ctx\u2192out)', color=ORANGE,
         text_color='white', fontsize=8)
    uarrow(13.4, 3.67, 13.4, 3.22, color=ORANGE, lw=2)

    # Noise injection point
    ubox(13.4, 2.2, 1.6, 0.4, '\u26a1 RCM noise', color='#FDF2E9',
         text_color=ORANGE, fontsize=8, ls='--')

    ax.text(13.4, 1.7, 'Tracks rule\nchanges across\ntrials', ha='center',
            fontsize=8, color=ORANGE, fontstyle='italic', linespacing=1.3)

    # ── OUTPUT COMBINATION ──
    ubox(9.0, 2.3, 1.2, 0.5, '+ bias', color='#FDEBD0', text_color=DARK, fontsize=10)
    uarrow(6.95, 2.3, 8.4, 2.3)
    uarrow(12.5, 3.0, 9.6, 2.3, color=ORANGE, lw=2.5)

    ubox(9.0, 1.2, 1.8, 0.6, 'Final\nOutput y', color='#1A5276',
         text_color='white', fontsize=10)
    uarrow(9.0, 2.05, 9.0, 1.5, lw=2)

    # ── Dual output heads note ──
    ax.text(9.0, 0.5, 'Dual heads: y_out_2 (binary) + y_out_4 (4-way)\n'
            'Task_id dropout (p=0.3) during training',
            ha='center', fontsize=8, color='#888', fontstyle='italic', linespacing=1.3)

    # ── TASK ID INPUT ──
    ubox(0.9, 7.7, 1.6, 0.5, 'Task ID', color='#D2B4DE',
         text_color=DARK, fontsize=10, ls='--')
    uarrow(1.7, 7.7, 3.0, 7.2, color=PURPLE, lw=1, ls='--')
    # Task ID also goes to RCM
    uarrow(1.7, 7.9, 12.5, 7.5, color='#999', lw=1, ls='--')
    uarrow(12.5, 7.5, 13.4, 7.07, color=ORANGE, lw=1.5)

    # Input to encoder
    ubox(2.0, 7.9, 1.2, 0.45, 'x_seq', color='#BDC3C7', text_color=DARK, fontsize=9)
    uarrow(2.6, 7.9, 3.4, 7.4)

    # ── CONSTRAINT ──
    constraint_box = FancyBboxPatch((2.5, -1.2), 10.0, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FDEDEC', edgecolor='#E74C3C',
                                     linewidth=2, zorder=2, alpha=0.9)
    ax.add_patch(constraint_box)
    ax.text(7.5, -0.8, 'KEYS and MEMORY are NEVER touched by context. '
            'Indirection pathway = pure ESBN.',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#E74C3C', zorder=3)

    plt.tight_layout()
    plt.savefig('fig_diagram_gee_unified.png', bbox_inches='tight',
                facecolor='white', dpi=200)
    print("Saved: fig_diagram_gee_unified.png")
    plt.close()


# =====================================================================
# Combined Task Visualization
# =====================================================================
def draw_all_tasks():
    fig, axes = plt.subplots(5, 1, figsize=(16, 28))
    fig.subplots_adjust(hspace=0.35)

    draw_same_different(axes[0])
    draw_rmts(axes[1])
    draw_dist3(axes[2])
    draw_identity_rules(axes[3])
    draw_wcst(axes[4])

    # Dividers between tasks
    for i in range(4):
        line_y = 1.0 - (i + 1) * 0.2  # approximate
        # Use figure-level coordinates
        fig.add_artist(plt.Line2D([0.05, 0.95],
                                   [1.0 - (i + 1) / 5 - 0.01, 1.0 - (i + 1) / 5 - 0.01],
                                   transform=fig.transFigure,
                                   color='#DDD', lw=1.5, ls='--'))

    fig.suptitle('The Five Tasks: Abstract Relational Reasoning Suite',
                 fontsize=20, fontweight='bold', y=0.995, color=DARK)

    plt.savefig('fig_tasks_overview.png', bbox_inches='tight', facecolor='white', dpi=200)
    print("Saved: fig_tasks_overview.png")
    plt.close()


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    draw_gee_unified()
    draw_all_tasks()
    print("\nAll diagrams generated.")
