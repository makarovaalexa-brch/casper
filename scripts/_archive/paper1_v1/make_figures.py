"""
Paper 1: generate all figures and table data from benchmark outputs.

Reads:
    experiments/paper1/instrument_acceptance.json
    experiments/paper1/benchmark_results.json
    experiments/paper1/episodes_<policy>.jsonl

Writes figures (PDF + PNG) to papers/paper1_casper/figures/ and table data
to papers/paper1_casper/tables/. Robust to missing inputs: generates
whatever the available data supports and lists what was skipped.

Run from casper root:  poetry run python scripts/paper1/make_figures.py
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = Path('C:/dev/phd/casper/experiments/paper1')
FIG_DIR = Path('C:/dev/phd/papers/paper1_casper/figures')
TAB_DIR = Path('C:/dev/phd/papers/paper1_casper/tables')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

POLICY_ORDER = ['random', 'popularity', 'greedy_infogain', 'botplay_rl',
                'llm_vanilla', 'llm_strategist', 'llm_gate']
POLICY_LABELS = {
    'random': 'Random', 'popularity': 'Popularity',
    'greedy_infogain': 'Greedy info-gain', 'botplay_rl': 'Bot-play RL',
    'llm_vanilla': 'LLM (vanilla)', 'llm_strategist': 'LLM (strategist)',
    'llm_gate': 'LLM (GATE)',
}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
TYPE_OF_IDX = None  # filled from slate metadata


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  wrote {name}")


def load_episodes():
    eps = {}
    for f in EXP_DIR.glob('episodes_*.jsonl'):
        pname = f.stem.replace('episodes_', '')
        eps[pname] = [json.loads(line) for line in f.read_text().splitlines()
                      if line.strip()]
    return eps


def slate_types():
    """entity idx -> type, from any instrument checkpoint."""
    global TYPE_OF_IDX
    if TYPE_OF_IDX is None:
        import torch
        from test_instrument_lib import CHECKPOINT_DIR
        for name in ('instrument_v3_onehot.pt', 'onehot_paper_config.pt'):
            p = CHECKPOINT_DIR / name
            if p.exists():
                items = torch.load(p, weights_only=False)['items']
                TYPE_OF_IDX = {i: it[0] for i, it in enumerate(items)}
                break
    return TYPE_OF_IDX


# ---------------------------------------------------------------------------
# Fig 1/2: accuracy and BCE curves
# ---------------------------------------------------------------------------

def fig_curves(results, metric='accuracy'):
    key = 'mean_curve' if metric == 'accuracy' else None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for i, p in enumerate(POLICY_ORDER):
        if p not in results or 'mean_curve' not in results[p]:
            continue
        curve = results[p]['mean_curve']
        ax.plot(range(len(curve)), curve, label=POLICY_LABELS.get(p, p),
                color=COLORS[i], lw=1.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel('Turn')
    ax.set_ylabel('Mean accuracy on held-out ratings')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, 'fig1_accuracy_curves')
    return True


def fig_bce_curves(episodes, n_turns=15):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for i, p in enumerate(POLICY_ORDER):
        if p not in episodes:
            continue
        curves = []
        for e in episodes[p]:
            c = e['bce'] + [e['bce'][-1]] * (n_turns + 1 - len(e['bce']))
            curves.append(c[:n_turns + 1])
        ax.plot(range(n_turns + 1), np.mean(curves, axis=0),
                label=POLICY_LABELS.get(p, p), color=COLORS[i], lw=1.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel('Turn')
    ax.set_ylabel('Mean BCE on held-out ratings')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, 'fig2_bce_curves')
    return True


# ---------------------------------------------------------------------------
# Fig 3: hit rate by turn; Fig 7: redundancy by turn
# ---------------------------------------------------------------------------

def fig_by_turn(episodes, n_turns=15):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plotted = False
    for i, p in enumerate(POLICY_ORDER):
        if p not in episodes:
            continue
        hit = np.full((len(episodes[p]), n_turns), np.nan)
        red = np.full((len(episodes[p]), n_turns), np.nan)
        for r, e in enumerate(episodes[p]):
            for t, a in enumerate(e['answers'][:n_turns]):
                hit[r, t] = a != 'unknown'
                if e.get('asked_scores') and t < len(e['asked_scores']):
                    s = e['asked_scores'][t]
                    confident = s > 0.8 or s < 0.2
                    agreed = ((s > 0.8 and a == 'liked') or
                              (s < 0.2 and a == 'disliked'))
                    red[r, t] = confident and agreed
        axes[0].plot(np.arange(1, n_turns + 1), np.nanmean(hit, axis=0),
                     label=POLICY_LABELS.get(p, p), color=COLORS[i], lw=1.6)
        if not np.all(np.isnan(red)):
            axes[1].plot(np.arange(1, n_turns + 1), np.nanmean(red, axis=0),
                         color=COLORS[i], lw=1.6)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    axes[0].set_title('Hit rate by turn')
    axes[1].set_title('Redundancy rate by turn')
    for ax in axes:
        ax.set_xlabel('Turn')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    save(fig, 'fig3_hit_fig7_redundancy')
    return True


# ---------------------------------------------------------------------------
# Fig 4: question-type mix; Fig 5: question-turn heatmap; Fig 6: coverage
# ---------------------------------------------------------------------------

def fig_type_mix(episodes, n_turns=15):
    types = slate_types()
    if types is None:
        return False
    tnames = ['genre', 'actor', 'director', 'movie']
    pol_list = [p for p in POLICY_ORDER if p in episodes]
    if not pol_list:
        return False
    fig, axes = plt.subplots(1, len(pol_list), figsize=(2.4 * len(pol_list), 3.2),
                             sharey=True)
    if len(pol_list) == 1:
        axes = [axes]
    for ax, p in zip(axes, pol_list):
        frac = np.zeros((len(tnames), n_turns))
        for t in range(n_turns):
            c = Counter(types[e['questions'][t]] for e in episodes[p]
                        if len(e['questions']) > t)
            tot = sum(c.values()) or 1
            for j, tn in enumerate(tnames):
                frac[j, t] = c.get(tn, 0) / tot
        ax.stackplot(np.arange(1, n_turns + 1), frac, labels=tnames, alpha=0.85)
        ax.set_title(POLICY_LABELS.get(p, p), fontsize=8)
        ax.set_xlabel('Turn', fontsize=8)
    axes[0].set_ylabel('Question-type fraction')
    axes[-1].legend(fontsize=7, loc='upper right')
    save(fig, 'fig4_question_type_mix')
    return True


def fig_heatmap_coverage(episodes, n_turns=15, top_k=20):
    types = slate_types()
    pol_list = [p for p in POLICY_ORDER if p in episodes]
    if not pol_list:
        return False
    # heatmaps
    fig, axes = plt.subplots(1, len(pol_list), figsize=(2.6 * len(pol_list), 4))
    if len(pol_list) == 1:
        axes = [axes]
    for ax, p in zip(axes, pol_list):
        counts = defaultdict(lambda: np.zeros(n_turns))
        for e in episodes[p]:
            for t, q in enumerate(e['questions'][:n_turns]):
                counts[q][t] += 1
        top = sorted(counts, key=lambda q: -counts[q].sum())[:top_k]
        mat = np.stack([counts[q] for q in top]) if top else np.zeros((1, n_turns))
        ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_title(POLICY_LABELS.get(p, p), fontsize=8)
        ax.set_xlabel('Turn', fontsize=8)
        ax.set_yticks([])
    axes[0].set_ylabel(f'Top-{top_k} entities')
    save(fig, 'fig5_question_turn_heatmap')

    # coverage entropy
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, p in enumerate(POLICY_ORDER):
        if p not in episodes:
            continue
        ent = []
        for t in range(n_turns):
            c = Counter(e['questions'][t] for e in episodes[p]
                        if len(e['questions']) > t)
            tot = sum(c.values())
            if tot == 0:
                ent.append(np.nan)
                continue
            ps = np.array(list(c.values())) / tot
            ent.append(float(-(ps * np.log2(ps)).sum()))
        ax.plot(np.arange(1, n_turns + 1), ent, label=POLICY_LABELS.get(p, p),
                color=COLORS[i], lw=1.6)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Coverage entropy (bits)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, 'fig6_coverage_entropy')
    return True


# ---------------------------------------------------------------------------
# Fig 9: per-user violins; Fig 10: power analysis
# ---------------------------------------------------------------------------

def fig_violins_power(episodes, n_turns=15):
    pol_list = [p for p in POLICY_ORDER if p in episodes]
    if not pol_list:
        return False
    finals = {p: [e['accuracy'][-1] for e in episodes[p]] for p in pol_list}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot([finals[p] for p in pol_list], showmedians=True)
    ax.set_xticks(range(1, len(pol_list) + 1))
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in pol_list],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Final accuracy per user')
    ax.grid(alpha=0.3, axis='y')
    save(fig, 'fig9_final_accuracy_violins')

    # power analysis from pooled dispersion
    sigma = float(np.mean([np.std(v) for v in finals.values()]))
    ns = np.arange(20, 1001, 10)
    detectable = 1.96 * sigma * np.sqrt(2.0 / ns)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, detectable, lw=2)
    ax.axvline(200, color='grey', ls='--', lw=1)
    ax.set_xlabel('Evaluation users $n$')
    ax.set_ylabel('Detectable accuracy difference (95%)')
    ax.set_title(f'Empirical per-user dispersion $\\sigma$ = {sigma:.3f}')
    ax.grid(alpha=0.3)
    save(fig, 'fig10_power_analysis')
    return True


# ---------------------------------------------------------------------------
# Fig 11: acceptance-test evolution (works with data available today)
# ---------------------------------------------------------------------------

def fig_acceptance():
    path = EXP_DIR / 'instrument_acceptance.json'
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    gens = [g for g in ('onehot', 'concept', 'instrument_v2_onehot',
                        'instrument_v3_onehot') if g in data]
    labels = {'onehot': 'OneHot', 'concept': 'Concept',
              'instrument_v2_onehot': '+RS', 'instrument_v3_onehot': '+RS+TR'}

    metrics = [
        ('T1v3 DiD polarity', lambda d: d.get('T1v3_attribute_did', {})
         .get('fraction_correct_direction')),
        ('T3 franchise flip', lambda d: d.get('T3v2_multifranchise', {})
         .get('fraction_correct_direction')),
        ('1 - L/D overlap', lambda d: None if 'T1b_prior_overlap' not in d
         else 1 - d['T1b_prior_overlap']['overlap_liked_disliked']),
        ('T2 monotonicity rho', lambda d: d.get('T2_T4_monotonicity_snr', {})
         .get('spearman_rho')),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(gens)
    x = np.arange(len(metrics))
    for gi, g in enumerate(gens):
        vals = [m[1](data[g]) for m in metrics]
        vals = [v if v is not None else 0 for v in vals]
        ax.bar(x + gi * width, vals, width, label=labels[g])
    ax.set_xticks(x + width * (len(gens) - 1) / 2)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.axhline(0.8, color='grey', ls=':', lw=1)
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    save(fig, 'fig11_acceptance_evolution')
    return True


# ---------------------------------------------------------------------------
# Table data exports
# ---------------------------------------------------------------------------

def export_leaderboard(results):
    rows = []
    for p in POLICY_ORDER:
        if p not in results:
            continue
        s = results[p]
        rows.append({
            'policy': POLICY_LABELS.get(p, p),
            'final_accuracy': s['final_accuracy'],
            'final_ci': s['final_accuracy_ci95'],
            'auac': s['auac'],
            'auac_ci': s['auac_ci95'],
            'hit_rate': s['hit_rate'],
            'n_users': s['n_users'],
        })
    (TAB_DIR / 'leaderboard.json').write_text(json.dumps(rows, indent=2))
    print("  wrote leaderboard.json")


def main():
    results_path = EXP_DIR / 'benchmark_results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    episodes = load_episodes()
    n_turns = results.get('_meta', {}).get('n_turns', 15)

    done, skipped = [], []
    for name, fn in [
        ('fig1', lambda: fig_curves(results)),
        ('fig2', lambda: fig_bce_curves(episodes, n_turns)),
        ('fig3+7', lambda: fig_by_turn(episodes, n_turns)),
        ('fig4', lambda: fig_type_mix(episodes, n_turns)),
        ('fig5+6', lambda: fig_heatmap_coverage(episodes, n_turns)),
        ('fig9+10', lambda: fig_violins_power(episodes, n_turns)),
        ('fig11', fig_acceptance),
    ]:
        try:
            (done if fn() else skipped).append(name)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            skipped.append(name)
    if results:
        export_leaderboard(results)

    print(f"\nGenerated: {done}")
    print(f"Skipped (no data yet): {skipped}")


if __name__ == '__main__':
    main()
