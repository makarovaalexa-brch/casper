r"""Targeted staleness check: each headline chapter claim against ITS OWN source file."""
import io
import os
import json

ROOT = r'C:\dev\phd\casper'


def J(rel):
    p = os.path.join(ROOT, rel.replace('/', os.sep))
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding='utf-8'))


def dig(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


CHECKS = []


def chk(label, claimed, actual, src):
    ok = (actual is not None) and (abs(float(claimed) - float(actual)) < 5e-5)
    CHECKS.append((ok, label, claimed, actual, src))


# ---- R1: tower vs RecVAE, arm A -------------------------------------------------
t = J('experiments/baselines/arm_n/tower.json')
rec = J('experiments/baselines/ml25m_liang/recvae.json')
chk('RecVAE arm-A full', 0.3540, dig(rec, 'ndcg@10'), 'ml25m_liang/recvae.json')
chk('RecVAE arm-A tail', 0.2497, dig(rec, 'tail_ndcg@10'), 'ml25m_liang/recvae.json')

# ---- R1 paired CI ---------------------------------------------------------------
pa = J('experiments/baselines/arm_n/paired_recvae_armA.json')
if pa:
    chk('R1 paired full delta', -0.0043, dig(pa, 'ndcg@10', 'diff'), 'paired_recvae_armA.json')
    ci = dig(pa, 'ndcg@10', 'ci95') or [None, None]
    chk('R1 paired full CI lo', -0.0059, ci[0], 'paired_recvae_armA.json')
    chk('R1 paired full CI hi', -0.0028, ci[1], 'paired_recvae_armA.json')
    chk('R1 paired tail delta', -0.0006, dig(pa, 'tail_ndcg@10', 'diff'), 'paired_recvae_armA.json')
    cit = dig(pa, 'tail_ndcg@10', 'ci95') or [None, None]
    chk('R1 paired tail CI lo', -0.0020, cit[0], 'paired_recvae_armA.json')
    chk('R1 paired tail CI hi', 0.0008, cit[1], 'paired_recvae_armA.json')

# ---- intensity premium (identical support) --------------------------------------
g = J('experiments/baselines/arm_n/graded_ab_paired.json')
if g:
    chk('intensity full', 0.0039, dig(g, 'ndcg@10', 'mean_diff'), 'graded_ab_paired.json')
    ci = dig(g, 'ndcg@10', 'ci') or [None, None]
    chk('intensity full CI lo', 0.0025, ci[0], 'graded_ab_paired.json')
    chk('intensity full CI hi', 0.0053, ci[1], 'graded_ab_paired.json')
    chk('intensity tail', 0.0020, dig(g, 'tail_ndcg@10', 'mean_diff'), 'graded_ab_paired.json')
    cit = dig(g, 'tail_ndcg@10', 'ci') or [None, None]
    chk('intensity tail CI lo', 0.0007, cit[0], 'graded_ab_paired.json')
    chk('intensity tail CI hi', 0.0031, cit[1], 'graded_ab_paired.json')

# ---- concept vs item, polar bank ------------------------------------------------
c = J('experiments/instrument/concept_vs_item_paired.json')
if c:
    for m, ax, val in [(1, 'full', 0.0045), (1, 'tail', 0.0109),
                       (8, 'full', 0.0189), (8, 'tail', 0.0310),
                       (16, 'full', 0.0146), (16, 'tail', 0.0265)]:
        chk('polar m%d %s' % (m, ax), val, dig(c, 'polar', 'm%d_%s' % (m, ax), 'delta'),
            'concept_vs_item_paired.json')
    chk('pop m16 full', -0.0782, dig(c, 'pop', 'm16_full', 'delta'), 'concept_vs_item_paired.json')

# ---- interview k=8 --------------------------------------------------------------
iv = J('experiments/baselines/interview/interview_paired.json')
if iv:
    p = dig(iv, 'points') or {}
    chk('ours E0 k8 full', 0.2053, dig(p, 'ours', 'entropy0', 'full'), 'interview_paired.json')
    chk('ours HELF k8 full', 0.1892, dig(p, 'ours', 'helf', 'full'), 'interview_paired.json')
    chk('user-kNN HELF k8 full', 0.1948, dig(p, 'golbandi_leaf', 'helf', 'full'), 'interview_paired.json')
    chk('ours E0 k8 tail', 0.0792, dig(p, 'ours', 'entropy0', 'tail'), 'interview_paired.json')
    chk('user-kNN HELF k8 tail', 0.0792, dig(p, 'golbandi_leaf', 'helf', 'tail'), 'interview_paired.json')
    chk('TaNP E0 k8 full', 0.1742, dig(p, 'tanp', 'entropy0', 'full'), 'interview_paired.json')
    chk('RBMF E0 k8 full', 0.1644, dig(p, 'rbmf', 'entropy0', 'full'), 'interview_paired.json')
    chk('recvae_likes POP k8 full', 0.1760, dig(p, 'recvae_likes', 'popularity', 'full'), 'interview_paired.json')
    chk('recvae_likes E0 k8 full', 0.1718, dig(p, 'recvae_likes', 'entropy0', 'full'), 'interview_paired.json')
    chk('recvae_likes HELF k8 full', 0.1003, dig(p, 'recvae_likes', 'helf', 'full'), 'interview_paired.json')
    chk('recvae_likes POP k8 tail', 0.0534, dig(p, 'recvae_likes', 'popularity', 'tail'), 'interview_paired.json')
    chk('belief_mf HELF k8 full', 0.1614, dig(p, 'belief_mf', 'helf', 'full'), 'interview_paired.json')
    chk('RBMF HELF k8 full', 0.0698, dig(p, 'rbmf', 'helf', 'full'), 'interview_paired.json')
    cm = dig(iv, 'cross_model') or {}
    chk('ours-minus-userkNN full', 0.0105,
        dig(cm, 'ours(entropy0)-minus-golbandi_leaf(helf)/full', 'delta'), 'interview_paired.json')
    chk('ours-minus-recvae_likes full', 0.0293,
        dig(cm, 'ours(entropy0)-minus-recvae_likes(popularity)/full', 'delta'), 'interview_paired.json')
    chk('ours-minus-recvae_likes tail', 0.0258,
        dig(cm, 'ours(entropy0)-minus-recvae_likes(popularity)/tail', 'delta'), 'interview_paired.json')

# ---- arm A / arm N master table rows --------------------------------------------
for name, a_full, a_tail, n_full, n_tail in [
        ('pop', .1345, .0226, .1626, .0262), ('itemknn', .2428, .1297, .2946, .1593),
        ('ease', .3476, .2441, .4135, .2941), ('edlae', .3433, .2395, .4110, .2893),
        ('turbocf', .2622, .1621, .3285, .2034), ('recvae', .3540, .2497, .4273, .3051),
        ('multvae', .3202, .2194, .3904, .2682), ('sasrec', .1486, .0805, .1842, .0986),
        ('golbandi_node', .3065, .1895, .3756, .2377), ('ials', .2427, .1855, .3044, .2293),
        ('belief_mf', .2015, .1201, .2521, .1511), ('tanp', .2645, .1635, .3324, .2078),
        ('rbmf_seed', .2457, .1722, .3015, .2078), ('eddi', .2153, .1077, .2718, .1383)]:
    d = J('experiments/baselines/arm_n/%s.json' % name)
    can = J('experiments/baselines/ml25m_liang/%s.json' % name)
    chk('%s LIANG full' % name, a_full, dig(can, 'ndcg@10'), 'ml25m_liang/%s.json' % name)
    chk('%s LIANG tail' % name, a_tail, dig(can, 'tail_ndcg@10'), 'ml25m_liang/%s.json' % name)
    chk('%s NATIVE full' % name, n_full, dig(d, 'N', 'ndcg@10'), 'arm_n/%s.json' % name)
    chk('%s NATIVE tail' % name, n_tail, dig(d, 'N', 'tail_ndcg@10'), 'arm_n/%s.json' % name)

bad = [c for c in CHECKS if not c[0]]
print('targeted checks: %d ; MISMATCH/MISSING: %d' % (len(CHECKS), len(bad)))
print('=' * 100)
for ok, label, claimed, actual, src in bad:
    a = 'MISSING' if actual is None else ('%.4f' % actual)
    print('  %-26s chapter %-9s  source %-9s  <- %s' % (label, '%.4f' % claimed, a, src))
