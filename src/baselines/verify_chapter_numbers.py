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

# ---- OUR OWN ROWS. Added 2026-08-08 after finding that tower.json was loaded here and never read
# from: 93 checks resolved every baseline against its source file and left the instrument's own four
# master-table cells unverified. That is the worst possible place to have a blind spot, and it had
# already cost us -- the NATIVE pair sat at .4179/.2955 in the chapter while the certified ep31 tower
# gives .4244/.3017. The snapshot assertion below is the part that matters most: a number can be
# arithmetically right and still be the wrong tower's, which is exactly the R5 defect found the same day.
chk('tower snapshot is the certified ep31', 1.0,
    1.0 if (t or {}).get('snapshot') == 't2i26c_ep31.pt' else 0.0, 'arm_n/tower.json:snapshot')
chk('ours arm-A (LIANG) full', 0.3496, dig(t, 'results', 'A_full', 'ndcg@10'), 'arm_n/tower.json')
chk('ours arm-A (LIANG) tail', 0.2491, dig(t, 'results', 'A_full', 'tail_ndcg@10'), 'arm_n/tower.json')
chk('ours arm-N (NATIVE) full', 0.4244, dig(t, 'results', 'N_full', 'ndcg@10'), 'arm_n/tower.json')
chk('ours arm-N (NATIVE) tail', 0.3017, dig(t, 'results', 'N_full', 'tail_ndcg@10'), 'arm_n/tower.json')

# iALS arm A. The chapter printed .2427 in the master table and .2442 in the ICF-ceiling sentence and in
# the gap-map coordinates -- the same row, two values, because the ceiling prose was written against an
# earlier run and never re-resolved. Checked here so one row cannot hold two numbers again.
_ia = J('experiments/baselines/arm_n/ials.json')
chk('iALS arm-A full (master table AND the ICF-ceiling sentence AND the gap map)',
    0.2427, dig(_ia, 'A', 'ndcg@10'), 'arm_n/ials.json')

# TaNP native (sec:nativelimit). The headline cell is the RMSE-selected run -- the model's own criterion,
# as its paper selects. The NDCG-selected run is the adversarially-favourable variant quoted alongside it;
# the two have DIFFERENT distinct-top-10 counts (1370 vs 1801) and the prose had taken the wrong one.
_tr = J('experiments/baselines/arm_n/tanp_native_armn_rmse.json')
_tn = J('experiments/baselines/arm_n/tanp_native_armn_ndcg.json')
chk('TaNP native rmse-selected full', 0.0060, dig(_tr, 'ndcg@10'), 'tanp_native_armn_rmse.json')
chk('TaNP native rmse-selected tail', 0.0005, dig(_tr, 'tail_ndcg@10'), 'tanp_native_armn_rmse.json')
chk('TaNP native rmse-selected distinct top-10 lists', 1370,
    dig(_tr, 'gates', 'distinct_top10_lists'), 'tanp_native_armn_rmse.json')
chk('TaNP native val RMSE', 0.6960, dig(_tr, 'val_rmse_best'), 'tanp_native_armn_rmse.json')
chk('TaNP native mean-predictor RMSE', 0.9549, dig(_tr, 'val_rmse_mean_predictor'),
    'tanp_native_armn_rmse.json')
chk('TaNP native ndcg-selected full (the adversarial variant)', 0.0118, dig(_tn, 'ndcg@10'),
    'tanp_native_armn_ndcg.json')

# ---- R5, and the snapshot assertions that were missing ---------------------------------------------
# Both R5 artefacts were measured on the PRE-certification tower until 2026-08-08 -- three chained stale
# defaults, plus arm_n_sign_probe being hardcoded to the i25 architecture so the certified tower could not
# be loaded at all. Assert the snapshot FIRST in each block: a number can match its file exactly and still
# belong to the wrong tower, which is the failure these checks exist to prevent.
_pv = J('experiments/battery/premium_vs_k_t2i26c_ep31.json')
chk('premium_vs_k snapshot is the certified ep31', 1.0,
    1.0 if (_pv or {}).get('snapshot') == 't2i26c_ep31.pt' else 0.0, 'premium_vs_k_t2i26c_ep31.json')
for _k, _v, _lo, _hi in (('2', 0.0102, 0.0087, 0.0118), ('4', 0.0080, 0.0063, 0.0097),
                         ('8', 0.0010, -0.0007, 0.0026), ('full', 0.0022, 0.0008, 0.0036)):
    _d = dig(_pv, 'per_k', _k, 'deltas', 'graded_over_binary(true-likes)')
    chk(f'graded premium k={_k} (fig:gradedcurve)', _v, dig(_d, 'mean'), 'premium_vs_k_t2i26c_ep31.json')
    chk(f'graded premium k={_k} CI lo', _lo, (_d or {}).get('ci95', [None, None])[0],
        'premium_vs_k_t2i26c_ep31.json')
    chk(f'graded premium k={_k} CI hi', _hi, (_d or {}).get('ci95', [None, None])[1],
        'premium_vs_k_t2i26c_ep31.json')

_sp = J('experiments/baselines/arm_n/sign_probe_t2i26c_ep31.json')
chk('sign_probe snapshot is the certified ep31', 1.0,
    1.0 if (_sp or {}).get('snapshot') == 't2i26c_ep31.pt' else 0.0, 'sign_probe_t2i26c_ep31.json')
chk('sign gap k=2 full (the value quoted in prose)', 0.0144, dig(_sp, 'results', 'k2', 'gap_full'),
    'sign_probe_t2i26c_ep31.json')
chk('sign gap k=8 full (gone by eight)', -0.0018, dig(_sp, 'results', 'k8', 'gap_full'),
    'sign_probe_t2i26c_ep31.json')

# ---- iALS graded ladder (sec:nativemech) ------------------------------------------------------------
_lp = J('experiments/baselines/arm_n/ials_ladder_paired.json')
for _lbl, _key, _ax, _v in (
        ('grades at fixed support full', 'GRADES at fixed support   r3-r2', 'full', 0.0282),
        ('grades at fixed support tail', 'GRADES at fixed support   r3-r2', 'tail', 0.0037),
        ('support with values erased full', 'support (values erased)   r2-r1', 'full', -0.0088),
        ('regularisation full', 'regularisation            r1-r0', 'full', 0.0257),
        ('grades vs best binary tail', 'grades vs best binary     r3-r1', 'tail', -0.0064)):
    chk(f'iALS ladder: {_lbl}', _v, dig(_lp, 'contrasts', _key, _ax, 'delta'), 'ials_ladder_paired.json')

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
