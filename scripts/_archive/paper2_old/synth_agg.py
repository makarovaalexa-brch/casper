"""Aggregate synth_ndcg seed runs -> mean +/- std per model (the Paper B synthetic table).
Usage: python synth_agg.py            # reads experiments/paper2/synth_ndcg_seed*.json
"""
import json, glob, numpy as np, os
fs=sorted(glob.glob('C:/dev/phd/casper/experiments/paper2/synth_ndcg_seed*.json'))
fs=[f for f in fs if 'seed99' not in f]                                      # seed99 = validation, exclude from the reported average
runs=[json.load(open(f)) for f in fs]
if not runs: print("no seed runs found (run synth_ndcg.py with SEED=0..4 first)"); raise SystemExit
seeds=[r['seed'] for r in runs]; models=list(runs[0]['results'].keys())
print(f"seeds={seeds}  (n={len(runs)})  T={runs[0]['T']}  NU={runs[0]['NU']} NTE={runs[0]['NTE']}\n")
print("%-18s %16s %16s %16s"%('model','NDCG@10','tailNDCG','Recall@10'))
agg={}
for k in models:
    nd=np.array([r['results'][k]['ndcg'] for r in runs]); tl=np.array([r['results'][k]['tail'] for r in runs]); rc=np.array([r['results'][k]['recall'] for r in runs])
    agg[k]=(nd.mean(),nd.std());
    print("%-18s %7.4f +/-%5.4f %7.4f +/-%5.4f %7.4f +/-%5.4f"%(k,nd.mean(),nd.std(),tl.mean(),tl.std(),rc.mean(),rc.std()))
hl=np.array([r['headroom'] for r in runs])
# capture fraction vs the STRONG non-adaptive baseline (static_best), the meaningful denominator in this symmetric world
o=agg['oracle_adaptive'][0]; s=agg['static_best'][0]; p=agg['policy_ours'][0]; e=agg['entropy'][0]
print(f"\nADAPTIVITY HEADROOM (oracle_adaptive - static_best) = +{o-s:.4f} (+/-{hl.std():.4f})")
print(f"policy_ours - static_best = {p-s:+.4f}  -> captures {100*(p-s)/max(o-s,1e-6):.0f}% of the oracle headroom over the best STATIC policy")
print(f"policy_ours - entropy     = {p-e:+.4f} (entropy is ~random here: symmetric world, all concepts ~equally divisive)")
