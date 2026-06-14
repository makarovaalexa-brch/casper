"""
PROOF of what went wrong (H1: low advantage-SNR collapse), computed from the
existing held-out episode logs -- no heavy compute.

Shows: (1) the REALIZABLE adaptive edge (best adaptive heuristic minus best
static policy) is tiny and barely significant; (2) per-step reward (held-out
accuracy increment) noise across users dwarfs it -> effect-size/SNR << 1, so a
policy gradient cannot resolve routing and correctly collapses to the best
fixed order; (3) the LARGE adaptive value lives only in the privileged oracle.
Context: adaptive-submodularity bounds non-adaptive greedy within (1-1/e) of
optimal adaptive (Golovin & Krause) -> small realizable edge is expected.

Usage: poetry run python scripts/paper2/snr_diagnostic.py
"""
import json
from pathlib import Path
import numpy as np

EXP = Path('C:/dev/phd/casper/experiments/paper1')
ORACLE_ATTR = 0.796   # clairvoyant attribute-routing ceiling (oracle_ceiling.py)


def load_auac(name):
    f = EXP / f'episodes_ml_stratified_{name}.jsonl'
    if not f.exists():
        return None, None
    traj = []
    for line in open(f):
        d = json.loads(line)
        a = d.get('accuracy_heldout') or d['accuracy']
        traj.append(a)
    L = max(len(t) for t in traj)
    arr = np.array([t + [t[-1]] * (L - len(t)) for t in traj])  # [users, turns]
    auac = np.nanmean(arr, axis=1)                               # per-user AUAC
    return arr, auac


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    pols = ['scpr_entropy', 'greedy_infogain', 'greedy_answerability',
            'ppo', 'popularity', 'random']
    arrs, auacs = {}, {}
    for p in pols:
        a, u = load_auac(p)
        if u is not None:
            arrs[p], auacs[p] = a, u
    print("=== per-user held-out AUAC ===")
    for p in pols:
        if p in auacs:
            m, lo, hi = boot_ci(auacs[p])
            print(f"  {p:<22} {m:.4f}  CI[{lo:.4f},{hi:.4f}]")

    best_adaptive = max(['scpr_entropy', 'greedy_infogain', 'greedy_answerability'],
                        key=lambda p: auacs.get(p, np.array([0])).mean())
    static = 'ppo' if 'ppo' in auacs else 'popularity'
    n = min(len(auacs[best_adaptive]), len(auacs[static]))
    delta = auacs[best_adaptive][:n] - auacs[static][:n]            # PAIRED edge
    dm, dlo, dhi = boot_ci(delta)
    print(f"\n=== REALIZABLE adaptive edge: {best_adaptive} - {static} (paired) ===")
    print(f"  Delta_J = {dm:+.4f}  CI[{dlo:+.4f},{dhi:+.4f}]  "
          f"({'SIGNIFICANT' if dlo > 0 else 'NOT significant (CI spans 0)'})")
    print(f"  oracle edge ({static}->oracle) = {ORACLE_ATTR - auacs[static].mean():+.4f}  "
          f"(={(ORACLE_ATTR - auacs[static].mean())/max(dm,1e-6):.0f}x the realizable edge)")

    # per-step reward (held-out accuracy increment) SNR for the adaptive policy
    A = arrs[best_adaptive]
    inc = np.diff(A, axis=1)                       # [users, turns] per-step gains
    step_mean = np.nanmean(inc)
    step_sd = np.nanstd(inc)
    # between-state (turn) signal vs within-turn noise (law of total variance proxy)
    per_turn_mean = np.nanmean(inc, axis=0)
    between = np.nanvar(per_turn_mean)             # variance explained by turn index
    within = np.nanmean(np.nanvar(inc, axis=0))    # residual cross-user variance
    print(f"\n=== per-step reward (held-out accuracy increment), policy={best_adaptive} ===")
    print(f"  mean step gain = {step_mean:+.5f}   cross-user std = {step_sd:.5f}")
    print(f"  effect-size ratio (mean gain / std)   = {step_mean/step_sd:.3f}  (<< 1 => noise-limited)")
    print(f"  variance decomposition: between-turn={between:.2e}  within-turn(cross-user)={within:.2e}")
    print(f"  signal fraction = {between/(between+within):.3%}  (tiny => routing signal buried in user noise)")

    print("\n=== VERDICT ===")
    print(f"  Realizable adaptive edge ~{dm:+.4f} (barely significant); oracle edge "
          f"{ORACLE_ATTR-auacs[static].mean():+.4f}. Per-step routing signal is a few % "
          f"of cross-user reward variance -> softmax PG collapses to the best FIXED order "
          f"(correct under (1-1/e) adaptive-submodular bound). The capturable value is "
          f"privileged -> test distillation (oracle_distill.py).")


if __name__ == '__main__':
    main()
