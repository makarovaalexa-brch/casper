"""Honest cost of pb2 folds. No surrogate. Just measure, then extrapolate."""
import os, sys, time, json
sys.path.insert(0, "scripts")
import numpy as np, torch
from set_mn import SetEncoder
from signed_latent import load_arena_base

torch.set_num_threads(os.cpu_count())
base = load_arena_base(); ni = base["ni"]
ck = torch.load(".cache/set_mn/pb2_best.pt", map_location="cpu")
NT = ck["student"]["item_emb.weight"].shape[0]
NLEV = ck["student"]["gamma.weight"].shape[0]
enc = SetEncoder(NT, token_mode="film", pool="attn", nlev=NLEV, nknow=0)
enc.load_state_dict(ck["student"], strict=False); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
dec = ck["student"]  # decoder frozen; Wd/bd live in base
# decoder weights: pull from checkpoint blob
Wd = torch.from_numpy(base["Wd"]) if "Wd" in base else None
print("catalog", ni, "tokens", NT, "levels", NLEV, flush=True)

# realistic state: a mid-interview profile. typical evidence-half length ~ tens of items.
# build a batch of B states each with L tokens, then time encode+decode.
def make_batch(B, L):
    ids = torch.randint(0, ni, (B, L))
    lvs = torch.randint(0, NLEV, (B, L))
    pad = torch.zeros(B, L, dtype=torch.bool)
    return ids, lvs, pad

# decoder from the SAME source the eval uses (z @ Wd.T + bd)
# find Wd/bd
import scipy.sparse  # noqa
Wd = None; bd = None
for k in ("decoder.weight","dec.weight"):
    pass
# the eval builds Wd,bd from a frozen decoder saved alongside; reuse arena base if present
Wd = torch.randn(ni, enc.d)  # PLACEHOLDER dims only affect matmul cost, not correctness of TIMING
bd = torch.zeros(ni)

for L in (30, 60, 120):
    B = 256
    ids, lvs, kn, pad = make_batch(B, L)
    # warmup
    with torch.no_grad():
        z = enc(ids, torch.zeros(B, L), pad, lvs, kn)
    t0 = time.time(); reps = 5
    with torch.no_grad():
        for _ in range(reps):
            z = enc(ids, torch.zeros(B, L), pad, lvs, kn)          # ENCODE
            s = z @ Wd.T + bd                                       # DECODE 18,430-way
            top = torch.topk(s, 10, dim=1).indices                 # top-10
    dt = (time.time() - t0) / reps
    per_state = dt / B
    print(f"L={L:3d}  batch {B}: {dt*1000:7.1f} ms/batch  ->  {per_state*1e6:7.1f} us / (state,candidate-fold+decode+top10)", flush=True)

print("\n--- extrapolation (per_state at L~60) ---", flush=True)
ps = per_state
grid_full = 150239 * 800 * ps / 3600
print(f"TRAINING GRID  150,239 states x 800 candidates (x1 answer)   = {grid_full:8.2f} h", flush=True)
# enumeration-EVOI at EVAL: 75k users x q turns x 800 candidates x ~2 answer levels (knows/level marginalized)
for q in (4, 8):
    ev = 75000 * q * 800 * 2 * ps / 3600
    print(f"EVAL enum-EVOI 75k users x {q} turns x 800 cand x 2 levels    = {ev:8.2f} h", flush=True)
# but m=V(z) TRAINING needs only post-answer STATES we choose to generate, NOT the full grid.
print(f"\nV(z) training states: we generate as many post-answer states as we want; cost is {ps*1e6:.0f} us each.", flush=True)
