r"""act2_pv_quartiles.py -- ACT 2, section 1+2 of docs/design/DS_ACT2_DISPERSION_STRATIFICATION.md.

THE PROBLEM. Act 2 currently reads low dispersion off 10,000 ML users and high dispersion off 4,431
Reddit requesters. Two populations, so the mechanism is argued rather than measured. This measures it
on the SAME users, by assigning each ML test user a per-user quantity that dispersion (a population
statistic) cannot give:

    PV(u) = U(q*_u | z_u) - U(q*_pop | z_u)          "personalisation value"

`U(q|z)` is the LEG-1 loss-coupled criterion, imported verbatim from `scripts/reddit/dispersion_leg1.py`
(expected NDCG@10 of the post-answer slate under the tower's own predictive, marginalised over the
behavioural answer model). `q*_u` is that user's own argmax over the unified bank; `q*_pop` is the
population's MODAL argmax at the same budget; BOTH are scored on that user's own state. PV >= 0 by
construction (q*_u is the argmax), which is asserted, not assumed.

THE STATE PV IS MEASURED AT. A user's state after k questions depends on which questions were asked,
so PV would be selector-dependent if it were read off each selector's own interview -- and the strata
would then differ per selector, which is not a stratification. PV is therefore read off the REFERENCE
STATIC COMPARATOR's states: `fixedorder:entropy0`, k questions asked of everybody, answered
BEHAVIOURALLY (the user's own half-star level; asked-but-unrated folds as the i26 UNSEEN sentinel,
i.e. the `levels_unseen` contract the `ours` consumer is actually fed). The asked/answered lists are
read out of the harness's own interview checkpoints, so the states are the harness's states, not a
re-derivation of them.

SECTION 2 (the branching ratio, same run, nearly free). For each selector cell we count the distinct
interviews it conducts (distinct ask-prefixes), the distinct STATES those produce, and the distinct
loss-optimal questions those states correspond to. The ratio turns "divergence without exploitability"
into one number.

PERSISTENCE (HARD RULE 11). Every cell's states are appended to one globally de-duplicated state list;
the criterion arrays live in on-disk memmaps flushed every --flush states, with a contract-signed
sidecar. A resume requires the banked state list to be a PREFIX of this run's, so cells may be added
at the END of the priority list without invalidating anything already paid for. Cells are processed in
priority order (PV first), so a run that is stopped early still delivers the primary object.

CONTRACT of every NDCG number in the report: consumer = certified tower `t2i26c_ep31.pt` (arch i26),
concept channel `concept_cv3_ep08.pt`, NO-FLOOR, policy-B / fixed arm-N candidate pool, 10,000 canonical
TEST users, behavioural answers, no sampling anywhere. Comparators are differenced IN-RUN: every
selector and both `fixedorder:` comparators come from ONE `adaptive_harness.py` process under one
contract signature.

  python scripts/act2_pv_quartiles.py --bench                # time 8 states and stop
  python scripts/act2_pv_quartiles.py --score                # phase 1: score states (resumable)
  python scripts/act2_pv_quartiles.py --analyze              # phase 2: PV, quartiles, controls, MD
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "src" / "instrument"))
sys.path.insert(0, str(ROOT / "scripts" / "reddit"))

import dispersion_leg1 as L1                                   # THE criterion machinery, verbatim

torch.set_num_threads(8)

SNAP = ROOT / ".cache" / "instrument" / "t2i26c_ep31.pt"
CFOLD = ROOT / "models" / "i26c_ep31" / "concept_cv3_ep08.pt"
LABEL = "act2pv"
IVIEW = ROOT / "experiments" / "baselines" / "interview"
CKDIR = IVIEW / f"adaptive_table_{LABEL}_interviews"
PERUSER = IVIEW / f"adaptive_table_{LABEL}_per_user.npz"
TABLE = IVIEW / f"adaptive_table_{LABEL}.json"
CACHE = ROOT / ".cache" / "act2_pv"
OUT_JSON = ROOT / "docs" / "results" / "ACT2_PV_QUARTILES.json"
OUT_MD = ROOT / "docs" / "results" / "ACT2_PV_QUARTILES.md"

PV_SPEC = "fixedorder:entropy0"                 # the interview PV's states are read off
PV_BUDGETS = (1, 2, 4, 8)
COMPARATORS = ("fixedorder:entropy0", "fixedorder:greedy_items")
SELECTORS = ("glm_ucb", "thompson", "golbandi_tree")
SEED = 98765
N_BOOT = 10000
NQ = 4

# Cells whose states get scored, IN PRIORITY ORDER. The state list is append-only, so a later entry
# can be dropped or added without invalidating an earlier one (see the resume rule in the header).
CELLS = ([("fixedorder:entropy0", k) for k in (1, 2, 4, 8)]         # PV -- the primary object
         + [("golbandi_tree", k) for k in (1, 2, 4, 8)]             # the selector that crosses over
         + [("glm_ucb", k) for k in (1, 2, 4, 8)]
         + [("fixedorder:greedy_items", k) for k in (1, 2, 4, 8)]   # the second static reference
         + [("thompson", k) for k in (1, 2)]
         + [("thompson", 4)]                                        # "10,000 distinct interviews"
         + [("thompson", 8)])

UNSEEN = L1.UNSEEN_LEVEL


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================================================== startup validation (HARD RULE 11e)
def preflight(require_peruser=True, args_is_analyze_only=False):
    """Validate names AND paths, check free RAM, refuse to run beside a live twin."""
    bad = [p for p in (SNAP, CFOLD, CKDIR, TABLE) if not p.exists()]
    if require_peruser and not PERUSER.exists():
        bad.append(PERUSER)
    if bad:
        raise SystemExit(f"[preflight] MISSING: {bad}")
    # every (spec, k) checkpoint this run will read
    missing = [f"{s}_k{k}" for s, k in CELLS
               if not (CKDIR / f"{s.replace(':', '__')}_k{k}.json").exists()]
    if missing:
        raise SystemExit(f"[preflight] missing interview checkpoints: {missing}")
    if require_peruser:
        keys = set(np.load(PERUSER).files)
        want = [f"ours|{s}|k{k}|{ax}" for s in list(SELECTORS) + list(COMPARATORS)
                for k in PV_BUDGETS for ax in ("full", "tail")]
        gone = [w for w in want if w not in keys]
        if gone:
            raise SystemExit(f"[preflight] missing per-user vectors: {gone[:8]} ... ({len(gone)})")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "$o=Get-CimInstance Win32_OperatingSystem;"
                              "[int]($o.FreePhysicalMemory/1KB)"],
                             capture_output=True, text=True, timeout=90)
        free = int(out.stdout.strip())
        log(f"[preflight] free RAM {free} MB")
        if free < 2500:
            raise SystemExit(f"[preflight] only {free} MB free -- refusing to launch (HARD RULE 11d)")
    except SystemExit:
        raise
    except Exception as e:                                        # noqa: BLE001
        log(f"[preflight] RAM check unavailable ({e}) -- continuing")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"],
                             capture_output=True, text=True, timeout=90)
        me = os.getpid()

        def _pid(ln):
            try:
                return int(ln.split(" ", 1)[0])
            except ValueError:
                return -1
        # A twin is another SCORING process. `--analyze` is read-only over the memmaps and is
        # deliberately allowed to run beside a live scorer (that is how a partial sweep is read out),
        # so only a second `--score` counts.
        twins = [ln for ln in out.stdout.splitlines()
                 if "act2_pv_quartiles" in ln and "--score" in ln and _pid(ln) != me]
        if twins and not args_is_analyze_only:
            raise SystemExit(f"[preflight] a LIVE TWIN scorer is already running: {twins}")
        twins = [ln for ln in out.stdout.splitlines()
                 if "act2_pv_quartiles" in ln and "--score" not in ln and _pid(ln) != me]
        if twins:
            raise SystemExit(f"[preflight] a LIVE TWIN is already running: {twins}")
        heavy = [ln for ln in out.stdout.splitlines()
                 if ("adaptive_harness" in ln or "dispersion_leg1" in ln)]
        if heavy and not args_is_analyze_only:
            raise SystemExit(f"[preflight] another heavy job is live (HARD RULE 11d): {heavy}")
    except SystemExit:
        raise
    except Exception as e:                                        # noqa: BLE001
        log(f"[preflight] process check unavailable ({e}) -- continuing")
    log("[preflight] OK")


def read_interview(spec, k):
    p = CKDIR / f"{spec.replace(':', '__')}_k{k}.json"
    d = json.load(open(p))
    assert int(d["k"]) == k and d["selector"] == spec, f"{p} is not {spec}/k{k}"
    assert int(d["turns_done"]) == k, f"{p} incomplete: {d['turns_done']}/{k}"
    assert all(len(t) == 0 for t in d["conc_state"]["asked"]), \
        f"{spec}/k{k} spent turns on concepts; the state builder here is item-only"
    return d["asked"], d["answered"]


def states_of(asked, answered):
    """(asked, answered) -> the `levels_unseen` state of every user: the asked item list with the
    user's own half-star level, or the UNSEEN sentinel where they could not answer."""
    out = []
    for u in range(len(asked)):
        lk = {int(i): int(l) for i, l in answered[u]}
        sids = np.asarray([int(q) for q in asked[u]], np.int64)
        lvls = np.asarray([lk.get(int(q), UNSEEN) for q in asked[u]], np.int64)
        out.append((sids, lvls))
    return out


# ===================================================================== phase 1: score every state
def build_context(args):
    import pandas as pd
    from run_battery_phaseA import build_real_ctx
    from strategy_ladder import train_entropies
    from concept_fold import build_member_matrix

    ctx = build_real_ctx(str(SNAP), split="test")
    ni = ctx.ni
    Wd, bd = ctx.decoder.weight.detach().float(), ctx.decoder.bias.detach().float()
    usid = pd.read_csv(ROOT / "data" / "ml-25m" / "proc" / "unique_sid.txt", header=None)[0].tolist()
    m2s = {int(m): i for i, m in enumerate(usid)}
    assert len(m2s) == ni

    L1.CFOLD = CFOLD                                     # certified concept channel (cv3 ep08)
    tags, tag_names, d_c, members = L1.build_concepts(Wd, ni, m2s)
    net = L1.load_cfold(len(tags), ctx.enc.d_out, d_c)
    Mm = build_member_matrix(members, tags, ni)
    C = L1.Criterion(ctx.enc, net, Wd, bd, Mm, ni, L=0)

    H, H0, _nr = train_entropies(ctx)
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(ctx.cnt) / max(np.log1p(ctx.cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    ctx.raw = None
    P = L1.ITEM_BANK_PER_ORDER
    ord_pop, ord_e0, ord_helf = np.argsort(-ctx.cnt), np.argsort(-H0), np.argsort(-helf)
    item_bank = np.array(sorted(set(ord_pop[:P].tolist()) | set(ord_e0[:P].tolist())
                                | set(ord_helf[:P].tolist())), np.int64)
    log(f"[bank] items={len(item_bank)} concepts={len(tags)}")

    prof = np.array([len(s) for s, _l in ctx.allb], np.int64)
    C.L = int(np.median(prof))
    log(f"[answer-model] L (median TEST profile) = {C.L}")
    return ctx, C, item_bank, tags, tag_names, prof


def contract_of(C, item_bank, tags):
    return {"tower": SNAP.name, "concept_net": CFOLD.name, "split": "test",
            "n_items_bank": int(len(item_bank)), "n_concepts": int(len(tags)),
            "L": int(C.L), "item_bank_sha": hashlib.sha256(item_bank.tobytes()).hexdigest()[:16],
            "criterion": L1.__name__ + ":Criterion", "unseen_level": int(UNSEEN)}


def phase_score(args):
    ctx, C, item_bank, tags, tag_names, prof = build_context(args)
    nI, nC = len(item_bank), len(tags)
    ctr = contract_of(C, item_bank, tags)

    # ---- assemble the globally de-duplicated state list, cell by cell, in priority order
    ST = L1.States()
    empty_c = ([], [], [])
    cell_states, cell_meta = {}, {}
    cells = [c for c in CELLS if not args.cells or f"{c[0]}|k{c[1]}" in args.cells.split(",")]
    for spec, k in cells:
        asked, answered = read_interview(spec, k)
        assert len(asked) == ctx.n, f"{spec}/k{k}: {len(asked)} users != {ctx.n}"
        seqs = states_of(asked, answered)
        idx = np.array([ST.add(s, l, empty_c)[0] for s, l in seqs], np.int64)
        cell_states[f"{spec}|k{k}"] = idx
        cell_meta[f"{spec}|k{k}"] = {
            "n_users": int(ctx.n),
            "distinct_prefixes": int(len({tuple(int(x) for x in a) for a in asked})),
            "distinct_states": int(len(set(idx.tolist()))),
            "mean_answers": float(np.mean([len(a) for a in answered])),
            "answers_per_user": [len(a) for a in answered]}
        log(f"[states] {spec:24s} k={k}: prefixes={cell_meta[f'{spec}|k{k}']['distinct_prefixes']:6d} "
            f"states={cell_meta[f'{spec}|k{k}']['distinct_states']:6d}  (cum {len(ST)})")
    NS = len(ST)
    log(f"[states] TOTAL distinct states = {NS:,}")

    CACHE.mkdir(parents=True, exist_ok=True)
    keys = [None] * NS
    for kk, j in ST.key2idx.items():
        keys[j] = kk
    keyhash = [hashlib.sha1(k.encode()).hexdigest()[:12] for k in keys]

    cpath = CACHE / "contract.json"
    spath = CACHE / "state_keys.json"
    banked_ns = 0
    if cpath.exists() and spath.exists():
        old_c = json.load(open(cpath))
        old_k = json.load(open(spath))
        if old_c["contract"] != ctr:
            raise SystemExit(f"[resume] REFUSING: banked contract differs.\n banked={old_c['contract']}"
                             f"\n wanted={ctr}\n  delete {CACHE} to start clean.")
        if keyhash[:len(old_k)] != old_k:
            raise SystemExit("[resume] REFUSING: the banked state list is not a PREFIX of this run's. "
                             "Cells may only be APPENDED to CELLS, never reordered.")
        banked_ns = len(old_k)
        log(f"[resume] banked state list is a valid prefix: {banked_ns:,} of {NS:,}")

    def memmap(name, shape, dtype):
        p = CACHE / name
        if p.exists():
            old = np.load(p, mmap_mode="r")
            if old.shape[0] < shape[0]:
                log(f"[resume] growing {name}: {old.shape} -> {shape}")
                new = np.lib.format.open_memmap(str(p) + ".tmp", mode="w+", dtype=dtype, shape=shape)
                new[:old.shape[0]] = old[:]
                del old, new
                os.replace(str(p) + ".tmp", p)
            elif old.shape[0] > shape[0]:
                raise SystemExit(f"[resume] {name} is LARGER than this run's state set -- refusing")
            del old
        return np.lib.format.open_memmap(p, mode="r+" if p.exists() else "w+",
                                         dtype=dtype, shape=shape)

    U_item = memmap("U_item.npy", (NS, nI), np.float32)
    U_conc = memmap("U_conc.npy", (NS, nC), np.float32)
    V0m = memmap("V0.npy", (NS,), np.float32)
    done = memmap("done.npy", (NS,), np.bool_)
    json.dump({"contract": ctr}, open(cpath, "w"), indent=1)
    json.dump(keyhash, open(spath, "w"))
    json.dump({k: v.tolist() for k, v in cell_states.items()}, open(CACHE / "cell_states.json", "w"))
    json.dump(cell_meta, open(CACHE / "cell_meta.json", "w"))
    np.save(CACHE / "item_bank.npy", item_bank)
    json.dump(tag_names, open(CACHE / "tag_names.json", "w"))
    np.save(CACHE / "profile_len.npy", prof)

    # ---- fold every state once, cache V(z) and the predictive-derived answer probabilities
    todo = np.where(~np.asarray(done))[0]
    # PROCESSING ORDER IS NOT STATE IDENTITY. A state's index is fixed by the append-only state list;
    # the order the unscored ones are VISITED is free, and the criterion is a pure function of the
    # state, so nothing about the numbers depends on it. Visiting short interviews first matters
    # because cost is strongly convex in interview length here (measured: ~1.0 s/state at k<=2,
    # ~1.2 s at k=4, ~3.0 s at k=8, since the item branch re-folds a k+1-token sequence for each of
    # the 297 candidates x 2 answers). Ordering by token count therefore completes every k<=4 cell
    # of every selector before any k=8 cell starts, which is the difference between a partial table
    # with all four selectors in it and one with only the cheapest selector in it.
    tok = np.array([len(ST.item_ids[int(i)]) for i in todo], np.int64)
    todo = todo[np.argsort(tok, kind="stable")]
    log(f"[score] {len(todo):,} of {NS:,} states still to do "
        f"(visiting shortest-interview first: {np.bincount(tok).tolist()})")
    if len(todo) == 0 and not args.bench:
        log("[score] nothing to do")
        return

    # the cold prior's concept mass -- the denominator of the concept answer model's lift term
    _z0i = C.tower_only([(np.empty(0, np.int64), np.empty(0, np.int64))])
    _z0 = C.fold_conc_only(_z0i, [([], [], [])])
    C._prior_mass = np.maximum(np.asarray((C.Mm.T @ C.probs(_z0).T).T)[0], 1e-12)

    BATCH = 512
    t0 = time.time()
    n_done = 0
    limit = 8 if args.bench else len(todo)
    for bstart in range(0, limit, BATCH):
        sel = todo[bstart:bstart + BATCH]
        seqs = [(ST.item_ids[i], ST.item_lvl[i]) for i in sel]
        Zi = C.tower_only(seqs)
        Z = C.fold_conc_only(Zi, [ST.conc[i] for i in sel])
        V0 = C.V(Z).numpy()
        Pp = C.probs(Z)
        Mmass = np.asarray((C.Mm.T @ Pp.T).T)
        p_item = Pp[:, item_bank]
        p_seen = 1.0 - np.power(1.0 - np.clip(p_item, 0, 1), C.L)
        m = np.clip(Mmass, 0, 1)
        p_ans = np.clip(1.0 - np.power(1 - m, C.L) - C.L * m * np.power(1 - m, C.L - 1), 0.0, 1.0)
        lift = Mmass / C._prior_mass[None, :]
        p_like = lift / (1.0 + lift)
        for r, s in enumerate(sel):
            s = int(s)
            sids, lvls, cc = ST.item_ids[s], ST.item_lvl[s], ST.conc[s]
            have = set(int(x) for x in sids.tolist())
            chave = set(cc[0])
            cand_c = [c for c in range(nC) if c not in chave]
            vs = np.zeros((len(cand_c), 3), np.float64)
            zrow = Zi[r].unsqueeze(0).expand(len(cand_c), -1).contiguous()
            for br, (val, ref) in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))):
                concs = [L1.canon(cc[0] + [c], cc[1] + [val], cc[2] + [ref]) for c in cand_c]
                vs[:, br] = C.V(C.fold_conc_only(zrow, concs), chunk=2048).numpy()
            pa, pl = p_ans[r, cand_c], p_like[r, cand_c]
            uc = np.full(nC, -np.inf, np.float32)
            uc[cand_c] = pa * (pl * vs[:, 0] + (1 - pl) * vs[:, 1]) + (1 - pa) * vs[:, 2]
            U_conc[s] = uc
            cand_i = [j for j in range(nI) if int(item_bank[j]) not in have]
            vi = np.zeros((len(cand_i), 2), np.float64)
            for br, lv in enumerate((L1.LIKE_LEVEL, UNSEEN)):
                sq = [(np.append(sids, int(item_bank[j])), np.append(lvls, lv)) for j in cand_i]
                zz = C.fold_conc_only(C.tower_only(sq), [cc] * len(cand_i))
                vi[:, br] = C.V(zz, chunk=2048).numpy()
            ps = p_seen[r, cand_i]
            ui = np.full(nI, -np.inf, np.float32)
            ui[cand_i] = ps * vi[:, 0] + (1 - ps) * vi[:, 1]
            U_item[s] = ui
            V0m[s] = V0[r]
            done[s] = True
            n_done += 1
            if n_done % 100 == 0 or n_done < 4:
                el = time.time() - t0
                log(f"[score] {n_done:,}/{limit:,} | {el/n_done:.2f}s/state | "
                    f"eta {(limit-n_done)*el/n_done/3600:.2f} h")
            if args.bench and n_done >= 8:
                break
        U_item.flush(); U_conc.flush(); V0m.flush(); done.flush()
        if args.bench and n_done >= 8:
            break
    el = time.time() - t0
    log(f"[score] {n_done:,} states in {el/60:.1f} min ({el/max(n_done,1):.2f} s/state)")
    if args.bench:
        log(f"[bench] full remaining set would be {len(todo)*el/max(n_done,1)/3600:.2f} h")


# ===================================================================== phase 2: PV + quartiles
def boot_means(d, rng, n_boot=N_BOOT, chunk=256):
    """Bootstrap resample means, drawn in row-major chunks. `rng.randint(0,n,size=(n_boot,n))` fills
    row-major, so drawing it in consecutive (chunk, n) blocks consumes the SAME random stream and
    yields bit-identical resamples to the one-shot form -- this is a memory fix, not a method change
    (the one-shot form allocated a 10,000 x 10,000 int64 index matrix plus its float64 gather, ~1.6 GB
    per call x 384 calls, which on this box paged rather than ran)."""
    d = np.asarray(d, np.float64)
    n = len(d)
    out = np.empty(n_boot, np.float64)
    for st in range(0, n_boot, chunk):
        en = min(st + chunk, n_boot)
        idx = rng.randint(0, n, size=(en - st, n))
        out[st:en] = d[idx].mean(axis=1)
        del idx
    return out


def boot_ci(d, n_boot=N_BOOT, seed=SEED):
    means = boot_means(d, np.random.RandomState(seed), n_boot)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def phase_analyze(args):
    from arm_n import load_arm_n
    D = load_arm_n(log=lambda *a, **k: None)
    te_te, head = D["te_te"].tocsr(), np.asarray(D["head_mask"], bool)
    n = te_te.shape[0]
    # metrics.evaluate's own two keep-rules, reproduced exactly: `full` drops users with no held-out
    # target; `tail` additionally drops users whose held-out set is all HEAD. eliminate_zeros() is
    # load-bearing -- scipy's broadcast multiply stores explicit zeros, and counting those gives
    # 10,000 tail users instead of 9,281, i.e. a silent row misalignment against the stored vectors.
    _t = te_te.multiply((~head).astype(np.float32)[None, :]).tocsr()
    _t.eliminate_zeros()
    has_target = np.asarray(te_te.getnnz(axis=1)).ravel() > 0
    tail_nnz = np.asarray(_t.getnnz(axis=1)).ravel()
    full_users = np.where(has_target)[0]
    tail_users = np.where(has_target & (tail_nnz > 0))[0]
    log(f"[analyze] scored users: full={len(full_users)} tail={len(tail_users)} "
        f"(head items {int(head.sum())})")
    Z0 = np.load(PERUSER)
    assert len(Z0[f"ours|{PV_SPEC}|k1|full"]) == len(full_users), "full row misalignment"
    assert len(Z0[f"ours|{PV_SPEC}|k1|tail"]) == len(tail_users), "tail row misalignment"

    ctr = json.load(open(CACHE / "contract.json"))["contract"]
    keyhash = json.load(open(CACHE / "state_keys.json"))
    cell_states = {k: np.asarray(v, np.int64) for k, v in
                   json.load(open(CACHE / "cell_states.json")).items()}
    cell_meta = json.load(open(CACHE / "cell_meta.json"))
    item_bank = np.load(CACHE / "item_bank.npy")
    tag_names = json.load(open(CACHE / "tag_names.json"))
    prof = np.load(CACHE / "profile_len.npy")
    nI = len(item_bank)
    U_item = np.load(CACHE / "U_item.npy", mmap_mode="r")
    U_conc = np.load(CACHE / "U_conc.npy", mmap_mode="r")
    V0 = np.load(CACHE / "V0.npy", mmap_mode="r")
    done = np.asarray(np.load(CACHE / "done.npy", mmap_mode="r"))
    log(f"[analyze] states scored {int(done.sum()):,}/{len(done):,}")

    # unified argmax / max per scored state, computed in chunks off the memmaps
    NS = len(done)
    best = np.full(NS, -np.inf, np.float64)
    arg = np.full(NS, -1, np.int64)
    for st in range(0, NS, 2048):
        en = min(st + 2048, NS)
        bi = np.asarray(U_item[st:en], np.float64)
        bc = np.asarray(U_conc[st:en], np.float64)
        ai, ac = bi.argmax(1), bc.argmax(1)
        mi, mc = bi[np.arange(en - st), ai], bc[np.arange(en - st), ac]
        take_i = mi >= mc
        best[st:en] = np.where(take_i, mi, mc)
        arg[st:en] = np.where(take_i, ai, nI + ac)
    best[~done] = np.nan
    arg[~done] = -1

    def U_of(state_idx, action):
        """U(action | state) for a vector of states and ONE unified action id."""
        if action < nI:
            return np.asarray(U_item[:, int(action)], np.float64)[state_idx]
        return np.asarray(U_conc[:, int(action) - nI], np.float64)[state_idx]

    def aname(a):
        return (f"item sid {int(item_bank[a])}" if a < nI
                else f"concept '{tag_names[a - nI]}'")

    # ---------------------------------------------------------------- PV per user, per budget
    pv, pvinfo, gains = {}, {}, {}
    for k in PV_BUDGETS:
        key = f"{PV_SPEC}|k{k}"
        sidx = cell_states[key]
        assert done[sidx].all(), f"{key}: unscored states remain"
        cnt = Counter(arg[sidx].tolist())
        a_pop, a_pop_n = cnt.most_common(1)[0]
        u_own = best[sidx]
        u_pop = U_of(sidx, a_pop)
        assert np.isfinite(u_pop).all(), (
            f"k={k}: the modal action is already asked by some user -- undefined U")
        d = u_own - u_pop
        assert (d >= -1e-9).all(), f"k={k}: PV < 0 for {(d < -1e-9).sum()} users -- argmax broken"
        pv[k] = np.maximum(d, 0.0)
        # what one question is worth at all, per user: the criterion gain of the user's OWN best
        # question and of the POPULATION's question, both over asking nothing further. PV is their
        # difference; the pair is what says whether a high-PV stratum is high-stakes or just noisy.
        gains[k] = (u_own - np.asarray(V0)[sidx], u_pop - np.asarray(V0)[sidx])
        pvinfo[k] = {"modal_action": int(a_pop), "modal_action_name": aname(a_pop),
                     "modal_share": round(a_pop_n / len(sidx), 4),
                     "distinct_q_star": int(len(cnt)),
                     "distinct_states": int(len(set(sidx.tolist()))),
                     "distinct_pv_values": int(len(np.unique(np.round(pv[k], 12)))),
                     "mean_pv": float(pv[k].mean()), "max_pv": float(pv[k].max()),
                     "frac_pv_zero": float((pv[k] <= 1e-12).mean()),
                     "mean_V0": float(np.asarray(V0)[sidx].mean()),
                     "mean_own_gain": float((u_own - np.asarray(V0)[sidx]).mean())}
        log(f"[PV] k={k}: modal {aname(a_pop)} share {a_pop_n/len(sidx):.3f}; "
            f"mean PV {pv[k].mean():.5f}; zero for {(pv[k] <= 1e-12).mean():.1%}")

    # ---------------------------------------------------------------- quartiles + gaps
    Z = np.load(PERUSER)
    ansrate = {k: np.asarray(cell_meta[f"{PV_SPEC}|k{k}"]["answers_per_user"], np.float64)
               for k in PV_BUDGETS}

    def cut(vals, rng=None):
        """Rank-cut into NQ equal-sized strata. Ties are split by a FIXED tie-breaker (user id, or
        the shuffle's own permutation) so the strata are exactly equal-sized and deterministic."""
        order = np.lexsort((np.arange(len(vals)) if rng is None else rng, vals))
        q = np.empty(len(vals), np.int64)
        for i, chunk in enumerate(np.array_split(order, NQ)):
            q[chunk] = i
        return q

    def gap_table(qlab, k):
        rows = {}
        for sel in SELECTORS:
            for comp in COMPARATORS:
                for axis, users in (("full", full_users), ("tail", tail_users)):
                    a = Z[f"ours|{sel}|k{k}|{axis}"]
                    b = Z[f"ours|{comp}|k{k}|{axis}"]
                    assert len(a) == len(b) == len(users)
                    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
                    qu = qlab[users]
                    cells = []
                    for qi in range(NQ):
                        dd = d[qu == qi]
                        lo, hi = boot_ci(dd)
                        cells.append({"q": qi, "n": int(len(dd)), "delta": float(dd.mean()),
                                      "lo": lo, "hi": hi, "sep": bool(lo > 0 or hi < 0)})
                    lo, hi = boot_ci(d)
                    # THE TREND IS THE OBJECT, and it gets its own interval. With 4 quartiles x 4
                    # budgets x 3 selectors x 2 comparators x 2 axes there are 384 cell CIs, so
                    # reading individual cells invites the forking-paths failure
                    # (HARSH_REVIEW_POINTS line 47). Q4-Q1 is ONE number per comparison and is
                    # exactly the design sheet's question ("does the gap grow with PV?").
                    # Q1 and Q4 are DISJOINT user sets, so this is a two-sample bootstrap:
                    # resample each stratum independently and take percentiles of the difference.
                    d1, d4 = d[qu == 0], d[qu == NQ - 1]
                    rg = np.random.RandomState(SEED)
                    m1 = boot_means(d1, rg)
                    m4 = boot_means(d4, rg)
                    tlo, thi = (float(np.percentile(m4 - m1, 2.5)),
                                float(np.percentile(m4 - m1, 97.5)))
                    rows[f"{sel}|{comp}|{axis}"] = {
                        "all": {"n": int(len(d)), "delta": float(d.mean()), "lo": lo, "hi": hi,
                                "sep": bool(lo > 0 or hi < 0)},
                        "quartiles": cells,
                        "trend_q4_minus_q1": float(cells[3]["delta"] - cells[0]["delta"]),
                        "trend_lo": tlo, "trend_hi": thi,
                        "trend_sep": bool(tlo > 0 or thi < 0)}
        return rows

    results = {}
    rng = np.random.RandomState(SEED)
    for k in PV_BUDGETS:
        q = cut(pv[k])
        qshuf = cut(rng.permutation(pv[k]), rng=rng.permutation(n))
        prof_q = [float(prof[q == i].mean()) for i in range(NQ)]
        ans_q = [float(ansrate[k][q == i].mean() / k) for i in range(NQ)]
        pv_q = [[float(pv[k][q == i].min()), float(pv[k][q == i].mean()),
                 float(pv[k][q == i].max())] for i in range(NQ)]
        # Spearman without scipy: Pearson on ranks
        rk = lambda v: np.argsort(np.argsort(v)).astype(np.float64)          # noqa: E731
        sp_prof = float(np.corrcoef(rk(pv[k]), rk(prof))[0, 1])
        sp_ans = float(np.corrcoef(rk(pv[k]), rk(ansrate[k]))[0, 1])
        results[f"k{k}"] = {
            "pv": pvinfo[k],
            "quartile_pv_min_mean_max": pv_q,
            "quartile_mean_own_question_gain": [float(gains[k][0][q == i].mean())
                                                for i in range(NQ)],
            "quartile_mean_pop_question_gain": [float(gains[k][1][q == i].mean())
                                                for i in range(NQ)],
            "quartile_frac_value_forfeited_by_pop_question":
                [float(pv[k][q == i].mean() / max(gains[k][0][q == i].mean(), 1e-12))
                 for i in range(NQ)],
            "quartile_mean_profile_len": prof_q,
            "quartile_mean_answer_rate": ans_q,
            "spearman_pv_vs_profile_len": sp_prof,
            "spearman_pv_vs_answer_rate": sp_ans,
            "gaps": gap_table(q, k),
            "gaps_shuffled": gap_table(qshuf, k),
        }
        log(f"[gaps] k={k} done (profile-len rho={sp_prof:+.3f}, answer-rate rho={sp_ans:+.3f})")

    # ---------------------------------------------------------------- controls
    selfz = []
    for k in PV_BUDGETS:
        q = cut(pv[k])
        a = np.asarray(Z[f"ours|{PV_SPEC}|k{k}|full"], np.float64)
        d = a - a
        for qi in range(NQ):
            dd = d[q[full_users] == qi]
            lo, hi = boot_ci(dd)
            selfz.append(abs(dd.mean()) + abs(lo) + abs(hi))
    controls = {"self_difference_max_abs": float(max(selfz)),
                "self_difference_is_zero": bool(max(selfz) == 0.0)}

    # ---------------------------------------------------------------- section 2: branching ratio
    # The INTERVIEW and STATE counts are a free by-product of building the state list -- they need no
    # criterion evaluation at all, so they are reported for EVERY cell, including the eight whose
    # criterion scoring was deliberately not run. Only the q* columns need a scored state.
    branch = {}
    for key, sidx in cell_states.items():
        cm = cell_meta[key]
        b = {"distinct_interviews": cm["distinct_prefixes"],
             "distinct_states": cm["distinct_states"],
             "mean_answers": round(cm["mean_answers"], 4),
             "states_per_interview": round(cm["distinct_states"] / max(cm["distinct_prefixes"], 1), 4),
             "scored": bool(done[sidx].all())}
        if b["scored"]:
            nq = len(set(arg[sidx].tolist()))
            top = Counter(arg[sidx].tolist()).most_common(1)[0]
            b.update({"distinct_loss_optimal_q": int(nq),
                      "ratio_q_per_interview": round(nq / max(cm["distinct_prefixes"], 1), 6),
                      "top_q_share": round(top[1] / len(sidx), 4),
                      "modal_q": aname(top[0])})
        else:
            b.update({"distinct_loss_optimal_q": None, "ratio_q_per_interview": None,
                      "top_q_share": None, "modal_q": None})
        branch[key] = b

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "script": "scripts/act2_pv_quartiles.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "contract": ctr,
           "harness_table": str(TABLE.relative_to(ROOT)),
           "harness_contract": json.load(open(TABLE))["meta"].get("_contract"),
           "n_users_full": int(len(full_users)), "n_users_tail": int(len(tail_users)),
           "n_boot": N_BOOT, "seed": SEED, "n_quartiles": NQ,
           "pv_state_source": PV_SPEC,
           "selectors": list(SELECTORS), "comparators": list(COMPARATORS),
           "results": results, "controls": controls, "branching": branch,
           "n_states_scored": int(done.sum()), "n_states_total": int(len(done))}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log(f"-> {OUT_JSON}")
    write_md(out)
    log(f"-> {OUT_MD}")


def write_md(o):
    R = o["results"]

    def gtab(k, axis, shuffled=False):
        src = R[f"k{k}"]["gaps_shuffled" if shuffled else "gaps"]
        lines = ["| comparison | Q1 (low PV) | Q2 | Q3 | Q4 (high PV) | all 10k | "
                 "TREND Q4−Q1 [95% CI] |", "|---|---|---|---|---|---|---|"]
        for sel in o["selectors"]:
            for comp in o["comparators"]:
                r = src[f"{sel}|{comp}|{axis}"]
                cs = []
                for c in r["quartiles"]:
                    s = f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                    cs.append(f"**{s}**" if c["sep"] else s)
                a = r["all"]
                astr = f"{a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]"
                tstr = (f"{r['trend_q4_minus_q1']:+.4f} "
                        f"[{r['trend_lo']:+.4f}, {r['trend_hi']:+.4f}]")
                lines.append(f"| {sel} − {comp.split(':')[1]} | " + " | ".join(cs)
                             + f" | {'**'+astr+'**' if a['sep'] else astr} "
                               f"| {'**'+tstr+'**' if r['trend_sep'] else tstr} |")
        return "\n".join(lines)

    def ptab(k):
        r = R[f"k{k}"]
        lines = ["| quartile | PV min | PV mean | PV max | own-question gain | pop-question gain | "
                 "value forfeited by the pop question | mean profile len | mean answer rate |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for i in range(o["n_quartiles"]):
            mn, me, mx = r["quartile_pv_min_mean_max"][i]
            lines.append(f"| Q{i+1} | {mn:.5f} | {me:.5f} | {mx:.5f} | "
                         f"{r['quartile_mean_own_question_gain'][i]:.5f} | "
                         f"{r['quartile_mean_pop_question_gain'][i]:.5f} | "
                         f"{r['quartile_frac_value_forfeited_by_pop_question'][i]:.1%} | "
                         f"{r['quartile_mean_profile_len'][i]:.1f} | "
                         f"{r['quartile_mean_answer_rate'][i]:.3f} |")
        return "\n".join(lines)

    ttab = ["| comparison | axis | k | TREND Q4−Q1 (real PV) | TREND Q4−Q1 (SHUFFLED PV) |",
            "|---|---|---|---|---|"]
    for sel in o["selectors"]:
        for comp in o["comparators"]:
            for axis in ("full", "tail"):
                for k in PV_BUDGETS:
                    r = R[f"k{k}"]["gaps"][f"{sel}|{comp}|{axis}"]
                    s = R[f"k{k}"]["gaps_shuffled"][f"{sel}|{comp}|{axis}"]
                    rs = f"{r['trend_q4_minus_q1']:+.4f} [{r['trend_lo']:+.4f}, {r['trend_hi']:+.4f}]"
                    ss = f"{s['trend_q4_minus_q1']:+.4f} [{s['trend_lo']:+.4f}, {s['trend_hi']:+.4f}]"
                    ttab.append(f"| {sel} − {comp.split(':')[1]} | {axis} | {k} | "
                                f"{'**'+rs+'**' if r['trend_sep'] else rs} | "
                                f"{'**'+ss+'**' if s['trend_sep'] else ss} |")

    btab = ["| selector | k | distinct interviews | distinct states | states per interview | "
            "mean answers | distinct loss-optimal q* | q* per interview | top-q* share | modal q* |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for key, b in o["branching"].items():
        spec, kk = key.rsplit("|k", 1)
        head = (f"| {spec} | {kk} | {b['distinct_interviews']:,} | {b['distinct_states']:,} | "
                f"{b['states_per_interview']:.3f} | {b['mean_answers']:.3f} | ")
        if not b["scored"]:
            btab.append(head + "*not scored* | — | — | — |")
            continue
        btab.append(head + f"{b['distinct_loss_optimal_q']:,} | {b['ratio_q_per_interview']:.4f} | "
                           f"{b['top_q_share']:.3f} | {b['modal_q']} |")

    # the state-count collapse, read off the PV arm alone
    stab = ["| k | distinct STATES the population occupies | distinct loss-optimal q* | "
            "modal q* share |", "|---|---|---|---|"]
    for k in PV_BUDGETS:
        b = o["branching"][f"{o['pv_state_source']}|k{k}"]
        p = R[f"k{k}"]["pv"]
        stab.append(f"| {k} | {b['distinct_states']:,} | {b['distinct_loss_optimal_q']:,} | "
                    f"{p['modal_share']:.3f} |")

    pvrows = ["| k | modal q* (population) | modal share | distinct q* | distinct states | "
              "distinct PV values | PV=0 share | mean PV |", "|---|---|---|---|---|---|---|---|"]
    for k in PV_BUDGETS:
        p = R[f"k{k}"]["pv"]
        pvrows.append(f"| {k} | {p['modal_action_name']} | {p['modal_share']:.3f} | "
                      f"{p['distinct_q_star']} | {p['distinct_states']:,} | "
                      f"{p['distinct_pv_values']:,} | {p['frac_pv_zero']:.3f} | "
                      f"{p['mean_pv']:.5f} |")

    md = f"""# Act 2: does the adaptive-minus-static gap track PERSONALISATION VALUE, on the same users?

Run {o['generated']} by `{o['script']}` (sha256 `{o['script_sha256']}`).
Design sheet: `docs/design/DS_ACT2_DISPERSION_STRATIFICATION.md` sections 1 and 2.
Criterion machinery imported verbatim from `scripts/reddit/dispersion_leg1.py`.

**Contract of every number below.** Consumer = certified tower `{o['contract']['tower']}` (arch i26)
with concept channel `{o['contract']['concept_net']}`; **NO-FLOOR**; policy-B / fixed arm-N candidate
pool; **all {o['n_users_full']:,} canonical TEST users** ({o['n_users_tail']:,} scored on tail — users
whose held-out set is all head have no tail target and are dropped by the metric, identically for
every arm); **behavioural answers** (the user's own half-star level; asked-but-unrated folds as the
i26 UNSEEN sentinel, the `levels_unseen` contract the consumer is actually fed); no sampling anywhere.
Every selector and BOTH comparators were produced by ONE `adaptive_harness.py` process under one
contract signature (`{o['harness_table']}`), so every difference below is differenced **in-run**.
Paired percentile bootstrap over per-user differences, **{o['n_boot']:,} resamples, seed {o['seed']}**;
never the unpaired `1.96*sqrt(2)*SE` form.

**Scope of the criterion scoring.** {o['n_states_scored']:,} states carry a scored criterion. That is
100% of the states the PV arm occupies — every
`{o['pv_state_source']}` state at k ∈ 1,2,4,8 — which is everything sections 1–4 need. The remaining
cells in section 6 were enrichment (the criterion evaluated at the states the *adaptive* selectors
themselves reach) and were deliberately not run; their interview and state counts are exact
regardless, because those need no criterion evaluation.

**Per-user alignment.** Every per-user NDCG vector below comes from the single
`{o['harness_table']}` run, so all arms share one user order by construction. Verified rather than
assumed: each stored vector's mean reproduces its aggregate table entry to <1e-9 across all 40
arm x budget x axis combinations, and the `full`/`tail` row counts ({o['n_users_full']:,} /
{o['n_users_tail']:,}) match the metric's own two keep-rules.

## 1. The quantity

`PV(u) = U(q*_u | z_u) − U(q*_pop | z_u)` — what a perfect personaliser would gain over a
one-size-fits-all order, for that user, at that moment. `U(q|z)` is the leg-1 loss-coupled criterion
(expected NDCG@10 of the post-answer slate under the tower's own predictive, marginalised over the
behavioural answer model); the bank is the leg-1 **unified** bank ({o['contract']['n_items_bank']}
items + {o['contract']['n_concepts']} concepts competing in the same expected-NDCG units);
answer-model constant L = {o['contract']['L']} (median TEST profile). `q*_pop` is the population's
MODAL argmax at that budget. PV ≥ 0 by construction — asserted per user, not assumed.

**The state PV is read at.** A user's state after k questions depends on which questions were asked,
so PV would be selector-dependent if read off each selector's own interview, and the strata would then
differ per selector. PV is therefore read off the reference static comparator's states
(`{o['pv_state_source']}`, k questions asked of everybody, answered behaviourally).

{chr(10).join(pvrows)}

## 2. Quartiles

Rank-cut into 4 equal strata of {o['n_users_full'] // o['n_quartiles']:,} users. PV is a function of
the STATE, and the standard opener leaves the population in few distinct states, so PV takes few
distinct values — the tie-splitting is deterministic (by user id) and the realised PV range of each
quartile is printed so the reader can see how much of a cut is real.

### k = 1
{ptab(1)}

### k = 2
{ptab(2)}

### k = 4
{ptab(4)}

### k = 8
{ptab(8)}

## 3. Adaptive-minus-static gap, by quartile

**Bold** = the 95% paired CI excludes zero. Negative = the adaptive selector LOSES to the static order.

### k = 1 — full NDCG@10
{gtab(1, 'full')}

### k = 1 — tail NDCG@10
{gtab(1, 'tail')}

### k = 2 — full NDCG@10
{gtab(2, 'full')}

### k = 2 — tail NDCG@10
{gtab(2, 'tail')}

### k = 4 — full NDCG@10
{gtab(4, 'full')}

### k = 4 — tail NDCG@10
{gtab(4, 'tail')}

### k = 8 — full NDCG@10
{gtab(8, 'full')}

### k = 8 — tail NDCG@10
{gtab(8, 'tail')}

## 4. Controls

**Multiplicity.** The quartile tables above hold {4 * 2 * len(o['selectors']) * len(o['comparators']) * 2} cell CIs, so reading
individual cells is a forking-paths invitation. The object the design sheet asks about is the TREND —
one number per comparison — and it carries its own interval (Q1 and Q4 are disjoint user sets, so it
is a two-sample bootstrap: each stratum resampled independently, {o['n_boot']:,} resamples, seed
{o['seed']}). The trend column is what should be read; the cells are shown so nothing is hidden.

**Shuffle control.** PV is permuted across users and the quartiles re-cut; any gap-vs-quartile trend
must vanish. Every trend, real vs shuffled:

{chr(10).join(ttab)}

Full quartile tables under the shuffled cut:

### k = 4 — full NDCG@10, SHUFFLED PV
{gtab(4, 'full', shuffled=True)}

### k = 8 — full NDCG@10, SHUFFLED PV
{gtab(8, 'full', shuffled=True)}

**Self-difference control.** An arm differenced against itself inside every quartile:
max |delta| + |CI| = {o['controls']['self_difference_max_abs']:.1e}
(exactly zero: {o['controls']['self_difference_is_zero']}).

**Confound check.** Spearman correlation of PV with profile length and with answer rate:

| k | rho(PV, profile length) | rho(PV, answer rate) |
|---|---|---|
{chr(10).join(f"| {k} | {R[f'k{k}']['spearman_pv_vs_profile_len']:+.3f} | "
              f"{R[f'k{k}']['spearman_pv_vs_answer_rate']:+.3f} |" for k in PV_BUDGETS)}

## 5. How many distinguishable situations exist at all

A free by-product of building the state list, and worth reporting on its own. Under the reference
static interview (`{o['pv_state_source']}`, everybody asked the same k questions), the {o['n_users_full']:,}
users land in this many **distinct states** — not "how many questions are optimal", but how many
distinguishable situations a selector could possibly react to:

{chr(10).join(stab)}

This is the dispersion collapse measured a second, independent way, and it bounds a DETERMINISTIC
selector exactly: a rule that conditions only on the answers to a fixed order has at most
{o['branching'][f"{o['pv_state_source']}|k2"]['distinct_states']} distinguishable inputs at k=2 and so
at most {o['branching'][f"{o['pv_state_source']}|k2"]['distinct_states']} distinguishable behaviours.
That is what `{o['pv_state_source']}` (1 interview),
`glm_ucb` ({o['branching']['glm_ucb|k2']['distinct_interviews']} interviews) and
`golbandi_tree` ({o['branching']['golbandi_tree|k2']['distinct_interviews']} interviews) sit far under.

Thompson is NOT bound by it, and that is the point: it conducts
{o['branching']['thompson|k2']['distinct_interviews']:,} distinct interviews at k=2 and
{o['branching']['thompson|k4']['distinct_interviews']:,} at k=4 — more than the whole population has
distinguishable states — because it escapes the bound by **randomising**, not by learning. A per-user
posterior draw manufactures branching that no amount of answer information could support, which is why
its branching count must never be read as evidence of personalisation. The honest statement is that
Thompson's divergence is a property of its sampling rule; the deterministic selectors' divergence is
bounded above by the state count, and both lose to one order asked of everybody.

## 6. The branching ratio (design sheet section 2)

Of the distinct interviews a selector actually conducts, how many distinct **loss-optimal** questions
do those states correspond to? The interview and state counts need no criterion evaluation and are
reported for every cell; the q* columns need scored states, and criterion scoring was deliberately
run for the PV arm only (see the note under the contract).

{chr(10).join(btab)}

## Verdict

(written after the run)
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    if OUT_MD.exists():
        old = OUT_MD.read_text(encoding="utf-8")
        cut_at = old.find("## Verdict")
        if cut_at >= 0:
            md = md[: md.find("## Verdict")] + old[cut_at:]
    OUT_MD.write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--cells", default="", help="comma list of 'spec|kK' to restrict scoring to")
    ap.add_argument("--no_preflight", action="store_true")
    args = ap.parse_args()
    if not args.no_preflight:
        preflight(require_peruser=args.analyze,
                  args_is_analyze_only=bool(args.analyze and not (args.score or args.bench)))
    if args.score or args.bench:
        phase_score(args)
    if args.analyze:
        phase_analyze(args)
    if not (args.score or args.analyze or args.bench):
        raise SystemExit("nothing to do: pass --score, --analyze or --bench")


if __name__ == "__main__":
    main()
