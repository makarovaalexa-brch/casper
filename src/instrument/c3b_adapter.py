r"""PAPER C / STAGE C3B -- variant (b): DOES THE VECTOR CONDITION THE TEXT?

Design sheet: `docs/design/DS_C3B_ADAPTER.md` (PRE-REGISTERED 2026-08-24, BINDING). Read it first.
Empirical motivation: `docs/results/C3_POC.md` (why variant (a) cannot carry the control).

THE QUESTION
------------
Train a small adapter so a FROZEN small LM writes a bank question's phrase from that question's
embedding. On HELD-OUT bank questions, does the generated text actually depend on the vector -- i.e.
does the true adapter A beat an identically-trained scrambled-correspondence twin A'? Fluency is
explicitly NOT evaluated (clumsy-but-faithful is the acceptable regime).

WHAT IS IMPORTED, NEVER REIMPLEMENTED
-------------------------------------
  * the space          `c0_geometry.bank_vectors` on the sha-asserted C0/C1 checkpoint (the same
                       rebuild `c3_poc_extract.py` / `c3_poc_reencode.py` perform);
  * the phrases        `c3_poc_extract.surface_forms` / `.phrase` (titles / genome tag names);
  * the re-encode      `c3_poc_reencode`'s frozen SBERT->concept->k(phi) bridge, imported as a module
                       (`sbert_sims`, `top_concepts`, `bridge`, `cos`, `tag_name_column`) so Stage 2
                       measures in the same space and with the same recipe of record;
  * the split          `outcome_q.stratify` / `.split_questions` -- experiment Q's construction, seed
                       98765, 20% held out BY QUESTION IDENTITY inside each channel x popularity
                       decile stratum (20 strata).

★ THE 2026-08-24 AMENDMENT (BINDING over the sheet's original Stage 2)
----------------------------------------------------------------------
The hostile-review pass found that `k(phi)` is E1's OWN head projection (`c0_geometry.py:133-138`),
so a gate scored there compares two images of one map. Five consequences, all pre-registered:
  1. the PRIMARY gate moves to the TOWER space -- text -> SBERT -> ConceptFoldNet(cv3) -> tower
     latent, against the question's own tower representation (item = decoder row `Wd[q]`,
     centred + popularity-stripped; concept = its `build_concept_dirs` direction). `k(phi)` is
     DEMOTED to a secondary row labelled *"in the emitter's own head space"*;
  2. criteria 1 and 3 are met PER CHANNEL and the ITEM channel is the gate (for a concept question
     the chain collapses to a string-identity lookup), with exact-phrase recovery printed beside
     every concept-channel number;
  3. A''s permutation is CHANNEL-PRESERVING (declared in the manifest) and the prior-only control is
     CHANNEL-MATCHED (zeroed latent + channel one-hot), so criterion 2 tests correspondence rather
     than channel membership;
  4. disclosed, not gated: the correlation between an item's score and its genome-membership count;
  5. the gate is stated RELATIVE TO A MEASURED CEILING (`ceiling`, run BEFORE training, no LM): the
     question's own TRUE phrase through the same bridge. Criterion 1 becomes "A - A' CI excludes
     zero AND A reaches >= 40% of that ceiling on the gate channel"; the absolute +0.10 is RETIRED.
     If the item ceiling is near zero the gate falls back to the concept channel and the item-channel
     weakness is reported as an INSTRUMENT limit, not a model failure. Every Stage-2 statistic
     carries a bootstrap CI.

SUBCOMMANDS
-----------
  prepare    build the 19,390 (k(phi_q), phrase) pairs, the split, the channel-preserving
             derangement, and a contract-signed manifest (checkpoint sha256, git rev, split
             hashes). No LM.
  ceiling    the instrument ceiling of AMENDMENT clause 5, per channel, with CIs. No LM. Run BEFORE
             training: it nominates the gate channel and sets criterion 1's threshold.
  train      --arm true|scrambled: adapter = 2-layer MLP 128 -> 8 soft prefix tokens x d_lm into a
             FROZEN distilgpt2; next-token CE on the phrase tokens; AdamW lr 1e-3 halved on val
             plateau, batch 32, <=10 epochs, best epoch by val CE on a 10% split of the TRAIN
             portion. Seed 98765. Per-epoch checkpoint, incremental jsonl; only the best survives
             (HARD RULE 9). `--smoke_steps N` runs N optimiser steps, prints s/step and exits.
  earlyread  the registered epoch-2 early read: both arms' train CE side by side.
  stage2     greedy <=16 tokens from every held-out vector under A, A' and the channel-matched
             prior-only arm; re-encode through BOTH bridges; the three registered criteria per
             channel (paired percentile bootstrap 10,000 resamples, seed 98765; identification
             against a seeded WITHIN-CHANNEL random pairing).

THE INPUT VECTOR. Bias-free `k(phi_q)`, MEAN-REMOVED (the bank mean is subtracted) and divided by ONE
GLOBAL SCALAR (the mean centred norm). Mean removal is the sheet's own measurement convention -- the
Stage 2 cosine is mean-removed, and C1_PILOT.md measured the raw bank-wide mean pairwise cosine at
0.9646 vs 0.0018 centred, so a raw-space input would hand the adapter a shared common mode as most of
its signal. The scalar is a single number applied to every row, so no direction and no relative
magnitude is altered; both constants are banked in the manifest and reused verbatim by train/stage2.

FIREWALL. Validation-side objects only: the 10,000 canonical TEST users are not built, not loaded,
not touched. No LLM API. distilgpt2 is fetched ONCE into the local HF cache (author-authorised) and
every later run is HF_HUB_OFFLINE=1. No truncation anywhere (HARD RULE 1).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

_NT = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _NT = sys.argv[_i + 1]
_NT = str(min(int(_NT), 4))                      # 4 PHYSICAL cores on this box; never exceed
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import glob as _glob
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn

from policy_net import Policy, build_phi, log, SEED
from c0_geometry import PAPERC, E1_CKPT, N_BANK, sha256, git, bank_vectors
import c1_pilot as C1
from c3_poc_extract import surface_forms, phrase as bank_phrase

OUT_DIR = os.path.join(PAPERC, "c3b")
PAIRS_NPZ = os.path.join(OUT_DIR, "c3b_pairs.npz")
PHRASES_JSON = os.path.join(OUT_DIR, "c3b_phrases.json")
MANIFEST = os.path.join(OUT_DIR, "c3b_manifest.json")

LM_NAME = "distilgpt2"
N_PREFIX = 8                 # soft prefix tokens (design sheet)
D_HID = 512                  # the MLP's single hidden width
FRAC_HELDOUT = 0.20          # experiment Q's fraction, by question identity
FRAC_INNER_VAL = 0.10        # of the TRAIN portion, for best-epoch selection
MAX_NEW_TOKENS = 16          # Stage 2 greedy budget
N_BOOT = 10000               # registered bootstrap resamples
CEILING_FRAC = 0.40          # AMENDMENT clause 5: A must reach >=40% of the true-phrase ceiling
ITEM_CEILING_FLOOR = 0.05    # below this the ITEM ceiling is "near zero" and the gate falls back
ID_FLOOR = 0.70              # criterion 3's threshold
# The absolute +0.10 floor of the original sheet is RETIRED by AMENDMENT clause 5.


# ============================================================================ small utilities
def sha_of(a):
    """A split hash: sha256 over the sorted int64 indices, ASCII, comma separated."""
    b = ",".join(str(int(x)) for x in np.sort(np.asarray(a, np.int64))).encode()
    return hashlib.sha256(b).hexdigest()


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def jsonl(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ============================================================================ the space + the pairs
def build_space(args):
    """The C0/C1 space, rebuilt exactly as c3_poc_extract/c3_poc_reencode rebuild it.

    Returns (E, K, names, ni, C, mu, ckpt_sha). `K` is the raw bias-free bank k(phi_q).
    """
    from strategy_channel_suite import build_greedy_env, genv_mean
    from ndcg_tree import build_pool

    blob = torch.load(args.ckpt, map_location="cpu")
    ck_sha = sha256(args.ckpt)
    assert ck_sha == C1.C0_CKPT_SHA, f"this is not the C0/C1-measured checkpoint (sha {ck_sha})"
    log(f"[ckpt] {os.path.basename(args.ckpt)} epoch={blob['epoch']} sha256 matches C0")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c3b")
    pool, _it, _cc = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not the ckpt's"

    ni = int(E["ctx"].ni)
    C = int(E["C"])
    full_pool = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    assert len(full_pool) == N_BANK, f"bank composition {ni}+{C} != {N_BANK}"
    phi_full, _ = build_phi(E, full_pool, norm=(blob["norm_mu"], blob["norm_sd"]))
    pol = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                 state=blob["state"], use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    _Eb, K = bank_vectors(pol, phi_full)
    names, ni2, missing = surface_forms(E)
    assert ni2 == ni
    log(f"[bank] k(phi) {K.shape}; surface forms built (missing titles "
        f"{missing['n_items_without_a_title']}, missing tag names "
        f"{missing['n_concepts_without_a_tag_name']})")
    return E, K, names, ni, C, missing, ck_sha, genv_mean


def input_vectors(K, mu, scale, chan):
    """The adapter's input: [ mean-removed k(phi) / one global scalar , CHANNEL ONE-HOT ].

    The latent block is direction-preserving (one scalar over every row). The two-column channel
    one-hot is required by the 2026-08-24 AMENDMENT clause 3: the prior-only control is "zeroed
    prefix + channel one-hot", which is only constructible if the channel enters the adapter
    separately from the latent -- otherwise zeroing the latent also blinds the control to the channel
    and criterion 2 would test channel membership instead of correspondence. Depth and width of the
    MLP, the 8 soft prefix tokens and everything else are unchanged; only the input width moves
    128 -> 130. Declared in the manifest as a deviation of record.
    """
    Z = ((K.astype(np.float64) - mu) / scale).astype(np.float32)
    oh = np.zeros((len(Z), 2), np.float32)
    oh[np.arange(len(Z)), np.asarray(chan, np.int64)] = 1.0
    return np.concatenate([Z, oh], 1)


def zero_latent_input(d_in, chan):
    """The AMENDMENT's channel-matched prior-only input: a ZEROED latent block + the channel one-hot."""
    x = np.zeros((len(chan), d_in), np.float32)
    x[np.arange(len(chan)), d_in - 2 + np.asarray(chan, np.int64)] = 1.0
    return x


def within_channel_derangement(idx, chan, seed):
    """A fixed permutation of phrases over vectors with NO fixed point, inside each channel."""
    rng = np.random.default_rng(seed)
    out = np.array(idx, np.int64).copy()
    for ch in (0, 1):
        m = np.nonzero(chan[idx] == ch)[0]
        if len(m) < 2:
            assert len(m) == 0, "a channel with a single member cannot be deranged"
            continue
        src = np.asarray(idx, np.int64)[m]
        perm = rng.permutation(len(src))
        # repair fixed points by rotating each one against its neighbour
        for _ in range(64):
            fx = np.nonzero(perm == np.arange(len(src)))[0]
            if not len(fx):
                break
            for f in fx:
                g = (f + 1) % len(src)
                perm[[f, g]] = perm[[g, f]]
        assert not (perm == np.arange(len(src))).any(), "derangement still has fixed points"
        out[m] = src[perm]
    assert not (out == np.asarray(idx, np.int64)).any(), "the derangement fixes a pair"
    return out


# ============================================================================ prepare
def cmd_prepare(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    from outcome_q import stratify, split_questions
    from strategy_channel_suite import CONCSTATS, N_TRAIN_USERS

    E, K, names, ni, C, missing, ck_sha, genv_mean = build_space(args)

    # HARD RULE 5 canonical snap: the env is the certified one (built only to read the descriptors)
    from c0_geometry import VAL_M0_FULL
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} tail={t0_!r} -- canonical")

    # ---- the phrases -----------------------------------------------------------------------------
    phrases = [bank_phrase(names, ni, i) for i in range(N_BANK)]
    unnamed = [i for i, p in enumerate(phrases) if p.startswith("<unnamed bank entry ")]
    assert len(phrases) == N_BANK

    # ---- the split: experiment Q's construction, imported ----------------------------------------
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    ans_rate = (np.load(CONCSTATS)["n"] / N_TRAIN_USERS).astype(np.float64)
    st = stratify(ni, C, cnt, ans_rate)
    train_q, held_q = split_questions(st, FRAC_HELDOUT, SEED)
    assert len(np.intersect1d(train_q, held_q)) == 0
    assert len(train_q) + len(held_q) == N_BANK

    # the inner val split of the TRAIN portion, same stratified construction, offset seed
    rng = np.random.default_rng(SEED + 11)
    inner_val = []
    for s in np.unique(st[train_q]):
        idx = train_q[st[train_q] == s].copy()
        rng.shuffle(idx)
        inner_val.append(idx[:int(round(len(idx) * FRAC_INNER_VAL))])
    inner_val = np.sort(np.concatenate(inner_val))
    inner_fit = np.sort(np.setdiff1d(train_q, inner_val))

    # ---- the twin's correspondence scramble, within channel, over the TRAIN portion only ----------
    chan = np.concatenate([np.zeros(ni, np.int64), np.ones(C, np.int64)])
    scram_src = within_channel_derangement(train_q, chan, SEED + 23)   # train_q[j] gets scram_src[j]
    scramble = np.full(N_BANK, -1, np.int64)
    scramble[train_q] = scram_src
    assert (chan[train_q] == chan[scram_src]).all(), "the scramble crossed a channel"

    # ---- the adapter input space -----------------------------------------------------------------
    mu = K.astype(np.float64).mean(0)
    Kc = K.astype(np.float64) - mu
    scale = float(np.linalg.norm(Kc, axis=1).mean())
    X = input_vectors(K, mu, scale, chan)
    log(f"[input] mean removed (bank mean norm {np.linalg.norm(mu):.4f}); global scale {scale:.4f}; "
        f"X {X.shape} (128 latent + 2 channel one-hot) mean |x| {np.abs(X[:, :-2]).mean():.4f}")

    np.savez_compressed(PAIRS_NPZ, X=X, K=K.astype(np.float32),
                        mu=mu.astype(np.float64), scale=np.float64(scale),
                        chan=chan, strat=st, train_q=train_q, held_q=held_q,
                        inner_val=inner_val, inner_fit=inner_fit, scramble=scramble)
    with open(PHRASES_JSON, "w", encoding="utf-8") as fh:
        json.dump({"_comment": "bank index -> surface form (c3_poc_extract.surface_forms/.phrase)",
                   "n": N_BANK, "n_items": ni, "n_concepts": C, "phrases": phrases}, fh)

    # ---- the stratification balance table ---------------------------------------------------------
    strata = [int(s) for s in np.unique(st)]
    bal = []
    for s in strata:
        a = int((st == s).sum()); h = int((st[held_q] == s).sum())
        bal.append({"stratum": s, "channel": "item" if s < 10 else "concept", "decile": s % 10,
                    "n_total": a, "n_train": a - h, "n_heldout": h,
                    "heldout_share": round(h / max(a, 1), 4)})
    plen = np.asarray([len(p) for p in phrases])
    comp = {
        "n_pairs": N_BANK, "n_items": ni, "n_concepts": C,
        "n_train_questions": int(len(train_q)), "n_heldout_questions": int(len(held_q)),
        "n_inner_fit": int(len(inner_fit)), "n_inner_val": int(len(inner_val)),
        "heldout_item_share": float((held_q < ni).mean()),
        "train_item_share": float((train_q < ni).mean()),
        "heldout_n_items": int((held_q < ni).sum()),
        "heldout_n_concepts": int((held_q >= ni).sum()),
        "n_strata": len(strata),
        "n_unnamed_phrases": len(unnamed),
        "unnamed_bank_indices": unnamed[:50],
        "phrase_chars": {"min": int(plen.min()), "median": float(np.median(plen)),
                         "mean": float(plen.mean()), "max": int(plen.max())},
        "n_duplicate_phrases": int(N_BANK - len(set(phrases))),
        "balance_by_stratum": bal}

    man = {
        "analysis": "paperC_C3B_adapter_prepare",
        "design_sheet": "docs/design/DS_C3B_ADAPTER.md",
        "produced_by": "src/instrument/c3b_adapter.py prepare",
        "citable": True, "test_touched": False, "split": "validation",
        "seed": SEED,
        "space": {
            "definition": "bias-free k(phi_q) via c0_geometry.bank_vectors (IMPORTED)",
            "d_k": int(K.shape[1]), "d_adapter_input": int(X.shape[1]),
            "adapter_input": ("[ MEAN-REMOVED k(phi_q) / one global scalar , CHANNEL ONE-HOT (2) ]"),
            "channel_one_hot": ("DEVIATION OF RECORD, required by AMENDMENT clause 3: the "
                                "channel-matched prior-only control ('zeroed prefix + channel "
                                "one-hot') is only constructible if the channel reaches the adapter "
                                "separately from the latent. MLP depth/width and the 8 soft prefix "
                                "tokens are unchanged; the input width moves 128 -> 130."),
            "bank_mean_norm": float(np.linalg.norm(mu)),
            "global_scale_mean_centred_norm": scale},
        "phrases": {"source": "c3_poc_extract.surface_forms / .phrase (IMPORTED)",
                    "items": "ML-25M title", "concepts": "genome tag name, lower-cased",
                    "surface_form_coverage": missing},
        "split": {
            "construction": ("outcome_q.stratify + outcome_q.split_questions, IMPORTED -- 20% of "
                             "the 19,390 questions held out BY IDENTITY, stratified by channel x "
                             "popularity decile (20 strata), seed 98765; popularity = train rating "
                             "count for an item, train answer rate for a concept"),
            "frac_heldout": FRAC_HELDOUT, "frac_inner_val_of_train": FRAC_INNER_VAL,
            "inner_val": "same stratified construction over the TRAIN portion, seed 98765+11",
            "hashes": {"train_q": sha_of(train_q), "heldout_q": sha_of(held_q),
                       "inner_fit": sha_of(inner_fit), "inner_val": sha_of(inner_val),
                       "scramble_train_targets": sha_of(scram_src),
                       "strata": hashlib.sha256(st.tobytes()).hexdigest(),
                       "X": hashlib.sha256(X.tobytes()).hexdigest()}},
        "scramble": {"what": ("the twin's correspondence control: a fixed derangement of phrases "
                              "over vectors across the TRAIN portion (so no held-out phrase can "
                              "enter training), seed 98765+23, no fixed points"),
                     "channel_preserving": True,
                     "declared": ("AMENDMENT clause 3: the permutation is CHANNEL-PRESERVING -- an "
                                  "item vector always receives an item phrase and a concept vector "
                                  "a concept phrase -- so criterion 2 tests correspondence, not "
                                  "channel membership. Asserted at build time."),
                     "n_scrambled": int(len(train_q))},
        "composition": comp,
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": ck_sha},
        "canonical_snap": {"val_m0_full": f0, "val_m0_tail": t0_},
        "git": {"rev": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "status_short": git("status", "--short")},
        "outputs": {"pairs": os.path.relpath(PAIRS_NPZ, _ROOT),
                    "phrases": os.path.relpath(PHRASES_JSON, _ROOT)},
        "minutes": round((time.time() - t0) / 60, 2)}
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=float)

    log(f"[split] pairs {N_BANK:,} | train {len(train_q):,} (fit {len(inner_fit):,} / inner-val "
        f"{len(inner_val):,}) | held-out {len(held_q):,}")
    log(f"[split] held-out items {comp['heldout_n_items']:,} concepts "
        f"{comp['heldout_n_concepts']:,}; item share held-out {comp['heldout_item_share']:.4f} vs "
        f"train {comp['train_item_share']:.4f}")
    log("stratum | channel | dec |  total |  train | heldout | share")
    for b in bal:
        log(f"{b['stratum']:>7} | {b['channel']:>7} | {b['decile']:>3} | {b['n_total']:>6} | "
            f"{b['n_train']:>6} | {b['n_heldout']:>7} | {b['heldout_share']:.4f}")
    rr = np.random.default_rng(SEED)
    for i in rr.choice(held_q, size=5, replace=False):
        log(f"[spot-check] bank {int(i):>6} ({'item' if i < ni else 'concept'}, stratum "
            f"{int(st[i])}) -> {phrases[int(i)]!r}")
    log(f"[done] wrote {MANIFEST} ({man['minutes']:.1f}m)")
    return man


# ============================================================================ the LM + the adapter
def ensure_lm():
    """distilgpt2 from the local HF cache; download ONCE if absent (author-authorised), then pin
    HF_HUB_OFFLINE=1 so nothing else can reach the network."""
    from huggingface_hub import snapshot_download
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache",
                                                     "huggingface")
    root = os.path.join(home, "hub", "models--" + LM_NAME.replace("/", "--"), "snapshots")
    # A DIRECTORY IS NOT A CACHE. A failed download leaves the folder behind; checking `isdir` and
    # then pinning HF_HUB_OFFLINE=1 would hand `from_pretrained` an offline flag and an empty cache.
    cached = any(os.path.exists(os.path.join(d, "config.json"))
                 and (os.path.exists(os.path.join(d, "model.safetensors"))
                      or os.path.exists(os.path.join(d, "pytorch_model.bin")))
                 for d in _glob.glob(os.path.join(root, "*")))
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        log(f"[lm] {LM_NAME} present in the local HF cache; HF_HUB_OFFLINE=1")
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        log(f"[lm] {LM_NAME} NOT cached -- performing the ONE-TIME authorised download")
        # safetensors ONLY: the repo also carries a duplicate .bin plus TF/Flax/Rust mirrors, and
        # this box has repeatedly run out of disk. One weight file, the tokenizer, the config.
        snapshot_download(LM_NAME, allow_patterns=["*.json", "*.txt", "model.safetensors"])
        os.environ["HF_HUB_OFFLINE"] = "1"
        log(f"[lm] downloaded; HF_HUB_OFFLINE=1 from here on")


def load_lm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LM_NAME)
    lm = AutoModelForCausalLM.from_pretrained(LM_NAME, torch_dtype=torch.float32)
    lm.eval()
    lm.requires_grad_(False)                                    # FROZEN. Only the adapter trains.
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    assert n_train == 0, f"the LM is not frozen ({n_train} trainable parameters)"
    return tok, lm


class Adapter(nn.Module):
    """2-layer MLP: d_in -> D_HID -> N_PREFIX * d_lm, reshaped to soft prefix tokens."""

    def __init__(self, d_in, d_lm, n_prefix=N_PREFIX, d_hid=D_HID):
        super().__init__()
        self.n_prefix, self.d_lm = n_prefix, d_lm
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(),
                                 nn.Linear(d_hid, n_prefix * d_lm))

    def forward(self, x):
        return self.net(x).view(x.shape[0], self.n_prefix, self.d_lm)


def encode_phrases(tok, phrases):
    """Phrase -> token ids + EOS. NO TRUNCATION (HARD RULE 1)."""
    eos = tok.eos_token_id
    return [tok(p, add_special_tokens=False)["input_ids"] + [eos] for p in phrases]


def batch_tensors(lm, prefix, ids_list):
    """(inputs_embeds, attention_mask, labels) for a batch, right-padded, prefix labels = -100."""
    B, P = prefix.shape[0], prefix.shape[1]
    L = max(len(t) for t in ids_list)
    ids = torch.zeros(B, L, dtype=torch.long)
    lab = torch.full((B, P + L), -100, dtype=torch.long)
    am = torch.zeros(B, P + L, dtype=torch.long)
    am[:, :P] = 1
    for i, t in enumerate(ids_list):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        lab[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)
        am[i, P:P + len(t)] = 1
    emb = lm.get_input_embeddings()(ids)
    return torch.cat([prefix, emb], 1), am, lab


def ce_on(lm, adapter, X, ids_all, idx, tgt, batch, train=False, opt=None, max_steps=None,
          log_every=0, tag=""):
    """Mean per-TOKEN next-token CE over `idx` (vectors) paired with `tgt` (phrase owners)."""
    tot = ntok = 0.0
    t0 = time.time()
    steps = int(np.ceil(len(idx) / batch)) if max_steps is None else int(max_steps)
    for si in range(steps):
        sl = slice(si * batch, min((si + 1) * batch, len(idx)))
        if sl.start >= len(idx):
            break
        xb = torch.from_numpy(X[idx[sl]])
        ids_b = [ids_all[int(t)] for t in tgt[sl]]
        with torch.set_grad_enabled(train):
            pre = adapter(xb)
            emb, am, lab = batch_tensors(lm, pre, ids_b)
            out = lm(inputs_embeds=emb, attention_mask=am, labels=lab)
            n = int((lab[:, 1:] != -100).sum())
            if train:
                opt.zero_grad(); out.loss.backward(); opt.step()
        tot += float(out.loss) * n; ntok += n
        if log_every and (si + 1) % log_every == 0:
            el = time.time() - t0
            log(f"[{tag} {si + 1}/{steps}] ce {tot / ntok:.5f} ({el / 60:.1f}m, "
                f"{el / (si + 1):.3f}s/step, projected {el / (si + 1) * steps / 60:.1f}m/epoch)")
    return tot / max(ntok, 1.0), ntok, steps


def cmd_train(args):
    assert args.arm in ("true", "scrambled")
    man = load_manifest()
    z = np.load(PAIRS_NPZ)
    X = z["X"]
    with open(PHRASES_JSON, encoding="utf-8") as fh:
        phrases = json.load(fh)["phrases"]
    inner_fit, inner_val = z["inner_fit"], z["inner_val"]
    scramble = z["scramble"]

    # the phrase each training vector is asked to produce
    def targets(idx):
        return idx if args.arm == "true" else scramble[idx]
    tgt_fit, tgt_val = targets(inner_fit), targets(inner_val)
    assert (tgt_fit >= 0).all() and (tgt_val >= 0).all()
    if args.arm == "scrambled":
        assert not (tgt_fit == inner_fit).any(), "the scrambled arm kept a true pair"

    ensure_lm()
    tok, lm = load_lm()
    d_lm = int(lm.config.n_embd)
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(SEED); np.random.seed(SEED)
    adapter = Adapter(X.shape[1], d_lm)
    n_par = sum(p.numel() for p in adapter.parameters())
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    ids_all = encode_phrases(tok, phrases)
    log(f"[build] adapter {X.shape[1]} -> {D_HID} -> {N_PREFIX}x{d_lm} ({n_par:,} params); LM "
        f"{LM_NAME} FROZEN; fit {len(inner_fit):,} / inner-val {len(inner_val):,}; arm {args.arm}")

    if args.smoke_steps:
        rr = np.random.default_rng(SEED)
        pe = inner_fit[rr.permutation(len(inner_fit))]
        t0 = time.time()
        ce, ntok, steps = ce_on(lm, adapter, X, ids_all, pe, targets(pe), args.batch, train=True,
                                opt=opt, max_steps=args.smoke_steps)
        el = time.time() - t0
        per_ep = int(np.ceil(len(inner_fit) / args.batch))
        log(f"[SMOKE] {steps} optimiser steps in {el:.1f}s = {el / steps:.3f}s/step; train CE "
            f"{ce:.5f} over {int(ntok):,} tokens")
        log(f"[SMOKE] {per_ep} steps/epoch => ~{el / steps * per_ep / 60:.1f} min/epoch/arm "
            f"(+ inner-val pass); NOTHING WAS SAVED -- this is a timing probe, not training")
        return {"s_per_step": el / steps, "steps_per_epoch": per_ep}

    stem = f"c3b_{args.arm}"
    logp = os.path.join(OUT_DIR, stem + ".jsonl")
    jsonl(logp, {"event": "start", "arm": args.arm, "lr": args.lr, "batch": args.batch,
                 "epochs": args.epochs, "n_params": n_par, "d_lm": d_lm,
                 "n_fit": int(len(inner_fit)), "n_val": int(len(inner_val)),
                 "seed": SEED, "manifest_git": man["git"]["rev"],
                 "split_hashes": man["split"]["hashes"], "time": time.time()})
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0)
    rr = np.random.default_rng(SEED + 3)
    hist, best = [], None
    t00 = time.time()
    for ep in range(1, args.epochs + 1):
        adapter.train()
        pe = inner_fit[rr.permutation(len(inner_fit))]
        te = time.time()
        tr_ce, _n, steps = ce_on(lm, adapter, X, ids_all, pe, targets(pe), args.batch, train=True,
                                 opt=opt, log_every=args.log_every, tag=f"{stem} ep{ep}")
        adapter.eval()
        va_ce, _nv, _s = ce_on(lm, adapter, X, ids_all, inner_val, tgt_val, args.batch)
        sched.step(va_ce)
        row = {"event": "epoch", "arm": args.arm, "epoch": ep, "train_ce": tr_ce, "val_ce": va_ce,
               "lr": float(opt.param_groups[0]["lr"]), "steps": steps,
               "minutes": round((time.time() - te) / 60, 2), "time": time.time()}
        hist.append(row); jsonl(logp, row)
        torch.save({"adapter": adapter.state_dict(), "arm": args.arm, "epoch": ep,
                    "train_ce": tr_ce, "val_ce": va_ce, "d_in": int(X.shape[1]), "d_lm": d_lm,
                    "n_prefix": N_PREFIX, "d_hid": D_HID, "lm": LM_NAME, "seed": SEED,
                    "split_hashes": man["split"]["hashes"]},
                   os.path.join(OUT_DIR, f"{stem}_ep{ep:02d}.pt"))
        log(f"[{stem} ep{ep}] train CE {tr_ce:.5f} | val CE {va_ce:.5f} | lr "
            f"{row['lr']:.2e} ({row['minutes']:.1f}m)")
        if best is None or va_ce < best["val_ce"]:
            best = row
        # Bound the live checkpoints to two (the running best + the epoch just written). The per-epoch
        # checkpoint exists so a crash costs one epoch, not so ten epochs accumulate; this box has run
        # out of disk mid-download once already today.
        for q in _glob.glob(os.path.join(OUT_DIR, f"{stem}_ep*.pt")):
            e = int(os.path.basename(q).rsplit("_ep", 1)[1].split(".")[0])
            if e not in (ep, best["epoch"]):
                os.remove(q)
    # HARD RULE 9: keep the best epoch only
    keep = os.path.join(OUT_DIR, f"{stem}_ep{best['epoch']:02d}.pt")
    final = os.path.join(OUT_DIR, f"{stem}_best.pt")
    for p in _glob.glob(os.path.join(OUT_DIR, f"{stem}_ep*.pt")):
        if os.path.abspath(p) != os.path.abspath(keep):
            os.remove(p)
    os.replace(keep, final)
    done = {"event": "done", "arm": args.arm, "best_epoch": best["epoch"],
            "best_val_ce": best["val_ce"], "best_train_ce": best["train_ce"],
            "kept": os.path.relpath(final, _ROOT), "history": hist,
            "minutes": round((time.time() - t00) / 60, 2), "time": time.time()}
    jsonl(logp, done)
    log(f"[{stem}] BEST epoch {best['epoch']} val CE {best['val_ce']:.5f}; kept {final} "
        f"(all other epoch checkpoints deleted, HARD RULE 9)")
    return done


# ============================================================================ the epoch-2 early read
def cmd_earlyread(args):
    rows = {a: [r for r in read_jsonl(os.path.join(OUT_DIR, f"c3b_{a}.jsonl"))
                if r.get("event") == "epoch"] for a in ("true", "scrambled")}
    log("arm       | " + " | ".join(f"ep{e}" for e in range(1, 1 + max(
        [len(v) for v in rows.values()] + [2]))))
    for a in ("true", "scrambled"):
        log(f"{a:>9} | " + " | ".join(f"{r['train_ce']:.5f}" for r in rows[a]))
    got = {a: next((r for r in rows[a] if r["epoch"] == args.epoch), None) for a in rows}
    if not all(got.values()):
        log(f"[earlyread] epoch {args.epoch} is not banked for both arms yet "
            f"({ {a: (r or {}).get('epoch') for a, r in got.items()} }) -- nothing to read")
        return None
    d_tr = got["scrambled"]["train_ce"] - got["true"]["train_ce"]
    d_va = got["scrambled"]["val_ce"] - got["true"]["val_ce"]
    out = {"analysis": "paperC_C3B_earlyread", "epoch": args.epoch,
           "true_train_ce": got["true"]["train_ce"],
           "scrambled_train_ce": got["scrambled"]["train_ce"],
           "delta_train_ce_scrambled_minus_true": d_tr,
           "true_val_ce": got["true"]["val_ce"], "scrambled_val_ce": got["scrambled"]["val_ce"],
           "delta_val_ce_scrambled_minus_true": d_va,
           "separated": bool(d_tr > 0.0),
           "registered_flag": ("DS_C3B_ADAPTER.md secondary: if A and A' train-CE curves have NOT "
                               "separated by epoch 2, FLAG IMMEDIATELY")}
    log("=" * 100)
    log(f"[EARLY READ ep{args.epoch}] train CE  true {got['true']['train_ce']:.5f}  vs  scrambled "
        f"{got['scrambled']['train_ce']:.5f}   (scrambled - true = {d_tr:+.5f})")
    log(f"[EARLY READ ep{args.epoch}]   val CE  true {got['true']['val_ce']:.5f}  vs  scrambled "
        f"{got['scrambled']['val_ce']:.5f}   (scrambled - true = {d_va:+.5f})")
    log("[EARLY READ] " + ("the curves HAVE separated in the expected direction"
                           if d_tr > 0 else "*** NOT SEPARATED -- FLAG THIS TO THE AUTHOR NOW ***"))
    log("=" * 100)
    with open(os.path.join(OUT_DIR, "c3b_earlyread.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


# ============================================================================ stage 2
def load_adapter(path, d_in):
    blob = torch.load(path, map_location="cpu")
    a = Adapter(blob["d_in"], blob["d_lm"], blob["n_prefix"], blob["d_hid"])
    a.load_state_dict(blob["adapter"]); a.eval()
    assert blob["d_in"] == d_in, f"adapter input dim {blob['d_in']} != pairs X dim {d_in}"
    return a, blob


@torch.no_grad()
def greedy(lm, tok, prefix, max_new=MAX_NEW_TOKENS):
    """Greedy continuation from a soft prefix. No cache (short sequences); deterministic."""
    B, P, _ = prefix.shape
    emb = prefix
    am = torch.ones(B, P, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    outs = [[] for _ in range(B)]
    emb_tab = lm.get_input_embeddings()
    for _ in range(max_new):
        logits = lm(inputs_embeds=emb, attention_mask=am).logits[:, -1, :]
        nxt = logits.argmax(-1)
        for i in range(B):
            if not done[i]:
                if int(nxt[i]) == tok.eos_token_id:
                    done[i] = True
                else:
                    outs[i].append(int(nxt[i]))
        if bool(done.all()):
            break
        emb = torch.cat([emb, emb_tab(nxt)[:, None, :]], 1)
        am = torch.cat([am, torch.ones(B, 1, dtype=torch.long)], 1)
    return [tok.decode(o).strip() for o in outs]


def norm_txt(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def tok_f1(a, b):
    A, B = norm_txt(a).split(), norm_txt(b).split()
    if not A or not B:
        return 0.0
    inter = 0
    bb = list(B)
    for w in A:
        if w in bb:
            bb.remove(w); inter += 1
    if not inter:
        return 0.0
    p, r = inter / len(A), inter / len(B)
    return 2 * p * r / (p + r)


def boot_mean(d, n=N_BOOT, seed=SEED):
    """Paired percentile bootstrap over the held-out QUESTIONS (the pairing is preserved by
    resampling questions, never arms). AMENDMENT clause 5: every Stage-2 statistic carries a CI."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_boot": int(n),
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(d)), "n_boot": int(n), "excludes_zero": bool(lo > 0 or hi < 0)}


def boot_corr(x, y, n=N_BOOT, seed=SEED):
    """Pearson + Spearman with paired bootstrap CIs (the disclosure row of AMENDMENT clause 4)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return {"n": int(len(x)), "pearson": None, "spearman": None}

    def rk(v):
        o = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)
        return o

    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / den) if den > 1e-30 else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    bp = np.asarray([pear(x[i], y[i]) for i in idx])
    bs = np.asarray([pear(rk(x[i]), rk(y[i])) for i in idx])
    return {"n": int(len(x)),
            "pearson": {"r": pear(x, y), "ci_lo": float(np.percentile(bp, 2.5)),
                        "ci_hi": float(np.percentile(bp, 97.5))},
            "spearman": {"r": pear(rk(x), rk(y)), "ci_lo": float(np.percentile(bs, 2.5)),
                         "ci_hi": float(np.percentile(bs, 97.5))}}


def derange(n, seed):
    """A seeded permutation with no fixed point (the random-pairing partner of criterion 3)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for _ in range(64):
        fx = np.nonzero(p == np.arange(n))[0]
        if not len(fx):
            break
        for f in fx:
            g = (f + 1) % n
            p[[f, g]] = p[[g, f]]
    assert n < 2 or not (p == np.arange(n)).any()
    return p


# ==================================================================== THE AMENDED (TOWER) BRIDGE
class TowerBridge:
    r"""AMENDMENT clause 1 -- the PRIMARY measurement space.

        text --SBERT--> concepts --ConceptFoldNet(cv3)--> tower latent --> compare to the question's
        OWN tower representation.

    Every map in that chain (the frozen SBERT of record, the certified cv3 concept fold, the
    certified tower) was fitted before and independently of E1's head, so unlike the `k(phi)` bridge
    it is not two images of one map.

    THE TWO SIDES, and why the whitening is what it is
    --------------------------------------------------
      * TEXT side. `ConceptFoldNet` expects its running vector to be the tower's own no-evidence
        latent `z0` (the fold of an EMPTY item set) and returns `z0 + U`; handing it zeros puts it
        far outside its input distribution (documented at `run_battery_phaseA.shift_mat_trained`).
        So the text's tower vector is the DISPLACEMENT `u = fold(z0, selected concepts) - z0`.
      * QUESTION side. An ITEM's tower representation is its decoder row `Wd[q]`, centred and
        popularity-stripped -- i.e. `Ew[q]`, exactly the per-member quantity `build_concept_dirs`
        averages. A CONCEPT's is `sh["d_c"][c]`, that same average, taken from the env.
      * Both sides are therefore DISPLACEMENTS in the latent basis, and both are popularity-stripped
        (`u <- u - <u,u1>u1`). The bank mean is removed from the question side only, because `u` is
        already a difference and subtracting a point-mean from a difference is not the same
        operation. That `Ew` is the right object is not asserted here but CHECKED: averaging it over
        each concept's members must reproduce the env's own `d_c` to float tolerance.
    """

    def __init__(self, args, need_sbert=True):
        from strategy_channel_suite import build_greedy_env
        from concept_fold import fold_items_frozen
        import c3_poc_reencode as RE
        self.RE = RE

        blob = torch.load(args.ckpt, map_location="cpu")
        self.ckpt_sha = sha256(args.ckpt)
        assert self.ckpt_sha == C1.C0_CKPT_SHA, "this is not the C0/C1-measured checkpoint"
        E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c3b_bridge")
        self.E = E
        ctx, sh = E["ctx"], E["sh"]
        self.ni, self.C = int(ctx.ni), int(E["C"])
        self.names, _n, _m = surface_forms(E)
        self.tag_names = RE.tag_name_column(E)
        bad = [c for c in range(self.C) if self.names[self.ni + c] != self.tag_names[c].lower()]
        assert not bad, f"SBERT tag column != the concept channel at {len(bad)} positions"

        # ---- the tower-space targets -------------------------------------------------------------
        Wd = np.asarray(ctx.Wd.detach().numpy(), np.float64)
        u1 = np.asarray(sh["u1"].detach().numpy(), np.float64)
        u1 = u1 / np.linalg.norm(u1)
        Ec = Wd - Wd.mean(0, keepdims=True)
        self.Ew = Ec - np.outer(Ec @ u1, u1)                        # item targets, centred+stripped
        self.u1 = u1
        self.d_c = np.asarray(sh["d_c"].detach().numpy(), np.float64)
        # CONTROL: the item targets must average, over each concept's members, back to the env's d_c
        tags, members = list(sh["tags"]), sh["members"]
        worst = 0.0
        for i, t in enumerate(tags):
            v = self.Ew[np.asarray(members[t], np.int64)].mean(0)
            v = v / max(np.linalg.norm(v), 1e-12)
            worst = max(worst, float(np.abs(np.abs(v @ self.d_c[i]) - 1.0)))
        assert worst < 1e-3, (f"the reconstructed whitening does not reproduce the env's concept "
                              f"directions (worst 1-|cos| {worst:g})")
        self.dc_reproduction_worst = worst
        log(f"[tower] item targets Ew {self.Ew.shape}; concept directions reproduced from them "
            f"(worst 1-|cos| {worst:g})")
        self.member_count = np.zeros(self.ni, np.int64)              # genome-membership disclosure
        for t in tags:
            self.member_count[np.asarray(members[t], np.int64)] += 1

        # ---- the fold ----------------------------------------------------------------------------
        self.net = E["net"]
        self.net.eval()
        with torch.no_grad():
            self.z0 = fold_items_frozen(ctx.enc,
                                        [(np.empty(0, np.int64), np.empty(0, np.int64))])
        self.d = int(self.z0.shape[1])
        log(f"[tower] cold base z0 (empty-item tower fold) norm {float(self.z0.norm()):.4f}, d={self.d}")

        # ---- the SECONDARY space: the emitter's own head ------------------------------------------
        z = np.load(PAIRS_NPZ)
        self.Kc = z["K"].astype(np.float64) - z["mu"]
        self.sb = None
        if need_sbert:
            from sentence_transformers import SentenceTransformer
            self.sb = SentenceTransformer(RE.SBERT_MODEL)
            log(f"[sbert] {RE.SBERT_MODEL} loaded "
                f"(HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')})")

    # ------------------------------------------------------------------ text -> the two spaces
    def select(self, texts):
        sims = self.RE.sbert_sims(list(texts), self.tag_names, self.sb)
        return self.RE.top_concepts(sims)

    def tower_vectors(self, sel, weighted=False, batch=256):
        """selected [(cid, cos)] per text -> the popularity-stripped fold displacement u."""
        U = np.full((len(sel), self.d), np.nan)
        live = [j for j, s in enumerate(sel) if s]
        for st in range(0, len(live), batch):
            js = live[st:st + batch]
            cid, val = [], []
            for j in js:
                c = [int(a) for a, _ in sel[j]]
                v = [float(b) if weighted else 1.0 for _, b in sel[j]]
                c, v = self.net.canonical_order(c, v)            # the documented fold order
                cid.append(c); val.append(v)
            refs = [[0.0] * len(c) for c in cid] if self.net.use_refusal else None
            with torch.no_grad():
                z = self.net(self.z0.expand(len(js), self.d).contiguous(), cid, val, refs)
            u = (z - self.z0).numpy().astype(np.float64)
            U[js] = u - np.outer(u @ self.u1, self.u1)
        return U

    def kphi_vectors(self, sel):
        """SECONDARY (the emitter's own head space): the PoC bridge, imported verbatim."""
        V = np.full((len(sel), self.Kc.shape[1]), np.nan)
        for j, s in enumerate(sel):
            u, _w = self.RE.bridge(s, self.Kc, self.ni)
            if u is not None:
                V[j] = u
        return V

    def target_tower(self, bank_idx):
        return np.stack([self.Ew[int(i)] if int(i) < self.ni else self.d_c[int(i) - self.ni]
                         for i in bank_idx])

    def target_kphi(self, bank_idx):
        return self.Kc[np.asarray(bank_idx, np.int64)]


def rowcos(A, B):
    """Row-wise cosine, NaN-safe."""
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    out = np.full(len(A), np.nan)
    ok = np.isfinite(na) & np.isfinite(nb) & (na > 1e-30) & (nb > 1e-30)
    out[ok] = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
    return out


# ============================================================================ the instrument ceiling
def cmd_ceiling(args):
    """AMENDMENT clause 5: what a PERFECT verbalisation earns. Pushes every held-out question's own
    TRUE surface phrase through the amended bridge and scores it against that question's own tower
    representation. No LM, no adapter, no training -- this is the number the gate is stated relative
    to, and it is measured BEFORE training so it cannot be chosen after seeing a result."""
    t00 = time.time()
    man = load_manifest()
    z = np.load(PAIRS_NPZ)
    held = z["held_q"]
    with open(PHRASES_JSON, encoding="utf-8") as fh:
        phrases = json.load(fh)["phrases"]
    br = TowerBridge(args)
    ni = br.ni
    gold = [phrases[int(i)] for i in held]
    is_item = held < ni

    sel = br.select(gold)
    U = br.tower_vectors(sel)
    Vk = br.kphi_vectors(sel)
    T = br.target_tower(held)
    cos_t = rowcos(U, T)
    cos_k = rowcos(Vk, br.target_kphi(held))
    nsel = np.asarray([len(s) for s in sel], np.int64)
    enc = nsel > 0

    # the identification ceiling too: is the TRUE phrase's encoding nearer its own target than a
    # random other same-channel target? (the ceiling for criterion 3)
    idcap = {}
    for nm, m in (("item", is_item), ("concept", ~is_item)):
        idx = np.nonzero(m & enc)[0]
        if len(idx) < 3:
            idcap[nm] = None; continue
        p = derange(len(idx), SEED)
        own = cos_t[idx]
        oth = rowcos(U[idx], T[idx][p])
        idcap[nm] = dict(boot_mean((own > oth).astype(np.float64)),
                         mean_cos_to_random_other=float(np.nanmean(oth)))

    per = {}
    for nm, m in (("item", is_item), ("concept", ~is_item), ("all", np.ones_like(is_item))):
        mm = m & enc
        per[nm] = {
            "n": int(m.sum()), "n_encodable": int(mm.sum()),
            "encodable_rate": float(mm.sum() / max(m.sum(), 1)),
            "mean_concepts_selected": float(nsel[m].mean()),
            "tower_cos": boot_mean(cos_t[mm]),
            "kphi_cos_SECONDARY_emitters_own_head_space": boot_mean(cos_k[mm]),
            "identification": idcap.get(nm)}
    it_ceil = per["item"]["tower_cos"]["mean"]
    cp_ceil = per["concept"]["tower_cos"]["mean"]
    gate_channel = "item" if (it_ceil is not None and it_ceil >= args.item_ceiling_floor) else "concept"

    out = {"analysis": "paperC_C3B_instrument_ceiling",
           "design_sheet": "docs/design/DS_C3B_ADAPTER.md AMENDMENT clause 5",
           "produced_by": "src/instrument/c3b_adapter.py ceiling",
           "citable": True, "test_touched": False, "split": "validation",
           "what": ("the held-out question's OWN true surface phrase pushed through the amended "
                    "tower-space bridge and scored against its own tower representation -- the "
                    "ceiling a perfect verbalisation could earn, per channel"),
           "primary_space": ("text -> SBERT -> concepts -> ConceptFoldNet(cv3) -> tower latent "
                             "displacement from the cold fold z0, popularity-stripped; target = "
                             "Wd[q] centred+stripped (item) / build_concept_dirs d_c (concept)"),
           "n_heldout": int(len(held)), "per_channel": per,
           "criterion1_fraction_of_ceiling_required": args.ceiling_frac,
           "required_A_tower_cos": {k: (None if per[k]["tower_cos"]["mean"] is None else
                                        args.ceiling_frac * per[k]["tower_cos"]["mean"])
                                    for k in ("item", "concept")},
           "item_ceiling_floor_for_gating": args.item_ceiling_floor,
           "gate_channel": gate_channel,
           "gate_rule": ("the ITEM channel is the gate unless its ceiling is near zero (< "
                         f"{args.item_ceiling_floor}), in which case the gate falls back to the "
                         "CONCEPT channel with exact-phrase recovery printed beside every number "
                         "and the item-channel weakness reported as an INSTRUMENT LIMIT, not a "
                         "model failure"),
           "controls": {"concept_dirs_reproduced_from_item_targets_worst_1_minus_abscos":
                        br.dc_reproduction_worst},
           "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": br.ckpt_sha},
           "split_hashes": man["split"]["hashes"],
           "git": {"rev": git("rev-parse", "HEAD"),
                   "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
           "bootstrap": {"n_resamples": N_BOOT, "seed": SEED},
           "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUT_DIR, "c3b_ceiling.json")
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(p + ".tmp", p)
    np.savez_compressed(os.path.join(OUT_DIR, "c3b_ceiling_per_question.npz"),
                        heldout_bank_idx=held, is_item=is_item, n_concepts=nsel,
                        tower_cos=cos_t, kphi_cos=cos_k)
    log("=" * 100)
    for nm in ("item", "concept", "all"):
        c = per[nm]["tower_cos"]
        idn = per[nm]["identification"]
        log(f"[CEILING/{nm:>7}] n={per[nm]['n']:>5} encodable {per[nm]['encodable_rate']:.4f} | "
            f"tower cos {c['mean']:+.5f} CI [{c['ci_lo']:+.5f}, {c['ci_hi']:+.5f}] | "
            f"kphi(secondary) {per[nm]['kphi_cos_SECONDARY_emitters_own_head_space']['mean']:+.5f}"
            + (f" | ident {idn['mean']:.4f} CI [{idn['ci_lo']:.4f}, {idn['ci_hi']:.4f}]"
               if idn else ""))
    log(f"[CEILING] gate channel = {gate_channel.upper()} (item ceiling {it_ceil:+.5f} vs floor "
        f"{args.item_ceiling_floor}; concept ceiling {cp_ceil:+.5f}); criterion 1 needs A to reach "
        f"{args.ceiling_frac:.0%} of it")
    log("=" * 100)
    return out

def cmd_stage2(args):
    """The registered kill-gate, AS AMENDED 2026-08-24.

    PRIMARY space = the tower (`TowerBridge`); the `k(phi)` cosine is reported as a SECONDARY row
    labelled "in the emitter's own head space" and may not carry the claim. Criteria 1 and 3 are
    evaluated PER CHANNEL and the gate channel is the one the pre-training ceiling nominated.
    Criterion 1's threshold is a FRACTION OF THE MEASURED CEILING, not an absolute cosine. Every
    statistic carries a paired-bootstrap CI.
    """
    t00 = time.time()
    man = load_manifest()
    ceil_p = os.path.join(OUT_DIR, "c3b_ceiling.json")
    assert os.path.exists(ceil_p), (
        "the instrument ceiling has not been measured -- run `ceiling` BEFORE stage2 (AMENDMENT "
        "clause 5: the gate is stated relative to a ceiling measured before training)")
    with open(ceil_p, encoding="utf-8") as fh:
        ceil = json.load(fh)
    assert ceil["split_hashes"] == man["split"]["hashes"], "the ceiling ran under a different split"

    z = np.load(PAIRS_NPZ)
    X, held, chan = z["X"], z["held_q"], z["chan"]
    with open(PHRASES_JSON, encoding="utf-8") as fh:
        phrases = json.load(fh)["phrases"]

    ensure_lm()
    tok, lm = load_lm()
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(SEED); np.random.seed(SEED)

    A, bA = load_adapter(os.path.join(OUT_DIR, "c3b_true_best.pt"), X.shape[1])
    S, bS = load_adapter(os.path.join(OUT_DIR, "c3b_scrambled_best.pt"), X.shape[1])
    for b, nm in ((bA, "true"), (bS, "scrambled")):
        assert b["split_hashes"] == man["split"]["hashes"], \
            f"the {nm} adapter was trained under a DIFFERENT split than the manifest's"
    log(f"[adapters] true ep{bA['epoch']} val CE {bA['val_ce']:.5f} | scrambled ep{bS['epoch']} "
        f"val CE {bS['val_ce']:.5f}")

    # ---- generation ------------------------------------------------------------------------------
    texts = {"true": [], "scrambled": []}
    for st in range(0, len(held), args.gen_batch):
        xb = torch.from_numpy(X[held[st:st + args.gen_batch]])
        with torch.no_grad():
            texts["true"] += greedy(lm, tok, A(xb))
            texts["scrambled"] += greedy(lm, tok, S(xb))
        if (st // args.gen_batch) % 10 == 0:
            log(f"  [gen] {min(st + args.gen_batch, len(held)):,}/{len(held):,} "
                f"({(time.time() - t00) / 60:.1f}m)")

    # ---- the PRIOR-ONLY controls -----------------------------------------------------------------
    # AMENDMENT clause 3: the registered comparator is CHANNEL-MATCHED -- a zeroed LATENT block plus
    # the channel one-hot, so criterion 2 tests correspondence and not channel membership. The strict
    # all-zero soft prefix of the original sheet is kept as a secondary reading.
    with torch.no_grad():
        pri_ch = greedy(lm, tok, A(torch.from_numpy(zero_latent_input(X.shape[1], [0, 1]))))
        strict_zero = greedy(lm, tok, torch.zeros(1, N_PREFIX, int(bA["d_lm"])))[0]
    texts["prior"] = [pri_ch[int(c)] for c in chan[held]]
    texts["prior_strict_zero_prefix"] = [strict_zero] * len(held)
    log(f"[gen] prior-only (zeroed latent + channel one-hot): item={pri_ch[0]!r} "
        f"concept={pri_ch[1]!r}; strict all-zero prefix={strict_zero!r}")

    # ---- the bridges -----------------------------------------------------------------------------
    br = TowerBridge(args)
    assert br.ckpt_sha == man["checkpoint"]["sha256"]
    ni = br.ni
    is_item = held < ni
    gold = [phrases[int(i)] for i in held]
    T_tow = br.target_tower(held)
    T_kphi = br.target_kphi(held)

    arms = ("true", "scrambled", "prior", "prior_strict_zero_prefix")
    sel, U, Vk, cos_t, cos_k, nsel = {}, {}, {}, {}, {}, {}
    for a in arms:
        sel[a] = br.select(texts[a])
        U[a] = br.tower_vectors(sel[a])
        Vk[a] = br.kphi_vectors(sel[a])
        cos_t[a] = rowcos(U[a], T_tow)
        cos_k[a] = rowcos(Vk[a], T_kphi)
        nsel[a] = np.asarray([len(s) for s in sel[a]], np.int64)
        log(f"[encode/{a}] encodable {int((nsel[a] > 0).sum()):,}/{len(held):,}; distinct texts "
            f"{len(set(texts[a])):,}")

    enc = {a: np.isfinite(cos_t[a]) for a in arms}
    both = enc["true"] & enc["scrambled"]
    all3 = both & enc["prior"]

    # ---- phrase recovery (printed BESIDE every concept-channel number, AMENDMENT clause 2) --------
    rec = {}
    for a in arms:
        ex = np.asarray([norm_txt(t) == norm_txt(g) for t, g in zip(texts[a], gold)], np.float64)
        f1 = np.asarray([tok_f1(t, g) for t, g in zip(texts[a], gold)], np.float64)
        rec[a] = {"exact": boot_mean(ex), "token_f1": boot_mean(f1),
                  "exact_items": boot_mean(ex[is_item]),
                  "exact_concepts": boot_mean(ex[~is_item]),
                  "token_f1_items": boot_mean(f1[is_item]),
                  "token_f1_concepts": boot_mean(f1[~is_item]),
                  "n_distinct_texts": int(len(set(texts[a])))}

    # ---- the three criteria, PER CHANNEL, in the PRIMARY (tower) space ----------------------------
    ceil_per = ceil["per_channel"]
    chans = {"item": is_item, "concept": ~is_item}
    crit, ident_partner = {}, {}
    for nm, m in chans.items():
        c_ceiling = ceil_per[nm]["tower_cos"]["mean"]
        need = None if c_ceiling is None else args.ceiling_frac * c_ceiling
        m1 = m & both
        c1 = boot_mean(cos_t["true"][m1] - cos_t["scrambled"][m1])
        a_lvl = boot_mean(cos_t["true"][m & enc["true"]])
        c1["A_mean_tower_cos"] = a_lvl["mean"]
        c1["A_ci"] = [a_lvl["ci_lo"], a_lvl["ci_hi"]]
        c1["ceiling"] = c_ceiling
        c1["required_fraction_of_ceiling"] = args.ceiling_frac
        c1["required_A_tower_cos"] = need
        c1["fraction_of_ceiling_reached"] = (None if not c_ceiling else
                                             (a_lvl["mean"] or 0.0) / c_ceiling)
        c1["pass"] = bool(c1["ci_lo"] is not None and c1["ci_lo"] > 0 and need is not None
                          and a_lvl["mean"] is not None and a_lvl["mean"] >= need)
        m2 = m & all3
        c2 = boot_mean(cos_t["scrambled"][m2] - cos_t["prior"][m2])
        c2["pass"] = bool(not c2["excludes_zero"])
        c2["comparator"] = "prior-only = zeroed latent + channel one-hot (channel-matched)"
        idx = np.nonzero(m & enc["true"])[0]
        if len(idx) >= 3:
            p = derange(len(idx), SEED)                 # WITHIN the channel: never a channel test
            oth = rowcos(U["true"][idx], T_tow[idx][p])
            hit = (cos_t["true"][idx] > oth).astype(np.float64)
            c3 = boot_mean(hit)
            c3["threshold"] = ID_FLOOR
            c3["mean_cos_to_random_other"] = float(np.nanmean(oth))
            c3["ceiling_identification"] = (ceil_per[nm].get("identification") or {}).get("mean")
            c3["pass"] = bool(c3["mean"] is not None and c3["mean"] >= ID_FLOOR)
            ident_partner[nm] = (idx, p, oth)
        else:
            c3 = {"n": int(len(idx)), "pass": False, "mean": None}
        crit[nm] = {"1_true_minus_scrambled": c1, "2_scrambled_vs_prior_only": c2,
                    "3_identification": c3,
                    "exact_phrase_recovery_true": rec["true"]["exact_items" if nm == "item"
                                                             else "exact_concepts"],
                    "n": int(m.sum()), "n_paired": int(m1.sum())}

    gate = ceil["gate_channel"]
    g = crit[gate]
    verdict = ("SUCCESS" if all(g[k]["pass"] for k in
                                ("1_true_minus_scrambled", "2_scrambled_vs_prior_only",
                                 "3_identification"))
               else ("channel_level_only" if (g["1_true_minus_scrambled"]["pass"]
                                              and not g["3_identification"]["pass"]) else "FAILURE"))

    # ---- disclosure (AMENDMENT clause 4): score vs genome-membership count, ITEMS -----------------
    it = np.nonzero(is_item & enc["true"])[0]
    mc = br.member_count[held[it]]
    disclosure = {"what": ("concept directions are built from member items' decoder rows, so a "
                           "held-out item leaks weakly into the tags its title matches; this is the "
                           "size of that association. REPORTED, NOT GATED."),
                  "correlation_true_item_score_vs_genome_membership_count":
                      boot_corr(cos_t["true"][it], mc),
                  "correlation_ceiling_item_score_vs_genome_membership_count": None,
                  "membership_count": {"mean": float(mc.mean()) if len(mc) else None,
                                       "median": float(np.median(mc)) if len(mc) else None,
                                       "zero_membership_items": int((mc == 0).sum())}}
    cz = np.load(os.path.join(OUT_DIR, "c3b_ceiling_per_question.npz"))
    if np.array_equal(cz["heldout_bank_idx"], held):
        ci_ = np.nonzero(is_item & np.isfinite(cz["tower_cos"]))[0]
        disclosure["correlation_ceiling_item_score_vs_genome_membership_count"] = \
            boot_corr(cz["tower_cos"][ci_], br.member_count[held[ci_]])

    def per_arm_channel(v):
        return {a: {nm: boot_mean(v[a][m & enc[a]]) for nm, m in chans.items()} for a in arms}

    out = {
        "analysis": "paperC_C3B_stage2",
        "design_sheet": "docs/design/DS_C3B_ADAPTER.md (+ AMENDMENT 2026-08-24, BINDING)",
        "produced_by": "src/instrument/c3b_adapter.py stage2",
        "citable": True, "test_touched": False, "split": "validation",
        "primary_space": ceil["primary_space"],
        "secondary_space": "k(phi) -- IN THE EMITTER'S OWN HEAD SPACE; may not carry the claim",
        "gate_channel": gate,
        "gate_channel_source": "c3b_ceiling.json, measured BEFORE training",
        "n_heldout": int(len(held)), "n_items": int(is_item.sum()),
        "n_concepts": int((~is_item).sum()),
        "n_paired_true_vs_scrambled": int(both.sum()), "n_paired_all_three": int(all3.sum()),
        "n_non_encodable": {a: int((~enc[a]).sum()) for a in arms},
        "mean_concepts_selected": {a: float(nsel[a].mean()) for a in arms},
        "adapters": {"true": {k: bA[k] for k in ("epoch", "val_ce", "train_ce")},
                     "scrambled": {k: bS[k] for k in ("epoch", "val_ce", "train_ce")}},
        "criteria_by_channel": crit,
        "gate_criteria": crit[gate],
        "verdict": verdict,
        "reading_rule": ("criterion 1 missing on the gate channel => the vector does not condition "
                         "the text at this scale (HARD RULE 6 weak-method report). Criterion 3 "
                         "missing with 1 passing => channel-level conditioning ONLY; no "
                         "faithfulness language. If the gate fell back to the concept channel, the "
                         "item-channel weakness is an INSTRUMENT limit, not a model failure, and "
                         "the exact-phrase-recovery rate must be quoted beside every number."),
        "ceiling": {"item": ceil_per["item"]["tower_cos"], "concept": ceil_per["concept"]["tower_cos"],
                    "identification": {k: ceil_per[k].get("identification") for k in chans}},
        "levels_tower_by_channel": per_arm_channel(cos_t),
        "levels_kphi_SECONDARY_emitters_own_head_space": per_arm_channel(cos_k),
        "phrase_recovery": rec,
        "disclosure_membership_leak": disclosure,
        "prior_only_texts": {"item": pri_ch[0], "concept": pri_ch[1],
                             "strict_zero_prefix": strict_zero},
        "examples": [{"bank_idx": int(held[j]), "channel": "item" if is_item[j] else "concept",
                      "gold": gold[j], "true": texts["true"][j],
                      "scrambled": texts["scrambled"][j],
                      "tower_cos_true": None if not np.isfinite(cos_t["true"][j])
                      else float(cos_t["true"][j]),
                      "tower_cos_scrambled": None if not np.isfinite(cos_t["scrambled"][j])
                      else float(cos_t["scrambled"][j])}
                     for j in range(0, len(held), max(len(held) // 20, 1))],
        "bootstrap": {"n_resamples": N_BOOT, "seed": SEED,
                      "kind": "paired percentile over held-out questions, within channel"},
        "reencode_recipe": {"source": "c3_poc_reencode (IMPORTED)", "sbert_model": br.RE.SBERT_MODEL,
                            "pooling": br.RE.SBERT_MODE, "topn": br.RE.SBERT_TOPN,
                            "mincos": br.RE.SBERT_MINCOS,
                            "fold": "certified ConceptFoldNet cv3 on the cold tower fold z0, v=+1",
                            "whitening": "popularity axis u1 stripped from both sides"},
        "controls": {"concept_dirs_reproduced_from_item_targets_worst_1_minus_abscos":
                     br.dc_reproduction_worst},
        "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": br.ckpt_sha},
        "split_hashes": man["split"]["hashes"],
        "git": {"rev": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUT_DIR, "c3b_stage2.json")
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(p + ".tmp", p)

    npz = {"heldout_bank_idx": held, "is_item": is_item,
           "genome_membership_count": np.where(is_item, br.member_count[np.clip(held, 0, ni - 1)], -1),
           "gold_phrase": np.asarray(gold, object)}
    for a in arms:
        npz[f"tower_cos_{a}"] = cos_t[a]
        npz[f"kphi_cos_{a}"] = cos_k[a]
        npz[f"n_concepts_{a}"] = nsel[a]
        npz[f"text_{a}"] = np.asarray(texts[a], object)
    for nm, (idx, p_, oth) in ident_partner.items():
        npz[f"ident_idx_{nm}"] = idx
        npz[f"ident_partner_{nm}"] = idx[p_]
        npz[f"ident_cos_other_{nm}"] = oth
    np.savez_compressed(os.path.join(OUT_DIR, "c3b_stage2_per_question.npz"),
                        allow_pickle=True, **npz)

    log("=" * 100)
    for nm in ("item", "concept"):
        c = crit[nm]
        c1, c2, c3 = (c["1_true_minus_scrambled"], c["2_scrambled_vs_prior_only"],
                      c["3_identification"])
        star = " <== GATE" if nm == gate else ""
        log(f"[{nm.upper()}]{star} n={c['n']:,} exact-phrase recovery (A) "
            f"{c['exact_phrase_recovery_true']['mean']:.4f}")
        log(f"   C1 A-A' {c1['mean']:+.5f} CI [{c1['ci_lo']:+.5f}, {c1['ci_hi']:+.5f}] | A "
            f"{c1['A_mean_tower_cos']:+.5f} vs required {c1['required_A_tower_cos']} "
            f"({args.ceiling_frac:.0%} of ceiling {c1['ceiling']}) -> "
            f"{'PASS' if c1['pass'] else 'FAIL'}")
        log(f"   C2 A'-prior {c2['mean']:+.5f} CI [{c2['ci_lo']:+.5f}, {c2['ci_hi']:+.5f}] -> "
            f"{'PASS (overlaps zero)' if c2['pass'] else 'FAIL (separates)'}")
        if c3.get("mean") is not None:
            log(f"   C3 own-closer {c3['mean']:.4f} CI [{c3['ci_lo']:.4f}, {c3['ci_hi']:.4f}] "
                f"floor {ID_FLOOR} -> {'PASS' if c3['pass'] else 'FAIL'}")
    d = disclosure["correlation_true_item_score_vs_genome_membership_count"]
    if d.get("spearman"):
        log(f"[disclosure] item score vs genome-membership count: spearman "
            f"{d['spearman']['r']:+.4f} CI [{d['spearman']['ci_lo']:+.4f}, "
            f"{d['spearman']['ci_hi']:+.4f}] (reported, not gated)")
    log(f"[VERDICT] {verdict} on the {gate.upper()} channel  ({out['minutes']:.1f}m) -> {p}")
    log("=" * 100)
    return out


# ============================================================================ cli
def main():
    ap = argparse.ArgumentParser(description="Paper C / C3B -- docs/design/DS_C3B_ADAPTER.md")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", default=E1_CKPT)
        p.add_argument("--snapshot", default=None)
        p.add_argument("--concept_ckpt", default=None)
        p.add_argument("--threads", type=int, default=4)

    def ceiling_flags(p):
        p.add_argument("--ceiling_frac", type=float, default=CEILING_FRAC,
                       help="AMENDMENT clause 5: the fraction of the true-phrase ceiling A must "
                            "reach on the gate channel (the absolute +0.10 floor is retired)")
        p.add_argument("--item_ceiling_floor", type=float, default=ITEM_CEILING_FLOOR,
                       help="below this the ITEM ceiling counts as near zero and the gate falls "
                            "back to the CONCEPT channel")

    p = sub.add_parser("prepare"); common(p)
    p = sub.add_parser("ceiling"); common(p); ceiling_flags(p)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=["true", "scrambled"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke_steps", type=int, default=0,
                   help="TIMING PROBE: run N optimiser steps, print s/step and exit. Saves nothing.")
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("earlyread")
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("stage2"); common(p); ceiling_flags(p)
    p.add_argument("--gen_batch", type=int, default=64)

    args = ap.parse_args()
    if args.cmd in ("prepare", "ceiling", "stage2"):
        from strategy_channel_suite import CERT_TOWER, CERT_CONC
        args.snapshot = args.snapshot or CERT_TOWER
        args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log(f"# PAPER C / C3B ADAPTER -- {args.cmd}. VALIDATION-side objects only; the 10,000 canonical "
        "TEST users are not built, not loaded, not touched. No LLM API.")
    log("#" * 100)
    return {"prepare": cmd_prepare, "ceiling": cmd_ceiling, "train": cmd_train,
            "earlyread": cmd_earlyread, "stage2": cmd_stage2}[args.cmd](args)


if __name__ == "__main__":
    main()
