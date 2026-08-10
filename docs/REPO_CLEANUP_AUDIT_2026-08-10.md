# Repo cleanup audit — 2026-08-10

> **EXECUTED the same day, author-approved.** `.cache` **11 GB → 8.2 GB**; 58 zero-byte `.err` removed;
> 33 of 35 `t2i26c` epoch checkpoints, the three `t2warm` generations, both retired-arena probe caches and
> the i25-EP4 cache deleted. Four superseded docs moved to `docs/archive/` and every archived file
> bannered. **Gate passed after deletion: 165/165 checks, and the certified chain still constructs** —
> ep31 loads (arch i26, val_full .3466) and the RecVAE teacher resolves, now from `models/`.
>
> **The audit changed during execution.** Answering "any more to keep?" surfaced that
> `models/i26c_ep31/` was **not self-sufficient**: `build_i26` needs the frozen RecVAE decoder to
> construct the architecture before ep31's weights can load, and that 347 MB file existed only in the
> gitignored scratch we were about to prune. Five baseline checkpoints (1.05 GB) were rescued to
> `models/baselines_ml25m_liang/` with a committed sha256 manifest before anything was deleted. Had the
> cleanup run as first written, the certified tower would have become unloadable — the exact
> `anchor-provenance-hole-pbc-recvae` failure, one level up.

This was a classification for the author to approve. The project has a scar
here: pbC's recorded $0.4946$ became permanently unreproducible when its weights were lost
(memory `anchor-provenance-hole-pbc-recvae`), so the rule applied throughout is *a file that backs a
number we still cite is not clutter, whatever its size*.

## The headline

| area | size | verdict |
|---|---|---|
| `.cache/` | **11 GB** | gitignored scratch. ~2.5 GB is provably superseded. |
| `data/` | 9.0 GB | **KEEP ALL** — raw MovieLens + derived splits, not clutter |
| `experiments/` | 293 MB | keep the JSONs; ~500 log/`.out`/`.err` files are transient |
| `models/` | 39 MB | **KEEP ALL** — this is the provenance store, and it works |
| `docs/` | 3.0 MB | not a size problem; a *navigability* problem (85 files, several superseded) |

**The single most reassuring finding:** `models/i26c_ep31/` already contains
`t2i26c_ep31.pt`, `concept_cv3_ep08.pt` and `concept_cv5_ep24.pt`. The certified artefacts are preserved
*outside* the gitignored scratch directory. That is what makes most of `.cache` safe to reclaim.

---

## A. `.cache/` — 11 GB, gitignored

### A1. Safe to delete, superseded by a certified copy — **~2.2 GB**

| what | size | why |
|---|---|---|
| `t2i26c_ep{01..35}.pt` **except ep31** | ~1.1 GB | training trajectory of the certified run. ep31 is certified and already in `models/`. Keep **ep23** as well (it is the one comparison point that was ever cited against ep31); delete the other 33. |
| `t2warm.pt`, `t2warm_v2.pt`, `t2warm_v3.pt` (+ `_best`, `_FORK_ep6`) | ~1017 MB | the pre-i25 warm-start line, superseded twice over (i25 → i26c). No current script default points at them. |
| `probe2_ndF.npy`, `probe2_ndT.npy` | 458 MB | probe caches from the retired arena; regenerable and not cited. |
| `cd_cache_seed4242_m16_t2i25_EP4_SNAP.pt` | 271 MB | a *cache* of an i25 EP4 snapshot — two towers behind. |

### A2. Check before deleting — small, and still named by a script default
`belief_i25.pt`, `cd_s1_l1_best.pt`, `cfold_best.pt`, `cfold_signed_best.pt`, `cfull_best.pt`,
`cv3_ep11.pt` are all referenced as defaults in current source. They are small. **Leave them.**

### A3. Keep regardless
`t2final_best.pt` (32 MB). It is the pre-certification tower and it backs the *retracted* R5 numbers,
which we preserved deliberately in `sign_probe_t2final_best.json`. Cheap insurance against ever having to
show what the retraction changed.

### A4. The concept epoch series
45 × `cv*_ep*.pt`, 169 MB. `cv3_ep08` is certified and copied to `models/`. The rest are trajectory.
Same treatment as A1 but low value — 169 MB is not worth the risk of a mistake. **Defer.**

---

## B. `experiments/` — keep the results, sweep the exhaust

* **KEEP every `.json`.** These are what `verify_chapter_numbers.py` resolves its 165 checks against.
  Deleting any of them breaks the verifier, which is the thing standing between us and another
  "one row, two numbers" week.
* **~500 `.log` / `.out` / `.err` files**, of which **56 are zero-byte `.err`**. The zero-byte ones are
  pure noise. The rest are the run trace, and several times this week the *log* was the only place a
  defect was visible (the OPENBLAS warning, the flat lambda sweep, the 17-hour stall). Suggest: delete
  zero-byte files, keep the rest until Paper B is done, then archive as one tarball.
* **3 partial/interrupted JSONs** (`*PARTIAL*`, `*.partial.json`) — by design from HARD RULE 11(a)
  incremental persistence. Delete the ones whose full run later completed.

---

## C. `docs/` — the real clutter, and it is about navigation not bytes

85 markdown files. The problem is that a cold-start reader cannot tell which are current.

### C1. Superseded, should be merged or marked
| file | issue |
|---|---|
| `STATE_OF_PLAY_2026-08-03.md` (30 KB) | Was the cold-start doc. `STATE.md` now is. Two documents claiming to orient a new reader is exactly how the stale-snapshot defect happened. **Add a SUPERSEDED banner pointing at `STATE.md`**, do not delete — it holds narrative not duplicated anywhere. |
| `PAPER_A_STATE_2026-08-04.md` | Snapshot of a state that has since changed substantially. Banner + keep. |
| `TODO.md` (Jul 26) | Predates the task system now in use. Reconcile once, then retire. |
| `REPO_AUDIT_2026-07-20.md` | A previous audit of this same kind. Worth reading before acting on this one. |

### C2. Keep, they are the archive by design
`docs/results/*` (46 files) and `docs/design/*` (26). Our own recording protocol names these as the home
for detailed results and design sheets. "Not referenced by filename" is not evidence of staleness — most
are cited through memory notes instead. **No action.**

### C3. Genuinely current
`STATE.md`, `VISION.md`, `CLAUDE.md`, `HARSH_REVIEW_POINTS.md`, `EXPERIMENTS.md`, `PUBLICATION_PLAN.md`,
both `PAPER_A_FABLE_REVIEW_*`, `PAPER_A_READTHROUGH_2026-08-05.md`.

---

## Recommended order

1. **Zero-byte `.err` files** (56). Zero risk.
2. **`.cache` A1** (~2.2 GB). Reclaims most of the space in one step; everything in it is superseded by a
   certified copy that lives outside `.cache`.
3. **Doc banners** on the four superseded files in C1. Costs minutes, fixes the navigability problem that
   actually bit us this week.
4. Defer A4 and the log archive until Paper B is done.

**Verification gate before deleting anything in `.cache`:** re-run `verify_chapter_numbers.py` and confirm
165/165 afterwards. It asserts the certified checkpoint by name, so if a deletion breaks provenance the
checker says so immediately rather than in three months.
