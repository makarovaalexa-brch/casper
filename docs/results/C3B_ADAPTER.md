# C3B — does the vector condition the text? (variant (b), the cheap test)

**Status: COMPLETE (validation only; no test look). 2026-08-25.** Design sheet
`docs/design/DS_C3B_ADAPTER.md` incl. all 2026-08-24 amendments (BINDING); conclusions are Fable's
own from the banked artifacts. Code `src/instrument/c3b_adapter.py` (committed pre-run, `c6406ad` +
ruling patch `f234d3e`); artifacts `experiments/paperC/c3b/` (`c3b_stage2.json`, per-question npz,
ceiling, manifest, jsonl logs, `driver_run.log`). Adapters: 2-layer MLP → 8 soft prefix tokens into
FROZEN distilgpt2; true arm best ep4 val CE 2.991, scrambled twin best ep3 val CE 3.270; training
2×~3.7 h detached; stage-2 36.9 min. Two session-side kills cost wall-clock only (epoch checkpoints
held); the completed run is a clean single pass.

**One-line verdict: the REGISTERED composite is `FAILURE` on the item gate channel — but the
failure is criterion 2 firing in the direction OPPOSITE to the threat it guarded, while the two
mechanism-bearing criteria pass decisively. The registered label stands; what the numbers show is
that the vector conditions the text.**

## The early read (registered secondary)

Separated by epoch 2, comfortably: scrambled − true = **+0.245 train CE / +0.257 val CE**
(final bests 2.991 vs 3.270). The flag ("not separated by epoch 2 ⇒ flag immediately") did not fire.

## The gate (ITEM channel, n = 3,670 held-out items; tower space; every CI a 10k paired bootstrap)

| criterion (as amended) | bar | result | cell |
|---|---|---|---|
| **1** A − A′ tower cos, CI excl. 0 AND A ≥ 40% of ceiling | Δ>0 excl. 0; A ≥ +0.0357 | **Δ = +0.1114 [+0.1054, +0.1173]; A = +0.0953 [+0.0903, +0.1003]** | **PASS** — A exceeds not 40% but **107% of the true-phrase ceiling (+0.0892)** |
| **2** A′ ≈ channel-matched prior (CI overlaps) | overlap | **A′ − prior = −0.0133 [−0.0185, −0.0081] — excludes zero, BELOW** | **FAIL as registered** |
| **3** identification ≥ chance + 60% of ceiling headroom, CI excl. 0.5 | ≥ 0.5974 | **0.6778 [0.6631, 0.6931]** | **PASS** — above the true-phrase ceiling 0.6624 |
| composite | all three | | **FAILURE (registered)** |

Concept channel (not the gate; exact-phrase recovery printed per the amendment): C1 +0.2551
[+0.2084, +0.3005], A +0.3533 vs required +0.2002 — PASS; C3 0.8476 [0.8000, 0.8952] vs floor
0.7714 — PASS; exact-phrase recovery on items **0.0000**, token-F1 0.111.

Arm levels, tower space, items: **true +0.0953** · scrambled **−0.0161** · prior-only **−0.0028** ·
strict-zero prior **−0.0030**. Non-encodable: 1 of 3,880 (true arm). Disclosure (amendment 4):
item score vs genome-membership count Spearman +0.107 [+0.082, +0.146] — small, reported, not gated.

## Reading 1 — criterion 2 failed DOWNWARD, which is not the smuggling it guarded against

C2 was registered as an equivalence ("the twin must not smuggle signal"), written without a
direction. The twin did not smuggle: it sits **below** prior-only — training on scrambled
correspondence actively *hurts* re-encode geometry (the twin also collapsed to **61 distinct
texts** over 3,880 vectors, against the true arm's **2,280**). An equivalence criterion fired on a
benign — indeed favourable — direction. **The registered composite label stands as FAILURE**; the
pre-registration does not get reinterpreted after the data. Whether C3B-final replaces C2 with the
directional form ("A′ must not EXCEED prior") is the author's call, recorded before any rerun.
Lesson for `HARSH_REVIEW_POINTS.md`: write equivalence controls with a direction.

## Reading 2 — the mechanism finding, on the criteria that carry it

Criteria 1 and 3 are the vector-conditioning tests, and both pass with room: the true adapter's
text re-encodes toward its OWN held-out vector (+0.111 over the twin, CI far from zero) and
identifies it against a same-channel decoy at 0.678 (floor 0.597, chance 0.5). **The answer to the
sheet's question — does the text generated from a vector depend on that vector — is yes, at
distilgpt2 scale, on held-out questions, through the independent tower bridge, per-channel, with
CIs.**

## Reading 3 — machine-only synonyms, observed in our space

Exact-phrase recovery on items is **0.0000** and token-F1 0.111 — the adapter almost never emits
the gold title — yet its text re-encodes **above the gold phrase's own ceiling** (+0.0953 vs
+0.0892; identification likewise above ceiling). Examples: *Rocky Balboa (2006)* → "Rocky (2005)";
*Princess Diaries 2* → "The Princess and the Frog (2006)"; *Zelig (1983)* → "Theatrical Man
(1962)". These are Hewitt et al.'s **machine-only synonyms** appearing unprompted: text a human
would grade wrong that reproduces the vector's semantic effect better than the true surface form.
This (a) is the strongest available argument against grading rendered questions by human
plausibility, now with our own numbers; (b) means "above ceiling" needs its caveat — the bridge
prefers concept-flavoured text, and invented titles are freer to be concept-flavoured than real
ones (the same reason the exact-phrase floor is 0). It is a property of the instrument AND a real
phenomenon; C4-final must carry both sentences.

## What may be claimed / not claimed

Claim: the conditioning mechanism works (criteria 1+3, gate channel, CIs); the scrambled twin is
worse than knowing nothing (its own finding); machine-only synonyms occur in this space. Do NOT
claim: the registered composite passed (it did not); any faithfulness of a *specific* rendered
question; any value; anything about fluency (not evaluated, by design).

## Next

1. Author ruling on C2's directional rewrite for any C3B-final.
2. The C3 chapter section + Introduction can now be drafted around the mechanism result.
3. C1-proper on the retrained tree (with persisted gain vectors) feeds this same harness its real
   emitted points; the renderer for the paper then trains on the same recipe one model class up if
   fluency ever matters (it does not for the claim).
