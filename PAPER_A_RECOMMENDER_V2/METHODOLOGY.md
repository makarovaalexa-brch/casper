# Paper A — Recommender V2: a sequence recommender that folds *any* elicitation history

**Locked 2026-06-27.** Checkpoint `cache/ftra_mix_LOCKED.pt` (sha `8a30039c…`), frozen code
`continuous_actor_FROZEN.py` (block `FTRA`, env `FTRA=1 RAMIX=0.5`). Belongs in **Paper A** (the
recommender/instrument). Paper B is **not** modified (re-baselining B on this recommender is left as
future work — flagged risky).

## 1. Motivation

The Paper-A instrument is a *set-encoder* that folds a set of `(entity, answer)` tokens into a user
belief `u`, which ranks items via `score(i) = pop_i + Ql_i · u`. V1 (`enc_concept.pt`) is a simple
attention-pooling encoder trained on **full user profiles**. In conversational elicitation, however,
the encoder is queried on **partial, mixed, graded** histories (a handful of answered questions over
items *and* concepts). V1 is therefore evaluated **out-of-distribution**: it ranks full profiles well
(NDCG@10 0.407) but degrades on short elicited beliefs, and any fine-tune that specializes it to a
single elicitation policy *forgets* full-profile ranking (a narrow overfit; e.g. one such fine-tune
dropped full-profile to 0.345).

V2 fixes this with two changes: a **recurrent+attention architecture** and a **training distribution
that spans all elicitation histories**.

## 2. Architecture (recurrent + attention)

Following our DAHCR conversational recommender (Makarova et al.), the encoder processes the interaction
history as a **sequence**, not a permutation-invariant set:

```
tokens x_t = [entity_embedding_t (D) ; answer_t (1)]        # D=64
h = ReLU( x + MultiHeadSelfAttention(x, x, x) )             # Vaswani, 4 heads, key-padding-masked
o = GRU(h)                                                  # recurrent over the history
u = Linear( o[last valid step] )  ∈ R^D                     # belief
```

Multi-head self-attention contextualizes the answers against each other; the GRU integrates them
sequentially (order-aware, suited to interactive Q&A); the last valid hidden state is read out to the
belief space. Hidden size H=128. (V1 = the `inp→attn-pool→val` set encoder, no recurrence.)

## 3. Training distribution (all sequences + graded mix)

The encoder is trained (BPR ranking loss on held-out likes, Adam, weight decay 1e-5) on a **mixture of
two history types, at all lengths**, so every query distribution it will face at deployment is
in-distribution:

1. **Real-rating profile foldings** — a random-length prefix `t ∈ {1…|profile|}` of the user's rated
   items with their *true* residual ratings. Spans the full-profile task *and* every partial length.
2. **Graded-answer question sequences** — `t ∈ {1…8}` questions drawn from the entropy ranking (most
   divisive entities) or uniformly at random over the **unified pool (items + concepts)**, answered by
   the *geometric* graded response `a_k = u*·entity_k` (predicted rating from the user's taste `u*`,
   the V1 frozen profile encoding). This is the elicitation distribution.

Each minibatch is one type or the other (mix probability 0.5). The held-out target set and the
train/test user split are disjoint; answers are derived from the user's taste, never from held-out
items (no leakage).

## 4. Results (seed-avg {1,2,3,7,11}, ML-1M, te[300:], NDCG@10 full / Cremonesi tail)

| task | V2 (this) | V1 frozen | narrow fine-tune (FTREC) |
|---|---|---|---|
| **full-profile** (native recommendation) | **0.4275 / 0.2296** | 0.408 / 0.214 | 0.345 / 0.136 |
| **8-question elicitation** (entropy + graded) | **0.4011 / 0.1831** | 0.367 / 0.158 | 0.385 / 0.165 |

V2 is **better than V1 on the general full-profile task (+0.020 / +0.016)** *and* lifts 8-question
elicitation to **0.401 / 0.183** — the best elicitation result on this instrument (for reference, the
Paper-B discrete SOTA CASPER-R scores 0.360 / 0.152 with its own policy). Crucially V2 is **not a
narrow overfit**: unlike a single-policy fine-tune, it retains (improves) full-profile ranking.

### Attribution (ablation, seed-avg)
- **All-sequence training recipe** drives the full-profile gain: even the *simple* V1 architecture,
  re-trained on all lengths, reaches 0.430 / 0.226 full-profile.
- **GRU+attention architecture** is essential for the **mixed** (profiles + graded questions)
  distribution: the simple architecture *collapses* on the mix (full 0.395, elicit 0.350 — below
  frozen-elicit), whereas GRU+attention handles both (0.428 / 0.401). On full-profile-only the
  architecture's distinct contribution is on the tail (+0.020).

## 5. Reproduce

```bash
cd C:/dev/phd/casper
# train: FTRA=1 RAMIX=0.5 RAN=2500 RAEP=40 RALR=1e-3 python scripts/paper2/continuous_actor.py
# eval (seed-avg): FTRA=1 RAMIX=0.5 RALOAD=1 SEED=$S python scripts/paper2/continuous_actor.py
# use as the instrument elsewhere: LOADREC=cache/ftra_mix_LOCKED.pt <any block>
```

## 6. Scope / honesty
- This upgrades the **instrument**, so Paper-B numbers (measured on V1) would shift if re-run on V2;
  re-baselining B is future work (not done — risky, and B's relative findings should be re-verified
  first; see the ablation-replication note alongside this lock).
- The elicitation gain is the *recommender's* contribution under a fixed simple entropy+graded policy;
  learned-policy gains on top of V2 are a separate (open) question.

## 7. Policy is entropy (resolved) + consolidated comparison

Learned elicitation policies were trained against V2 (the aligned/de-OOD'd recommender) two ways:
profile-match distillation (reconstruct u*) → 0.313/0.122, and NDCG-direct REINFORCE → 0.368/0.160.
Both LOSE to the static entropy questionnaire on V2 (0.400/0.186). Across the project ~10 learned-policy
attempts never beat entropy. Conclusion: **the elicitation policy is near-optimally a simple static
entropy questionnaire; the gain is the recommender, not the policy** (interpretable, robust — a feature).

Consolidated (ML-1M, te[300:], NDCG@10 full / tail; CASPER-R ablations reproduce on V1):
```
                                          FULL    TAIL
V1 random (concept):                      0.331   0.131
V1 conc_pop (concept):                    0.343   0.128
V1 entropy (CONCEPT-only, binary cans):   0.362   0.139   # the Paper-B "entropy"
V1 CASPER-R (learned concept policy):     0.361   0.150   # = known 0.360/0.152
V1 uent+GRAW (item-answerable, graded):   0.367   0.158
V2 uent+GRAW (item-answerable, graded):   0.401   0.183   # +0.040/+0.033 vs CASPER-R
V2 full-profile:                          0.428   0.230
```
