# LLM KNOWLEDGE-CHANNEL DECOMPOSITION (D-ANS v2.1 -- citable record for sec:distill)
Author-directed equating pass, signed 2026-07-09. NO LLM calls. This file is the permanent, citable decomposition of *what the LLM judge's knowledge answers actually measure*: how much is stable per-viewer trait, how much is per-call presentation flutter, and which observable profile features genuinely predict answerability.
## Why equating was necessary
The 3-user shuffle probe (`experiments/SHUFFLE_PROBE.md`) re-sent IDENTICAL content three times with shuffled question/profile order. The judge's know-well / rough-idea cutoff FLUTTERS per call: know-well RATE spread up to **0.66** on identical content (users 85673 0.660, 23228 0.540; mid-user 2389 0.160). STARS are stable (repeat MAE ~0.12-0.20). WITHIN-CALL structure is stable (~0.99). So the per-call level shifts wholesale while the relative ordering of items inside a call is preserved. In the production battery each question sits in a FIXED call (call index = question position // 260; concept spans 5 calls, attribute 3, item ~4 per user), so this flutter leaks into per-user answer rates (few calls per channel) and contaminates the iter-2 fuel ICCs and sigma_u, which conflated trait with flutter.
## Method (two-step equating -- NOT the joint fit)
A joint refit with per-(user,call) intercepts is UNIDENTIFIABLE here: the production battery was never shuffled, so a cell's call is a deterministic function of its question position and the call intercept is collinear with question-block difficulty. We therefore used the explicitly-permitted TWO-STEP: (1) remove each QUESTION's mean answer rate (absorbs all content difficulty, including whatever block-level difficulty a call carries); (2) on the question-residualized outcome, a nested random-effects variance-components model (Searle unbalanced MoM) splits the remainder into USER-trait (constant across the user's calls, flutter-free), CALL-flutter (between-call within-user -- the presentation offset), and cell residual. Because content was removed first, any remaining between-call-within-user variation is presentation flutter, not content. The per-(user,call) offsets are the shrunk nuisance intercepts (nuisance -> pooled toward 0 by the MoM sampling-variance subtraction). Headline outcome = knows-of-it margin y=1[knowledge>=1] (the G2 fuel margin). User-feature vs user-trait split = honest 5-fold user-grouped ridge CV R^2 of the census block on the flutter-free per-user means.
## (b) HEADLINE -- variance decomposition per knowledge channel (share of Var(y))
Both knowledge margins are shown: y=1[k>=1] (knows-OF-it) and y=1[k>=2] (knows-WELL). The meaningful answerability signal differs by channel: for concept/entity the knows-of-it margin carries the fuel; for ITEM the knows-of-it margin is SATURATED (~99% of top-800 items are recognized -> Var(y)~0.009), so the item signal lives in the knows-WELL margin -- which is exactly the margin the shuffle probe measured.

| channel / margin | question-difficulty | user-FEATURE | user-TRAIT (equated) | CALL-FLUTTER | cell residual | sum |
|---|--:|--:|--:|--:|--:|--:|
| concept k>=1 (headline) | 54.6% | 0.6% | 1.3% | 4.3% | 39.3% | 100.0% |
| concept k>=2 | 53.5% | 2.3% | 1.2% | 4.2% | 38.8% | 100.0% |
| entity k>=1 (headline) | 32.0% | 0.0% | 2.5% | 16.2% | 49.3% | 100.1% |
| entity k>=2 | 35.3% | 0.2% | 1.3% | 16.1% | 47.1% | 100.0% |
| item k>=1 | 17.9% | 0.0% | 0.0% | 6.3% | 76.1% | 100.4% |
| item k>=2 (headline) | 34.3% | 0.3% | 4.5% | 21.7% | 39.2% | 100.1% |

(Shares are raw variance components / Var(y); the sum deviates from 100% only by the small crossed-design non-additivity between the question removal and the nested VC.)

**Reading the headline margins**: concept knows-of-it = 4.3% flutter / 1.3% trait; entity knows-of-it = 16.2% flutter / 2.5% trait (flutter is the LARGEST user-side component -- attribute answerability is dominated by presentation noise); item knows-WELL = 21.7% flutter / 4.5% trait.

**Reading it**: CALL-FLUTTER is the presentation artifact the shuffle probe exposed -- it is NOT viewer signal and NOT content. USER-TRAIT (equated) is the stable, flutter-free, feature-unexplained per-viewer component -- the only part a policy could learn to exploit as a durable person trait. USER-FEATURE is the part an observable-profile model already captures.
## (c) CORRECTED FUEL -- trait-only ICC vs the old flutter-inflated ICC
| channel | margin | OLD ICC one-way [95% CI] | raw VC ICC | CORRECTED trait ICC | flutter var |
|---|---|--:|--:|--:|--:|
| concept | k>=1 (headline) | 0.0338 [0.0263,0.0410] | 0.0615 | **0.0427** | 0.00819 |
| concept | k>=2 | 0.0519 [0.0418,0.0618] | 0.0932 | **0.0752** | 0.00990 |
| entity | k>=1 (headline) | 0.0920 [0.0748,0.1083] | 0.1169 | **0.0374** | 0.02350 |
| entity | k>=2 | 0.0804 [0.0662,0.0952] | 0.1062 | **0.0234** | 0.03371 |
| item | k>=1 | 0.0164 [0.0112,0.0213] | 0.0207 | **0.0000** | 0.00058 |
| item | k>=2 (headline) | 0.1367 [0.1142,0.1593] | 0.1625 | **0.0729** | 0.04619 |

The old one-way ICC attributes ALL user-level grouping to trait; because each user answers each channel over only a few calls, per-call flutter does not average out and leaks into the per-user rate, INFLATING the apparent between-user fuel. The corrected trait ICC uses only the flutter-free between-user component. The single biggest correction is the ENTITY/attribute knows-of-it channel: old ICC 0.0920 -> corrected trait 0.0374 (flutter var 0.0235 is the largest of any channel). The ITEM knows-OF-it margin is saturated so its trait ICC is 0.0000 (item recognition is not a person dial at all); the item knows-WELL margin retains a real but roughly-halved trait ICC (0.1367 -> 0.0729) once its 22%-of-variance flutter is removed.

**Flutter magnitude cross-check (item KNOW-WELL, the margin the probe measured)**: the item knows-well call-flutter variance recovered by the equating (question-residualized nested VC over 173 users) = **0.04619** vs the shuffle probe's directly-measured between-call know-well-RATE variance on identical content = **0.06773** (3 users, n_items 250). These two INDEPENDENT routes -- one from re-sending identical content 3x, one from question-residualized production data at 173-user scale -- agree to the same order of magnitude, cross-validating the flutter magnitude. The probe estimate is somewhat higher and much noisier (3 users, two of them extreme: per-user rate vars 0.110, 0.008, 0.085); the 173-user equating is the population-scale number to cite. Both confirm the item knows-well gate carries large per-call presentation flutter (~20% of that margin's variance). The formal repeat-study (below) is still required to pin the channel-by-channel flutter profile.

**Flutter-robust within-call contrasts (unchanged, for completeness)**: the near-far validity gaps (A2) are computed WITHIN a call and the probe confirmed within-call structure is stable (~0.99), so they are NOT flutter-contaminated: concept near-far(k1) = +0.105 [+0.095,+0.115]; item-mid near-far(k2) = +0.028 [+0.017,+0.039] (from g2_iter2). These content-validity signals survive equating untouched.
## (a) Which features GENUINELY predict answerability
### Per-QUESTION knowledge coefficients (standardized; sign = effect on higher knowledge; iter-2 fixed effects, unchanged by equating)
| channel | feature | coefficient |
|---|---|--:|
| concept | sat_ratedmembers | +0.525 |
| concept | tag_logmemb | -0.043 |
| concept | tag_logpw | +1.002 |
| concept | b_int_size_disp | +0.008 |
| concept | era_density | +0.121 |
| concept | era_dist | -0.009 |
| entity | sat_ratedfilmo | +0.336 |
| entity | frac_filmo | +0.201 |
| entity | entity_logpop | +0.089 |
| entity | b_int_size_disp | +0.054 |
| entity | era_density | -0.006 |
| entity | era_dist | -0.179 |
| item | coknow_mean | +0.482 |
| item | coknow_max | -0.054 |
| item | genre_align | +0.094 |
| item | fame_pr | +0.261 |
| item | fame_logcnt | +1.003 |
| item | decade_align | -0.147 |
| item | era_dist | -0.126 |

These are the content-difficulty axes: fame/co-knowledge (item), membership size + pop-weight (concept), filmography fraction + entity popularity (attribute), and taste/genre + era alignment across all three. They carry the QUESTION-DIFFICULTY share above and are the load-bearing, stable predictors.
### User-level (census) features -- ONLY what survived cross-validation, stated bluntly
| channel | in-sample R^2 (features on user rate) | 5-fold CV R^2 (honest) | verdict |
|---|--:|--:|---|
| concept | 64% | 32% | features generalize |
| entity | 29% | -28% | NO out-of-sample signal |
| item | 36% | 3% | weak |

Blunt reading: **concept** user-level answerability is genuinely profile-predictable out of sample (CV R^2 ~32%). **item** is barely predictable at the user level (CV ~3%) -- item knowledge is a QUESTION property (fame), not a stable person dial. **entity/attribute** does NOT survive CV (negative CV R^2): the census OVERFITS it; there is no reliable observable-profile predictor of per-user attribute answerability beyond the question's own fame. The matched-pair qualitative review (QUAL_REVIEW_DANS2.md) reached the same conclusion: user-level census features are partly modelling judge idiosyncrasy, not a human trait.
Ranked user-feature importance (LOUO fold-mean |beta|, top 8) is retained from iter-2 in `experiments/DANS_BUILD.md` (STANDALONE STUDY table) and `.cache/dans/decomp_iter2.json`; the signs are stable but the ABSOLUTE scale is ridge-penalty-limited (lambda hit the grid edge), consistent with weak user-level signal.
## (d) The stars-are-stable finding
The flutter is confined to the KNOWLEDGE gate (know-well vs rough-idea vs no_clue cutoff). The shuffle probe's star ratings on cells judged know-well in both re-sends are STABLE (repeat-vs-repeat MAE ~0.12-0.20 stars, comparable to the vs-original MAE). So the VALUE channel (stars) does not need equating -- only the knowledge/answerability gate flutters. This is why v2.1 equates the knowledge channel only and leaves the value refit (Stage B) to target signal strength, not flutter.
## (e) FLAGGED FUTURE TO-DO (author) -- confirm the flutter decomposition
The flutter magnitude here rests on a 3-user shuffle probe plus the question-residualized nested VC. Before any human-transfer or policy-value claim leans on the corrected trait ICCs, run a FORMAL REPEAT-STUDY on a larger user set (e.g. 30-50 users x >=3 shuffled re-sends, possibly across judge snapshots) to (i) confirm the per-call flutter variance and its channel profile, (ii) confirm that item user-trait ICC is ~0 (item answerability is a question property, not a person dial), and (iii) measure whether the flutter is judge-temperature / snapshot dependent. NO LLM calls were made for v2.1; this is a pre-registered future data-collection item, not something to synthesize.
