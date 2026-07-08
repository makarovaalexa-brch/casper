# I2.5 Phase 4 -- FAIR granularity-ladder probing test (survivorship bug fixed)

Date 2026-07-08. Script `scripts/i25_phase4_fair.py`. NO LLM calls. Canonical scripts/caches untouched; `i25_phase4.py` preserved as evidence.

## The fix
`i25_phase4.eval_arm` set per-user NDCG to NaN for users with no foldable token at a turn (all-refusal users) and averaged with `nanmean`, so each arm was scored over a DIFFERENT user subset. The popular-item static thus posted an impossible 0.4017 anytime at 0.10 coverage. Here EVERY arm is scored over the SAME full user set (n=298) EVERY turn: a refusal consumes the turn and leaves the belief unchanged; a user with no answers yet sits at cold belief z=0 (NDCG=0.1481); no user dropped, no NaN. Schedule builders average over ALL users too.

## Scope (author's final design)
Open recall is OUT of the adaptivity verdict. The test is PURE system-selected probing over a coarse->fine granularity ladder; each candidate carries continuous granularity g=-log(pop answer-rate).

## Config
- ML-25M; RecVAE-d512 + I2.5 learned fold (.cache/i25_fold_best.pt, val 0.4629); belief z_t=fold(answers so far), cold=z0.
- Horizon T=24; headline budgets T=(8, 16, 24); NDCG@10; paired per-user bootstrap (BOOT=5000, seed=0).
- Ladder: 227 candidates, L0..L4 counts={0: 28, 1: 20, 2: 19, 3: 80, 4: 80}, mean g per level={0: 0.465, 1: 0.001, 2: 0.088, 3: 2.117, 4: 2.501}.
  L0 attr (genre/decade), L1 broad concepts, L2 niche concepts, L3 popular items, L4 niche items.

## Schedules (fair greedy, level:key)
- s1 concepts: [L1:C:136, L2:C:146, L2:C:3, L2:C:140, L2:C:138, L2:C:78, L2:C:12, L2:C:67, L2:C:74, L1:C:71, L1:C:0, L1:C:143, L1:C:72, L2:C:135, L1:C:77, L1:C:11, L1:C:134, L2:C:75, L2:C:142, L2:C:70, L2:C:137, L1:C:145, L1:C:2, L1:C:139]
- s3 popular items: [L3:I:469, L3:I:12251, L3:I:1154, L3:I:4739, L3:I:446, L3:I:1165, L3:I:2602, L3:I:579, L3:I:4636, L3:I:357, L3:I:49, L3:I:1139, L3:I:592, L3:I:1141, L3:I:749, L3:I:2698, L4:I:337, L3:I:2416, L4:I:218, L3:I:5641, L3:I:574, L3:I:3373, L3:I:3998, L4:I:583]
- s4 mixed ladder: [L0:A:gen:5, L0:A:gen:10, L0:A:gen:9, L0:A:gen:2, L2:C:146, L0:A:gen:12, L0:A:gen:11, L0:A:gen:13, L0:A:gen:1, L0:A:gen:18, L0:A:dec:1990, L0:A:gen:17, L0:A:dec:1970, L0:A:gen:7, L0:A:dec:1980, L2:C:141, L0:A:dec:2000, L0:A:gen:3, L0:A:gen:16, L0:A:gen:4, L0:A:gen:14, L0:A:gen:0, L2:C:137, L2:C:138]

## STATIC family (fair; NDCG@10 anytime/endpoint at each budget; delta vs s4 mixed)

| arm | any/end @8 | any/end @16 | any/end @24 | delta-any@24 vs s4 [CI] | ansT | depl | note |
|---|---|---|---|---|---|---|---|
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | -0.0086[-0.0148,-0.0021] | 22.9 | yes | broad+niche concept greedy |
| s2 concepts+skip | 0.2117/0.2097 | 0.2105/0.2094 | 0.2102/0.2097 | -0.0087[-0.0149,-0.0022] | 24.0 | yes | refused turn refunded |
| s3 popular-item | 0.2303/0.2683 | 0.2602/0.3035 | 0.2786/0.3236 | +0.0598[+0.0442,+0.0754] | 3.4 | yes | SANITY: fixed item list, most refuse |
| s4 mixed ladder | 0.2197/0.2202 | 0.2195/0.2182 | 0.2188/0.2165 | +0.0000[+0.0000,+0.0000] | 19.8 | yes | BASELINE to beat |

**s3 SANITY GATE (pre-registered: must come out WEAK, else STOP and diagnose):** any@8=0.2303 vs s1 0.2120 -> WEAK=False; any@24=0.2786.

**STOP-AND-DIAGNOSE (executed):** s3 is STRONG, and the diagnosis shows it is REAL, not the survivorship bug returning:
1. Fair averaging verified: all 298 users in every mean; s3 mean answered turns = 3.4/24 (1.3 within t<=8); non-answerers sit at cold 0.1481 and are IN the average (t1 mean 0.1720 < concept statics' t1 -- the refusal tax is visible, unlike the buggy 0.4017).
2. The strength is arithmetically consistent with the certified gates: G-fold1 says ONE real item answer from cold = +0.095 (approx 18x a concept answer, I25_FOLD_RESULTS), and the s3 pool is the top-coverage frontier (10-27% of users rated each), so ~1-2 answers by t8 / ~3-4 by t24 buy 0.27-0.32 while concepts plateau at 0.21.
3. The prior 'fixed item lists die on answerability' result belongs to the CATALOGUE-SCALE regime (0.03% answerable, LLM-grid item questions). This arena's item probes are drawn from the 160 highest-coverage study items (structural answerability = user rated it), 2-3 orders of magnitude more answerable. Under a fold that rewards item answers this heavily, a popular-item static is genuinely strong here -- an arena fact (the two-regime boundary: answerability base rate x channel bandwidth), not an evaluation artifact.

## ADAPTIVE family (delta vs s4 mixed ladder)

| arm | any/end @8 | any/end @16 | any/end @24 | delta-any@24 vs s4 [CI] | ansT | depl | note |
|---|---|---|---|---|---|---|---|
| a2-blind climber | 0.2107/0.2097 | 0.2100/0.2094 | 0.2098/0.2094 | -0.0090[-0.0155,-0.0024] | 23.5 | yes | blind est-answerability from answers/refusals |
| a2-table climber | 0.2172/0.2231 | 0.2324/0.2645 | 0.2488/0.2886 | +0.0299[+0.0144,+0.0458] | 24.0 | PRIV/ref | PRIVILEGED true-table selector = policy ceiling |
| r6 emergent | 0.2904/0.3277 | 0.3258/0.3791 | 0.3484/0.4045 | +0.1296[+0.1114,+0.1482] | 24.0 | yes | concept+own-item marginal-info (own-item = mild recall) |
| u1 clairvoyant | 0.4850/0.4942 | 0.4878/0.4888 | 0.4872/0.4802 | +0.2684[+0.2429,+0.2940] | 24.0 | PRIV/ref | PRIVILEGED true-NDCG greedy = arena ceiling |

Best static (verdict opponent) = **s3 popular-item**.

## Headline contrast a2-blind vs s4 (SAME channels, blind conditioning is the only diff)

| budget T | a2-blind any | s4 any | delta [95% CI] | a2-blind end | s4 end | delta-end [CI] |
|---|---|---|---|---|---|---|
| 8 | 0.2107 | 0.2197 | -0.0090[-0.0153,-0.0026] | 0.2097 | 0.2202 | -0.0105[-0.0189,-0.0019] |
| 16 | 0.2100 | 0.2195 | -0.0096[-0.0162,-0.0029] | 0.2094 | 0.2182 | -0.0088[-0.0154,-0.0020] |
| 24 | 0.2098 | 0.2188 | -0.0090[-0.0155,-0.0024] | 0.2094 | 0.2165 | -0.0071[-0.0134,-0.0006] |

## First separation (a2-blind belief(t) vs s4 belief(t), CI excl 0): NEVER within T=24

## NDCG@10(t) curves (t=1..24)

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 concepts | 0.215 | 0.214 | 0.212 | 0.211 | 0.211 | 0.211 | 0.211 | 0.210 | 0.210 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.210 | 0.209 | 0.209 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 |
| s2 concepts+skip | 0.215 | 0.214 | 0.212 | 0.211 | 0.211 | 0.211 | 0.210 | 0.210 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 |
| s3 popular-item | 0.172 | 0.197 | 0.216 | 0.230 | 0.244 | 0.254 | 0.262 | 0.268 | 0.274 | 0.280 | 0.285 | 0.289 | 0.293 | 0.296 | 0.299 | 0.304 | 0.308 | 0.310 | 0.312 | 0.314 | 0.316 | 0.319 | 0.322 | 0.324 |
| s4 mixed ladder | 0.215 | 0.219 | 0.221 | 0.221 | 0.221 | 0.220 | 0.220 | 0.220 | 0.221 | 0.220 | 0.220 | 0.219 | 0.219 | 0.219 | 0.218 | 0.218 | 0.217 | 0.217 | 0.218 | 0.218 | 0.218 | 0.218 | 0.217 | 0.217 |
| a2-blind climber | 0.214 | 0.213 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.210 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 | 0.209 |
| a2-table climber | 0.217 | 0.215 | 0.213 | 0.207 | 0.216 | 0.221 | 0.226 | 0.223 | 0.228 | 0.236 | 0.238 | 0.242 | 0.252 | 0.257 | 0.262 | 0.264 | 0.268 | 0.271 | 0.274 | 0.286 | 0.287 | 0.288 | 0.290 | 0.289 |
| r6 emergent | 0.236 | 0.252 | 0.277 | 0.296 | 0.303 | 0.312 | 0.319 | 0.328 | 0.339 | 0.348 | 0.353 | 0.357 | 0.365 | 0.372 | 0.375 | 0.379 | 0.382 | 0.387 | 0.388 | 0.393 | 0.394 | 0.400 | 0.401 | 0.405 |
| u1 clairvoyant | 0.440 | 0.480 | 0.490 | 0.493 | 0.494 | 0.494 | 0.494 | 0.494 | 0.493 | 0.492 | 0.492 | 0.491 | 0.491 | 0.489 | 0.489 | 0.489 | 0.488 | 0.488 | 0.488 | 0.488 | 0.487 | 0.485 | 0.483 | 0.480 |

## Extrapolation sanity (fold trained on reveal lengths 1-16)
s4 NDCG step t16->t17 = -0.0007; a2-blind step = +0.0004. Cliff (drop>0.03) = False. No cliff; T=24 valid.

## Granularity trajectory g(t) (mean over users; higher g = finer)

| t | a2-blind g | a2-table g | s4 g(schedule) |
|---|---|---|---|
| 1 | 0.000 | 0.841 | 0.052 |
| 2 | 0.017 | 0.031 | 0.313 |
| 3 | 0.000 | 0.248 | 1.175 |
| 4 | 0.054 | 0.579 | 0.303 |
| 5 | 0.020 | 0.441 | 0.034 |
| 6 | 0.001 | 0.511 | 0.429 |
| 7 | 0.014 | 0.574 | 0.515 |
| 8 | 0.004 | 0.352 | 0.168 |
| 9 | 0.033 | 0.571 | 0.034 |
| 10 | 0.020 | 0.482 | 0.673 |
| 11 | 0.002 | 0.546 | 0.024 |
| 12 | 0.137 | 0.535 | 0.285 |
| 13 | 0.046 | 0.597 | 0.389 |
| 14 | 0.004 | 0.524 | 0.003 |
| 15 | 0.007 | 0.591 | 0.208 |
| 16 | 0.025 | 0.523 | 0.007 |
| 17 | 0.001 | 0.511 | 0.355 |
| 18 | 0.001 | 0.540 | 0.216 |
| 19 | 0.062 | 0.525 | 0.027 |
| 20 | 0.024 | 0.590 | 0.014 |
| 21 | 0.002 | 0.474 | 0.048 |
| 22 | 0.013 | 0.615 | 0.027 |
| 23 | 0.005 | 0.505 | 0.055 |
| 24 | 0.014 | 0.464 | 0.020 |

## Climb-vs-outcome (a2-blind): Pearson r(mean-selected-g, endpoint@24) = +0.109; high-climb users end 0.2263 vs low-climb 0.1295 (split at median g).

## Channel strength at fixed budget 8 (pure level, fair; descriptive)

| level | 8x-pure any@8 | 8x-pure end@8 | mean g | note |
|---|---|---|---|---|
| L0 attributes | 0.2104 | 0.2106 | 0.029 | broadest-first |
| L1 broad concepts | 0.2106 | 0.2093 | 0.000 | broadest-first |
| L2 niche concepts | 0.2105 | 0.2096 | 0.011 | broadest-first |
| L3 popular items | 0.1984 | 0.2155 | 1.539 | broadest-first |
| L4 niche items | 0.1733 | 0.1919 | 2.225 | broadest-first |
| _ref: all-item-8 (own items, OPEN RECALL - out of scope)_ | 0.2921 | 0.3443 | - | reference only |

## NDCG@50 (headline arms; same plans rescored; u1 omitted -- its selection objective is @10)

| arm | any@8 | any@16 | any@24 | end@24 |
|---|---|---|---|---|
| s1 concepts | 0.2132 | 0.2120 | 0.2113 | 0.2095 |
| s4 mixed ladder | 0.2140 | 0.2145 | 0.2144 | 0.2147 |
| a2-blind climber | 0.2118 | 0.2113 | 0.2108 | 0.2095 |
| a2-table climber (PRIV) | 0.2183 | 0.2351 | 0.2503 | 0.2892 |

## VERDICT (pre-registered A/B/C; opponent = BEST fair static)

- Best fair static opponent: **s3 popular-item** (any@24 0.2786). NOTE: the s4 myopic greedy never picked items (locked into attrs/concepts at ~0.22), so the strongest FIXED schedule found is the item list s3 -- the verdict is scored against it.
- u1 clairvoyant (PRIV) vs best static: T8 +0.2546[+0.2332,+0.2766], T16 +0.2277[+0.2071,+0.2481], T24 +0.2086[+0.1890,+0.2285] -> beats=True.
- a2-table (PRIV) vs best static: T8 -0.0131[-0.0282,+0.0022], T16 -0.0278[-0.0434,-0.0125], T24 -0.0298[-0.0459,-0.0137] -> beats=False (marginal-info x true table LOSES to the myopic item stack -- the E0e anti-correlation of marginal info with ranking value, reproduced under the valid fold).
- a2-blind vs best static: T8 -0.0196[-0.0362,-0.0020], T16 -0.0502[-0.0670,-0.0329], T24 -0.0688[-0.0859,-0.0514] -> beats=False.
- r6 emergent (own-item pool = mild recall, LABELLED) vs best static: T8 +0.0601[+0.0435,+0.0769], T16 +0.0656[+0.0495,+0.0821], T24 +0.0698[+0.0541,+0.0863] -> beats=True (not a verdict arm; scope note).
- s3 sanity gate WEAK=False (diagnosed above). First-sep (a2-blind vs s4)=never.

### -> BRANCH B -- prize EXISTS, DISCOVERY is the bottleneck: privileged knowledge (u1 beats=True, a2-table beats=False) beats the best fair static, but the blind climber -- conditioning only on answers/refusals -- cannot discover per-user answerability fast enough to capture it.

## ASSUMPTIONS / judgment calls
1. Cold belief = z=0 (E0 convention); refusal = no-op turn, belief unchanged, user retained.
2. Granularity g = -log(population answer-rate); levels L0..L4 by kind + median split (concepts by answer-rate median, items by popularity median). Continuous g reported in g(t).
3. Attribute answerability (L0) = structural: user has >=1 known member of that genre/decade (genre/decade not in the LLM judged grid; decade carries no genre tagvec).
4. Item probe answerable iff the user rated it (real centered rating); shared item pool = coverage>=3 items, top 160 by coverage (the answerable-item frontier; truly zero-coverage items are excluded as they would be universal refusals).
5. Greedy forward selection is prefix-consistent, so one greedy-to-24 schedule is simultaneously the fair greedy schedule for T=8/16/24 (no post-hoc truncation advantage).
6. a2-blind estimate: ghat = sum of answered questions' genre tagvecs (refusal -0.5x); est_ans = clip(pop_rate + 0.5*max(cos(tagvec,ghat),0), 0,1); score = marginal-info x est_ans. Coarse->fine is NOT hard-coded -- it can only emerge from this tradeoff.
7. a2-table / u1 use the TRUE answerability table (privileged; labelled; not deployable).
8. r6 emergent reused from i25_phase4 (concept + own-rated-item pool; the own-item channel is a mild open-recall flavor -> labelled, kept for continuity not as the verdict arm).
9. Fold trained on reveal lengths 1-16; T=17..24 is mild extrapolation -- cliff-checked (t16->t17 step reported); study capped at T=16 only if a cliff appears.
10. u1 candidate pool = the answerable ladder subset (probing ceiling), NOT the user's full rated catalogue (open recall out of scope).
11. Bootstrap paired per-user, BOOT=5000 seed=0 deterministic; all selectors deterministic.
12. SPEED (documented): greedy builders evaluate only the top-60 pool candidates by single-question fair cohort value (computed once over all candidates); positions beyond that ranking are never greedy-optimal in practice.
13. SPEED (documented): u1's per-user candidate pool is capped at the top-30 answerable candidates by single-token NDCG gain (measured at turn 1; the turn-1 argmax is unaffected). u1 is a privileged ceiling; the cap can only make it a slightly CONSERVATIVE ceiling.
14. Determinism: attribute candidate order is sorted (an earlier draft iterated a Python set, whose hash-randomized order perturbed climber tie-breaks by <0.006 between runs -- caught and fixed; no verdict was ever affected).


## A3 co-known prober

Date 2026-07-08. Script `scripts/i25_phase4_a3.py` (imports/reuses `i25_phase4_fair.py` UNMODIFIED). NO LLM calls; deterministic; local compute.

**Reproduction gate (existing harness, before adding anything):** s1 any@24 = 0.2103 (target 0.2103), s3 any@24 = 0.2786 (target 0.2786) -> reproduced = True.

**Pre-registered hypothesis:** a3-blind raises the mean-answered-turns (hit rate) above s3's 3.4/24 (~14%) by exploiting answered-item co-known neighborhoods. If the hit rate does not rise, adaptive conditioning found no purchase and Branch B stands unqualified.

**Arm.** Pool = the 160-item top-coverage ladder bank s3 draws from (coverage>=3), item probes only. Turn 1 = the globally best s3 item (I:469). After each turn: if ANSWERED -> next = highest coverage_prior(j) x mean co-known cosine(j | answered set), unasked, tie-break global coverage; if REFUSED -> next unasked item in global coverage order. Co-known cosine built from ML-25M training users (trU, n=159345) with all 298 study users' rows dropped (zero leak, asserted). a3-decay: a sponsoring anchor's weight decays 1->0.5->0 as its neighbors are refused (2 refusals in X's region stop probing X). a3-table (PRIVILEGED): candidates restricted to TRUE-answerable items -> hit-rate ceiling for this policy class.

| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |
|---|---|---|---|---|---|
| s3 popular-item (opponent) | 0.2303/0.2683 | 0.2602/0.3035 | 0.2786/0.3236 | 3.4 (14%) | -- |
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | 22.9 (95%) | -- |
| a3-blind | 0.1976/0.2166 | 0.2195/0.2675 | 0.2412/0.2932 | 4.4 (18%) | -0.0374[-0.0486,-0.0258] |
| a3-decay | 0.1977/0.2169 | 0.2193/0.2693 | 0.2408/0.2915 | 4.4 (18%) | -0.0378[-0.0490,-0.0262] |
| a3-table (PRIV) | 0.2874/0.3351 | 0.3216/0.3659 | 0.3386/0.3728 | 12.9 (54%) | +0.0600[+0.0461,+0.0755] |

**Contrast a3-blind vs s3 (THE contrast) and vs s1, per budget:**

| budget T | a3-blind vs s3 [CI] | a3-blind vs s1 [CI] |
|---|---|---|
| 8 | -0.0327[-0.0451,-0.0203] | -0.0144[-0.0280,-0.0004] |
| 16 | -0.0407[-0.0534,-0.0277] | +0.0089[-0.0036,+0.0216] |
| 24 | -0.0374[-0.0486,-0.0258] | +0.0310[+0.0177,+0.0444] |

**Mechanism metric -- hit rate (mean answered turns / 24):** s3 = 3.4 (~14%); a3-blind = 4.4 (~18%); a3-decay = 4.4; a3-table (ceiling) = 12.9. Hit rose vs s3? **True**. Did it convert to NDCG (beat s3, CI excl 0)? **False**.

**First separation (a3-blind belief(t) vs s3 belief(t), CI excl 0):** turn 2 (sign -).

**NDCG@10(t) curves (t=1..24):**

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 popular-item | 0.172 | 0.197 | 0.216 | 0.230 | 0.244 | 0.254 | 0.262 | 0.268 | 0.274 | 0.280 | 0.285 | 0.289 | 0.293 | 0.296 | 0.299 | 0.304 | 0.308 | 0.310 | 0.312 | 0.314 | 0.316 | 0.319 | 0.322 | 0.324 |
| a3-blind | 0.172 | 0.176 | 0.186 | 0.197 | 0.207 | 0.214 | 0.213 | 0.217 | 0.218 | 0.223 | 0.227 | 0.235 | 0.247 | 0.253 | 0.260 | 0.267 | 0.274 | 0.278 | 0.283 | 0.285 | 0.287 | 0.289 | 0.290 | 0.293 |
| a3-decay | 0.172 | 0.176 | 0.186 | 0.197 | 0.207 | 0.214 | 0.213 | 0.217 | 0.218 | 0.223 | 0.226 | 0.235 | 0.244 | 0.252 | 0.261 | 0.269 | 0.274 | 0.278 | 0.283 | 0.283 | 0.286 | 0.287 | 0.288 | 0.292 |
| a3-table (PRIV) | 0.215 | 0.244 | 0.262 | 0.287 | 0.301 | 0.322 | 0.332 | 0.335 | 0.342 | 0.347 | 0.353 | 0.356 | 0.358 | 0.361 | 0.363 | 0.366 | 0.369 | 0.372 | 0.372 | 0.373 | 0.374 | 0.375 | 0.374 | 0.373 |

**VERDICT:** A3-blind RAISES the hit rate (4.4 vs s3 3.4/24) but does NOT beat s3 on NDCG -- extra answers land on lower-value neighbors; hit rate does not convert. Branch B stands.

**ASSUMPTIONS / judgment calls (a3):**
1. Probe bank = the 160 top-coverage ladder items (coverage>=3) that s3 draws from; item probes only.
2. Co-known statistic = COSINE of binary co-rating counts, cooc(a,b)/sqrt(cooc(a,a)cooc(b,b)); diagonal zeroed (an item never scores against itself). Cosine removes each item's marginal popularity so the separate coverage prior is not double-counted.
3. Neighbor value = coverage_prior(j) x mean cosine affinity of j to the answered set. coverage_prior = study-cohort coverage (pop_rate). Affinity = MEAN over answered anchors (not sum -> not confounded with #answers).
4. Population for cooc = ML-25M training users (trU, n=159345) with all 298 study users' rows dropped; study users are eval-split and disjoint from trU (asserted leak=0). Co-rating uses each population user's FULL rated profile intersected with the bank.
5. Turn 1 = s3[0] (the globally best s3 item). Fallback (refusal) order = global coverage descending, tie-break cid. Neighborhood tie-break = coverage.
6. a3-decay: each neighborhood-selected probe records a SPONSORING anchor = raw-cosine argmax over the answered set; a refusal increments that sponsor's counter; anchor weight = max(0, 1 - 0.5*refusals) -> 1, 0.5, 0 (hard stop at 2 refusals in that region).
7. a3-table (PRIVILEGED, labelled): candidate set restricted to true-answerable items each turn -> every probe answered -> hit-rate ceiling min(24, #answerable bank items); turn 1 = first answerable item in global order.
8. No neighborhood-size cap (full 160-item bank scored each turn); refusal = no-op turn (belief unchanged), user retained; all 298 users in every mean (fair, inherited from the harness).
9. Deterministic: bootstrap paired per-user BOOT=5000 seed=0; all selectors deterministic (coverage/cid tie-breaks).

