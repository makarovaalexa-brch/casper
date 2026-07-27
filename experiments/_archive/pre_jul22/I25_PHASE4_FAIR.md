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


## A4 pmodel-blind prober

Date 2026-07-08. Script `scripts/i25_phase4_a4.py` (imports/reuses `i25_phase4_fair.py` and `i25_phase4_a3.py` UNMODIFIED). NO LLM calls; deterministic; local compute.

**Reproduction gate (existing harness, before adding anything):** s1 any@24=0.2103 (t 0.2103), s3 any@24=0.2786 (t 0.2786), a3-blind any@24=0.2412 (t 0.2412) -> reproduced=True.

**Idea.** A3's co-rating cosine de-popularised the signal and wandered into low-coverage refusal territory (hit +1.0 but any@24 -0.037 vs s3). A4 instead ranks item probes by the VALIDATED answerability surrogate `.cache/instrument2/answerability_pmodel` (AUC .921 held-out users; features [pop_pct, log_rcount, decade, genre_match, franchise, is_concept]; log_rcount coef +1.55 DOMINANT, genre_match +0.63 the user tilt). All features except genre_match are public/static; genre_match is imputed ONLINE from a running per-user genre estimate g-hat. Popularity stays dominant; discovered taste only TILTS the ranking.

**Arm.** Pool = the same 160-item top-coverage ladder bank s3/a3 draw from (item probes only). Online g-hat: init = coverage-weighted population genre prior (low weight w0=1.0); ANSWERED item (any polarity) += Gmat[j]; REFUSED item -= 0.5*p_hat_chosen*Gmat[j] then clip>=0 (surprising refusals subtract more). Per turn pick argmax p_hat(j)*V(j), p_hat=pmodel(., genre_match=cos(g-hat,Gmat[j])), V=study-cohort coverage prior (pop_rate). a4-explore adds a small annealed novelty bonus (1+0.30*anneal_t*novelty) for low-g-hat-mass genre regions, off by turn 6.

| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |
|---|---|---|---|---|---|
| s3 popular-item (opponent) | 0.2303/0.2683 | 0.2602/0.3035 | 0.2786/0.3236 | 3.4 (14%) | -- |
| a3-blind (prior arm) | 0.1976/0.2166 | 0.2195/0.2675 | 0.2412/0.2932 | 4.4 (18%) | -- |
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | 22.9 (95%) | -- |
| a4-blind | 0.1986/0.2171 | 0.2233/0.2723 | 0.2431/0.2890 | 4.4 (18%) | -0.0356[-0.0477,-0.0231] |
| a4-explore | 0.1986/0.2171 | 0.2233/0.2723 | 0.2431/0.2890 | 4.4 (18%) | -0.0355[-0.0477,-0.0231] |

**Contrast a4-blind vs s3 (THE contrast) and vs a3-blind, per budget:**

| budget T | a4-blind vs s3 [CI] | a4-blind vs a3-blind [CI] |
|---|---|---|
| 8 | -0.0317[-0.0460,-0.0168] | +0.0010[-0.0058,+0.0073] |
| 16 | -0.0369[-0.0503,-0.0226] | +0.0038[-0.0006,+0.0080] |
| 24 | -0.0356[-0.0477,-0.0231] | +0.0018[-0.0015,+0.0050] |

**Mechanism metric -- hit rate (mean answered turns / 24):** s3 = 3.4 (~14%); a3-blind = 4.4; a4-blind = 4.4 (~18%); a4-explore = 4.4; a3-table ceiling = 12.9. Hit rose vs s3? **True**. Hit rose vs a3-blind? **True**. a4-blind BEATS s3 at T=24 (CI excl 0)? **False**.

**First separation (a4-blind belief(t) vs s3 belief(t), CI excl 0):** turn 2 (sign -).

**Calibration diagnostic (is the online p_hat honest?): per turn, mean chosen-probe p_hat vs realized answer rate.**

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean chosen p_hat | 0.958 | 0.941 | 0.960 | 0.936 | 0.887 | 0.954 | 0.924 | 0.896 | 0.922 | 0.890 | 0.901 | 0.888 | 0.883 | 0.881 | 0.907 | 0.893 | 0.889 | 0.893 | 0.883 | 0.877 | 0.883 | 0.875 | 0.875 | 0.873 |
| realized answer rate | 0.275 | 0.258 | 0.235 | 0.185 | 0.218 | 0.232 | 0.164 | 0.164 | 0.144 | 0.195 | 0.154 | 0.164 | 0.185 | 0.174 | 0.191 | 0.164 | 0.185 | 0.148 | 0.164 | 0.164 | 0.168 | 0.144 | 0.151 | 0.174 |

Mean p_hat - answer-rate gap = +0.720 (corr +0.70). p_hat is OPTIMISTIC (over-predicts answerability) -- the surrogate ranks, it is not a per-user probability oracle.

**g-hat convergence (PRIVILEGED-INFO DIAGNOSTIC ONLY -- cos of online g-hat to the user's true known-half genre distribution; NOT used by the policy):**

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cos(g-hat, true genre) | 0.864 | 0.658 | 0.653 | 0.548 | 0.492 | 0.490 | 0.465 | 0.473 | 0.458 | 0.455 | 0.460 | 0.457 | 0.469 | 0.477 | 0.477 | 0.468 | 0.471 | 0.469 | 0.459 | 0.454 | 0.445 | 0.461 | 0.457 | 0.458 |

cos@t1 0.864 -> cos@t24 0.458 (delta -0.406) -- g-hat barely moves (few answers to learn from).

**NDCG@10(t) curves (t=1..24):**

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 popular-item | 0.172 | 0.197 | 0.216 | 0.230 | 0.244 | 0.254 | 0.262 | 0.268 | 0.274 | 0.280 | 0.285 | 0.289 | 0.293 | 0.296 | 0.299 | 0.304 | 0.308 | 0.310 | 0.312 | 0.314 | 0.316 | 0.319 | 0.322 | 0.324 |
| a3-blind | 0.172 | 0.176 | 0.186 | 0.197 | 0.207 | 0.214 | 0.213 | 0.217 | 0.218 | 0.223 | 0.227 | 0.235 | 0.247 | 0.253 | 0.260 | 0.267 | 0.274 | 0.278 | 0.283 | 0.285 | 0.287 | 0.289 | 0.290 | 0.293 |
| a4-blind | 0.165 | 0.178 | 0.194 | 0.199 | 0.210 | 0.211 | 0.214 | 0.217 | 0.224 | 0.231 | 0.236 | 0.243 | 0.253 | 0.259 | 0.266 | 0.272 | 0.275 | 0.278 | 0.280 | 0.281 | 0.284 | 0.286 | 0.287 | 0.289 |
| a4-explore | 0.165 | 0.178 | 0.194 | 0.199 | 0.210 | 0.211 | 0.214 | 0.217 | 0.224 | 0.231 | 0.236 | 0.243 | 0.253 | 0.259 | 0.266 | 0.272 | 0.275 | 0.278 | 0.280 | 0.281 | 0.284 | 0.286 | 0.287 | 0.289 |

**VERDICT:** A4-blind RAISES the hit rate (4.4 vs a3-blind 4.4, s3 3.4/24) but does NOT beat s3 on NDCG at T=24 -- extra answers still land on lower-value items; hit does not convert. Branch B stands.

**ASSUMPTIONS / judgment calls (a4):**
1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3 draw from; item probes only.
2. p_hat = the fitted logistic answerability surrogate (answerability_pmodel; sigmoid(intercept + coef.(x-mean)/std)). Features pop_pct=D.pr[j], log_rcount=log(D.cnt[j]+1), decade=(year-1900)/100 (0.5 if no year), genre_match=cos(g-hat, Gmat[j]), franchise=title-regex, is_concept=0. All but genre_match are public/static per item.
3. g-hat init = coverage-weighted population genre prior (sum_j pop_rate[j]*Gmat[j], unit-normed) with evidence weight w0=1.0 (~one pseudo-item). ANSWERED (any polarity, incl dislikes -- a dislike still proves knowledge) += raw Gmat[j]. REFUSED -= 0.5*p_hat_chosen*Gmat[j], clipped >=0 (down-weight proportional to the p_hat we predicted -> surprising refusals count more).
4. V(j) = study-cohort coverage prior (pop_rate) -- the SAME coverage prior that defines s3's pool and a3's coverage_prior. Selection = argmax p_hat(j)*V(j); popularity stays dominant (V and the pmodel's log_rcount term), genre_match only tilts.
5. a4-explore bonus = (1 + 0.3*anneal_t*novelty(j)), anneal_t=max(0,1-t/6), novelty(j)=1 - (item genre distribution . g-hat evidence fraction) -- cheap directed exploration, annealed off after ~6 turns.
6. No turn-1 seeding: turn 1 uses g-hat=prior only, so the first pick emerges from the model (deployable, fully model-driven; refusals do NOT switch to a hard fallback order -- every turn is argmax p_hat*V).
7. Refusal = no-op turn (belief unchanged), user retained; all 298 users in every mean (fair, inherited from the harness). Bootstrap paired per-user BOOT=5000 seed=0; selectors deterministic (V/cid tie-breaks).
8. g-hat-to-true-genre cosine and per-turn realized answer rate are DIAGNOSTICS; the true known-half genre distribution is PRIVILEGED and never enters the policy.


## ARENA-V2 (LLM-measured answerability)

Date 2026-07-08. Script `scripts/i25_phase4_arena2.py` (imports/reuses `i25_phase4_fair.py`, `i25_phase4_a3.py`, `i25_phase4_a4.py` UNMODIFIED). NO LLM calls (grids cached); deterministic; local compute.

**Two pre-registered answerability models.** The study fixes TWO environments and reports the adaptivity verdict under BOTH: **arena-v1** = STRUCTURAL lower-bound answerability (an item probe is answerable only if the user rated it in the known half; the sections above), and **arena-v2** (this section) = the project's VALIDATED **LLM-MEASURED** answerability (cached LLM judged grid + fitted pmodel, base answer-rate 0.732). The A4 diagnosis showed the v1 harsh rule is what starves blind adaptivity of hits; arena-v2 tests whether blind adaptive probing separates from the static once answerability is the measured, human-like model. The verdict may legitimately differ between the two -- that contrast is itself a headline finding.

**Arena-v1 reproduction (old rule, before switching):** s1 any@24 = 0.2103 (target 0.2103), **s3 any@24 = 0.2786 (target 0.2786)**, s3 hit 3.4/24 -> reproduced = True.

**Arena-v2 item answer model (environment-side only; agent arms stay exactly as blind/privileged as before).** An item probe is ANSWERABLE iff (a) the user rated it in the known half (as v1), OR (b) the cached LLM judged grid (gate + main-study, union; answerable if any grid judged YES) says yes, OR (c) for a bank item NOT in that user's grid, the fitted pmodel with the user's TRUE known-half genre_match has p_hat >= BASE_RATE=0.732 (env-side => true-profile features are LEGAL in the answering rule). ANSWER VALUE: rated -> real centered rating; answerable-but-unrated -> **LLM/CF-predicted** = bank-restricted EASE (universe = top-4000 popular UNION the 160-item bank, lambda=500, 4000 items; trained on KNOWN-portion ratings with all study-user held-out rows dropped -- cross-check A recipe) + Gaussian noise sigma=0.7 stars seeded per (user,item), clipped [0.5,5], centered; native recall channel nat=item iff the noised predicted star >= 4 (the same channel v1 uses for a rated-liked item). Concepts/attributes UNCHANGED.

**Grid coverage of the 160-item probe bank:** mean 4.9/160 items judged per user (3.1%); on those judged bank items the LLM YES-rate is **0.780** (popular bank items are far more answerable than the 0.269 all-strata item base). Bank items outside a user's grid fall back to the pmodel threshold. Provenance of item-probe outcomes across all (user,bank-item) cells: {'answerable_grid': 1034, 'answerable_pmodel': 38198, 'rated': 5000, 'refuse_grid': 323, 'refuse_pmodel': 3125}.

| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3(v2) [CI] |
|---|---|---|---|---|---|
| s3 popular-item (v2, opponent) | 0.2250/0.2336 | 0.2327/0.2448 | 0.2377/0.2480 | 23.3 (97%) | -- |
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | 22.9 (95%) | -- |
| a3-blind | 0.2163/0.2203 | 0.2190/0.2223 | 0.2201/0.2261 | 24.0 (100%) | -0.0176[-0.0249,-0.0102] |
| a4-blind | 0.2084/0.2150 | 0.2109/0.2152 | 0.2135/0.2224 | 23.9 (100%) | -0.0242[-0.0340,-0.0143] |
| a3-table (PRIV, class ceiling) | 0.2064/0.2147 | 0.2117/0.2229 | 0.2152/0.2255 | 24.0 (100%) | -0.0224[-0.0325,-0.0124] |
| u1 clairvoyant (PRIV, arena ceiling) | 0.6006/0.6258 | 0.6117/0.6162 | 0.6076/0.5805 | 24.0 (100%) | +0.3699[+0.3462,+0.3935] |

**Hit rates v1 vs v2 (mean answered turns / 24):** s3 v1 = 3.4 (~14%) -> **s3 v2 = 23.3 (~97%)**; a3-blind = 24.0; a4-blind = 23.9; a3-table (ceiling) = 24.0; u1 = 24.0.

**THE verdict contrasts (per budget):**

| budget T | a3-blind vs s3 [CI] | a4-blind vs s3 [CI] |
|---|---|---|
| 8 | -0.0086[-0.0169,+0.0003] | -0.0165[-0.0327,+0.0000] |
| 16 | -0.0137[-0.0213,-0.0057] | -0.0218[-0.0341,-0.0089] |
| 24 | -0.0176[-0.0249,-0.0102] | -0.0242[-0.0340,-0.0143] |

**First separation vs s3 (belief(t) paired CI excl 0):** a3-blind turn 8 (sign -); a4-blind turn 3 (sign -).

**a4 calibration under arena-v2 (per turn, mean chosen-probe p_hat vs realized answer rate -- roughly right by construction now, since the answer model IS the pmodel/grid):**

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean chosen p_hat | 0.958 | 0.960 | 0.962 | 0.979 | 0.885 | 0.957 | 0.939 | 0.955 | 0.952 | 0.925 | 0.962 | 0.913 | 0.920 | 0.878 | 0.961 | 0.956 | 0.951 | 0.928 | 0.958 | 0.940 | 0.914 | 0.968 | 0.925 | 0.859 |
| realized answer rate | 1.000 | 1.000 | 0.990 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 | 0.963 | 1.000 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.997 | 1.000 | 0.993 | 0.997 | 1.000 | 0.997 | 1.000 |

Mean p_hat - answer-rate gap = -0.059 (corr -0.09) -- vs arena-v1's +0.720 gap: the surrogate is now well-aligned in level because the environment answers by the SAME measured model the agent ranks with.

**NDCG@10(t) curves (t=1..24):**

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 popular-item (v2) | 0.213 | 0.218 | 0.223 | 0.225 | 0.226 | 0.229 | 0.232 | 0.234 | 0.236 | 0.236 | 0.238 | 0.241 | 0.242 | 0.242 | 0.243 | 0.245 | 0.246 | 0.246 | 0.247 | 0.248 | 0.248 | 0.248 | 0.249 | 0.248 |
| a3-blind | 0.213 | 0.212 | 0.215 | 0.213 | 0.213 | 0.221 | 0.224 | 0.220 | 0.219 | 0.221 | 0.223 | 0.222 | 0.224 | 0.222 | 0.221 | 0.222 | 0.222 | 0.221 | 0.222 | 0.223 | 0.220 | 0.222 | 0.223 | 0.226 |
| a4-blind | 0.199 | 0.199 | 0.202 | 0.207 | 0.217 | 0.214 | 0.215 | 0.215 | 0.213 | 0.215 | 0.212 | 0.214 | 0.213 | 0.211 | 0.213 | 0.215 | 0.212 | 0.213 | 0.217 | 0.220 | 0.219 | 0.223 | 0.223 | 0.222 |
| a3-table (PRIV) | 0.199 | 0.199 | 0.199 | 0.199 | 0.208 | 0.219 | 0.215 | 0.215 | 0.215 | 0.217 | 0.217 | 0.217 | 0.216 | 0.215 | 0.216 | 0.223 | 0.222 | 0.221 | 0.222 | 0.223 | 0.221 | 0.222 | 0.223 | 0.225 |
| u1 clairvoyant (PRIV) | 0.511 | 0.577 | 0.604 | 0.616 | 0.622 | 0.624 | 0.625 | 0.626 | 0.627 | 0.626 | 0.626 | 0.625 | 0.624 | 0.621 | 0.618 | 0.616 | 0.615 | 0.610 | 0.608 | 0.604 | 0.598 | 0.592 | 0.587 | 0.581 |

**VERDICT:** BRANCH B (arena-v2) -- prize EXISTS, DISCOVERY still the bottleneck: the clairvoyant ceiling beats the rebuilt s3 static, but blind probing cannot capture it. (a3-blind beats s3=False, a4-blind beats s3=False, u1 PRIV beats s3=True.) Read against the arena-v1 verdict (Branch B, blind loses): the adaptivity verdict is reported under BOTH pre-registered answerability models.

**ASSUMPTIONS / judgment calls (arena-v2):**
1. Grid coverage/fallback: an item probe is answerable if the user rated it in the known half, OR the gate+main-study LLM grids (union; YES if ANY grid judged YES) say yes; for bank items NOT in that user's grid (mean grid coverage 4.9/160 items/user), fall back to the fitted pmodel with the user's TRUE known-half genre_match, thresholded at p_hat >= BASE_RATE=0.732 (the calibrated MAIN-STUDY grid base answer-rate). The answering rule is environment-side, so true-profile features are legal here; the AGENT arms (a3/a4 blind) never see it.
2. EASE rebuild: bank-restricted EASE (cross-check A recipe), universe = top-4000 popular UNION the 160 bank items = 4000 items, L2 lambda=500, binary item-item weights, item-mean-centered prediction. Trained on the KNOWN-portion ratings of the full ML-25M population with ALL study users' seed-123 held-out interactions dropped (never trains on recommendation targets). Predicts each bank item for each study user from their known-half context. Cached to `.cache/instrument2/arena2_ease_bankpred.npz` (rebuilt on bank/uid mismatch).
3. Value-channel noise: answerable-but-unrated value = EASE predicted star + N(0,0.7^2) stars, seeded DETERMINISTICALLY per (user,item) via rng((u*2654435761+j) mod 2^32), clipped [0.5,5], then centered by the user's known-half mean. Labelled LLM/CF-predicted (sensitivity-only convention). Rated items keep their REAL centered rating with no noise.
4. Native recall channel: an answerable item contributes a native (full-factor) recall token iff its star (real, or noised-predicted) >= 4 -- identical to the v1 rule for rated-liked items; the probe is system-selected (not open recall).
5. s3 greedy REBUILT under arena-v2 over the same top-60-coverage candidate pool; s1 concepts and the coverage prior / granularity g are unchanged (structural coverage) for comparability with v1.
6. All harness conventions inherited UNMODIFIED: refusal = no-op turn (belief unchanged), user retained, all 298 users in every mean; paired per-user bootstrap BOOT=5000 seed=0; u1/a3-table use the (now v2) TRUE answerability table (privileged, labelled); a3/a4 blind arms condition only on answers/refusals; co-known cosine from trU minus study users (leak=0 asserted).


## A5 K-map prober

Date 2026-07-08. Script `scripts/i25_phase4_a5.py` (imports/reuses `i25_phase4_fair.py`, `i25_phase4_a3.py`, `i25_phase4_a4.py` UNMODIFIED, and the ONLINE-INFERENCE `Kmap` class from `scripts/kmap_validate.py`). NO LLM calls; deterministic; local compute.

**Reproduction gate (existing harness, before adding anything):** s1 any@24=0.2103 (t 0.2103), s3 any@24=0.2786 (t 0.2786), a3-blind any@24=0.2412 (t 0.2412) -> reproduced=True. (a4-blind any@24=0.2431 for the in-loop comparison.)

**Idea.** The scaffold is a3/a4's blind item prober; the BELIEF is the validated LEARNED K-map `P(u knows j)=sigmoid(alpha_u + b_j + k_u.e_j)` (kmap_build/kmap_validate). Offline in this STRUCTURAL arena the per-user term k_u.e_j is the ONLY thing that lifts held-out AUC over popularity (t8 full 0.669 vs pop 0.623). A5 asks whether that offline superiority converts to a blind IN-LOOP win. Each user starts at the prior (k=0, alpha=0 -> P=sigmoid(b_j+beta0)); after every turn (alpha_u,k_u) are re-inferred by the MAP-Newton online inference from ALL observed events (answered=1/refused=0 for the probed dense id -- the probe outcome IS the structural label). Selection = argmax_{unasked} P_kmap(j)*V(j), V=study-cohort coverage prior (IDENTICAL to a3/a4).

**Calibration (level only, as offline).** b_j is the population rated-ever logit; the STRUCTURAL label is rated-in-the-known-half (lower base rate). We fit ONE global logit shift beta0 (1-param Platt, slope=1) on a deterministic TRAIN HALF (seed 0) so that the t=0 population prediction mean_j sigmoid(b_j+beta0) equals the empirical structural base rate over (train users x 160 bank items). Fitted: base rate = 0.1107 -> beta0 = -1.8510 (calibrated mean P@t0 = 0.1107). AUC is invariant to this monotone shift (map-health uses the uncalibrated logit); in the deep low-probability regime P*V argmax is near beta0-invariant, so beta0 sets the reported P LEVEL far more than the picks.

**Variants.** a5-blind (above, deployable). a5-ucb: one-standard-error optimism on the knowledge logit from the Laplace posterior of (alpha_u,k_u) -- H^{-1} with H the Newton Hessian reconstructed AT the returned MAP (the posterior variance IS cheaply available; the iterative inference is not duplicated); logit_ucb(j)=(alpha+b_j+k.e_j)+sqrt(a_j^T H^{-1} a_j), a_j=[1,e_j]. Principled exploration, no hand rules.

| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |
|---|---|---|---|---|---|
| s3 popular-item (opponent) | 0.2303/0.2683 | 0.2602/0.3035 | 0.2786/0.3236 | 3.4 (14%) | -- |
| a3-blind (co-known) | 0.1976/0.2166 | 0.2195/0.2675 | 0.2412/0.2932 | 4.4 (18%) | -- |
| a4-blind (pmodel) | 0.1986/0.2171 | 0.2233/0.2723 | 0.2431/0.2890 | 4.4 (18%) | -- |
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | 22.9 (95%) | -- |
| a5-blind (K-map) | 0.1981/0.2281 | 0.2233/0.2624 | 0.2407/0.2858 | 4.8 (20%) | -0.0379[-0.0504,-0.0251] |
| a5-ucb (K-map + Laplace 1se) | 0.2021/0.2347 | 0.2292/0.2711 | 0.2470/0.2914 | 4.7 (20%) | -0.0317[-0.0433,-0.0196] |

**Contrast a5-blind vs s3 (THE contrast), vs a3-blind, vs a4-blind, per budget:**

| budget T | a5-blind vs s3 [CI] | a5-blind vs a3-blind [CI] | a5-blind vs a4-blind [CI] |
|---|---|---|---|
| 8 | -0.0322[-0.0461,-0.0175] | +0.0005[-0.0079,+0.0086] | -0.0005[-0.0075,+0.0066] |
| 16 | -0.0369[-0.0501,-0.0234] | +0.0038[-0.0034,+0.0113] | -0.0000[-0.0068,+0.0070] |
| 24 | -0.0379[-0.0504,-0.0251] | -0.0005[-0.0069,+0.0058] | -0.0023[-0.0084,+0.0037] |

**Mechanism metric -- hit rate (mean answered turns / 24):** s3 = 3.4 (~14%); a3-blind = 4.4; a4-blind = 4.4; a5-blind = 4.8 (~20%); a5-ucb = 4.7; a3-table ceiling = 12.9. Hit rose vs s3? **True**. Hit rose vs a4-blind? **True**. a5-blind BEATS s3 at T=24 (CI excl 0)? **False**.

**First separation (a5-blind belief(t) vs s3 belief(t), CI excl 0):** turn 2 (sign -).

**Calibration in the loop (mean chosen-probe P_kmap vs realized answer rate per turn):**

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean chosen P_kmap | 0.306 | 0.189 | 0.130 | 0.096 | 0.076 | 0.066 | 0.056 | 0.057 | 0.051 | 0.053 | 0.056 | 0.055 | 0.051 | 0.056 | 0.058 | 0.055 | 0.054 | 0.054 | 0.050 | 0.050 | 0.048 | 0.048 | 0.047 | 0.044 |
| realized answer rate | 0.275 | 0.258 | 0.181 | 0.188 | 0.178 | 0.218 | 0.205 | 0.171 | 0.201 | 0.221 | 0.188 | 0.195 | 0.225 | 0.195 | 0.225 | 0.205 | 0.188 | 0.201 | 0.164 | 0.181 | 0.218 | 0.185 | 0.134 | 0.208 |

Mean P_kmap - answer-rate gap = -0.125 (corr +0.63) -- the calibrated K-map probability is PESSIMISTIC (under-predicts) against the in-loop realized answer rate.

**Map-health (does the policy starve the map?): per-user held-out STRUCTURAL AUC @t=8 using a5-blind's OWN probe order, vs the offline (s3-order) t8 number.**

- In-loop a5-blind AUC@t8 = **0.629** (n=292) vs offline **0.669** -> delta -0.040. a5-ucb AUC@t8 = 0.642 (n=292).
- Reading: in-loop map is DEGRADED vs offline -- the policy concentrates probes on high-P items and starves the K-map of the diverse evidence its offline s3-order run had.

**NDCG@10(t) curves (t=1..24):**

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 popular-item | 0.172 | 0.197 | 0.216 | 0.230 | 0.244 | 0.254 | 0.262 | 0.268 | 0.274 | 0.280 | 0.285 | 0.289 | 0.293 | 0.296 | 0.299 | 0.304 | 0.308 | 0.310 | 0.312 | 0.314 | 0.316 | 0.319 | 0.322 | 0.324 |
| a3-blind | 0.172 | 0.176 | 0.186 | 0.197 | 0.207 | 0.214 | 0.213 | 0.217 | 0.218 | 0.223 | 0.227 | 0.235 | 0.247 | 0.253 | 0.260 | 0.267 | 0.274 | 0.278 | 0.283 | 0.285 | 0.287 | 0.289 | 0.290 | 0.293 |
| a4-blind | 0.165 | 0.178 | 0.194 | 0.199 | 0.210 | 0.211 | 0.214 | 0.217 | 0.224 | 0.231 | 0.236 | 0.243 | 0.253 | 0.259 | 0.266 | 0.272 | 0.275 | 0.278 | 0.280 | 0.281 | 0.284 | 0.286 | 0.287 | 0.289 |
| a5-blind | 0.165 | 0.178 | 0.183 | 0.195 | 0.200 | 0.213 | 0.222 | 0.228 | 0.234 | 0.238 | 0.244 | 0.244 | 0.252 | 0.255 | 0.258 | 0.262 | 0.264 | 0.270 | 0.271 | 0.274 | 0.275 | 0.282 | 0.283 | 0.286 |
| a5-ucb | 0.165 | 0.178 | 0.190 | 0.197 | 0.207 | 0.217 | 0.228 | 0.235 | 0.241 | 0.245 | 0.250 | 0.256 | 0.257 | 0.263 | 0.267 | 0.271 | 0.273 | 0.274 | 0.282 | 0.283 | 0.286 | 0.283 | 0.287 | 0.291 |

**VERDICT:** A5-blind RAISES the hit rate (4.8 vs a4-blind 4.4, s3 3.4/24) but does NOT beat s3 on NDCG at T=24 -- the K-map's per-user knowledge tilt finds more answerable items than the fixed list, yet the extra answers do not add enough ranking value to overturn the popular-item static. Branch B stands.

**ASSUMPTIONS / judgment calls (a5):**
1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3/a4 draw from; item probes only; V(j)=study-cohort coverage prior (pop_rate) -- identical to a3/a4.
2. Belief = the LEARNED K-map (kmap_emb.npz/kmap_intercepts.npz), online inference = kmap_validate.Kmap.infer (MAP-Newton, 12 it, priors tau(k)=1.0, tau_a(alpha)=2.0); imported UNMODIFIED, not duplicated. t=0 => k=alpha=0 => P=sigmoid(b_j+beta0) (popularity-only), so turn 1 is fully model-driven and deployable.
3. Event label = the probe outcome (answered=1 iff the user rated the dense item id in the known half, i.e. the STRUCTURAL arena the K-map's four offline gates were passed in; refused=0). No privileged features enter the policy.
4. Level calibration beta0: 1-parameter logit shift (Platt slope=1) fit on a deterministic TRAIN HALF (seed 0) so mean_j sigmoid(b_j+beta0)=structural base rate 0.1107 over (train users x bank); applied to all users/turns. AUC-invariant (map-health uncalibrated). In the low-P regime P*V argmax is near beta0-invariant -> calibration sets the reported P LEVEL, barely the picks.
5. Selection = argmax_{unasked} P_kmap(j)*V(j); tie-break higher V then lowest global cid (same deterministic rule as a4).
6. a5-ucb: Laplace posterior cov = H^{-1}, H = Newton Hessian reconstructed at the returned MAP from KM.emb_of/b_of (the posterior variance is a by-product of the same infer; the iterative fitting is not re-run). logit_ucb=(alpha+b_j+k.e_j)+sqrt(a_j^T H^{-1} a_j), 1-sigma optimism; empty history -> prior covariance diag(tau_a^2, tau^2 I).
7. Map-health = held-out structural AUC at t=8 over bank items NOT among the arm's first 8 probes (a5-blind's OWN order), label=rated-in-known-half, per-user AUC needs both classes; compared to the offline s3-order 0.669 to detect probe-choice starvation.
8. Refusal = no-op turn (belief unchanged for the fold; the K-map still records the refusal event), user retained; all 298 users in every mean (fair, inherited). Bootstrap paired per-user BOOT=5000 seed=0; all selectors deterministic.


## A6 tie-by-construction prober

Date 2026-07-08. Script `scripts/i25_phase4_a6.py` (imports/reuses `i25_phase4_fair.py`, `i25_phase4_a3.py`, `i25_phase4_a5.py` [calibrate_beta0] UNMODIFIED, and the ONLINE-INFERENCE `Kmap` class from `scripts/kmap_validate.py`). NO LLM calls; deterministic; local compute.

**Reproduction gate (existing harness, before adding anything):** s1 any@24=0.2103 (t 0.2103), s3 any@24=0.2786 (t 0.2786), a3-blind any@24=0.2412 (t 0.2412) -> reproduced=True.

**The confound A6 removes.** a2/a3/a4/a5 ranked probes by P(answerable)xCOVERAGE, but the best static s3 was built by greedy NDCG-VALUE ordering -- so those arms deviated from s3 at t=0 for VALUE-MODEL reasons (not evidence), violating the program's tie-by-construction rule (warm-start from the best heuristic so learning can only ADD). Their losses to s3 are confounded. A6 is a WARM-STARTED s3: it starts from s3's exact schedule and tilts it ONLY by an evidence-driven answerability likelihood ratio.

**Value V(j).** The s3 greedy ordering itself, realized as the monotone rank proxy V(j)=1/rank_s3(j) (FA.build_greedy does not return greedy marginal contributions, so the rank proxy is used -- see ASSUMPTIONS). s3's 24 scheduled items take ranks 1..24 in schedule order; remaining bank items take ranks 25.. by coverage (pop_rate) descending. So argmax V over unasked = s3's next item exactly.

**Belief + selection.** BELIEF = the validated LEARNED K-map `P(u knows j)=sigmoid(alpha_u+b_j+k_u.e_j)`, online MAP-Newton inference from all observed events, IDENTICAL to a5; level calibration beta0 reused from a5 (base rate 0.1107 -> beta0 -1.8510, mean P@t0 0.1107). SELECTION each turn = argmax over UNASKED bank items of V(j) x LR(j), LR(j)=P_kmap(j|events)/P_kmap(j|no events). At t=0, alpha=k=0 => LR=1 for ALL j => the pick is EXACTLY s3's next item, so s3 is the policy FLOOR at t=0 and every deviation is a pure evidence-driven swap. **Tie-by-construction floor check (all users' turn-1 pick == s3[0]): True.**

**Variants.** a6 (argmax V*LR, deployable). a6-margin (hysteresis: swap off the s3-base order only if LR(argmax) > 1.2, else take the next s3-order item). a6-table (PRIVILEGED, labelled: same V*LR policy but LR from the TRUE-answerability posterior -- LR=1 if the user rated j, ~0 else -- the tie-by-construction ceiling of the policy class; recovers most of a3-table's +0.060 iff the V*LR form is not the bottleneck).

| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |
|---|---|---|---|---|---|
| s3 popular-item (opponent) | 0.2303/0.2683 | 0.2602/0.3035 | 0.2786/0.3236 | 3.4 (14%) | -- |
| a3-blind (co-known, prior arm) | 0.1976/0.2166 | 0.2195/0.2675 | 0.2412/0.2932 | 4.4 (18%) | -- |
| s1 concepts | 0.2120/0.2099 | 0.2106/0.2092 | 0.2103/0.2097 | 22.9 (95%) | -- |
| a6 (V=s3-order x K-map LR) | 0.2280/0.2661 | 0.2582/0.3012 | 0.2765/0.3162 | 4.1 (17%) | -0.0021[-0.0056,+0.0015] |
| a6-margin (swap iff LR>1.2) | 0.2295/0.2679 | 0.2596/0.3022 | 0.2785/0.3240 | 3.4 (14%) | -0.0002[-0.0011,+0.0008] |
| a6-table (PRIV, true-table LR = class ceiling) | 0.3235/0.3422 | 0.3410/0.3682 | 0.3514/0.3754 | 12.9 (54%) | +0.0727[+0.0596,+0.0871] |

**THE contrast -- a6 vs s3, and the PRIVILEGED ceiling a6-table vs s3, per budget (can now differ from s3 ONLY through evidence-driven swaps):**

| budget T | a6 vs s3 [CI] | a6-margin vs s3 [CI] | a6-table (PRIV) vs s3 [CI] |
|---|---|---|---|
| 8 | -0.0023[-0.0050,+0.0001] | -0.0008[-0.0016,-0.0001] | +0.0931[+0.0786,+0.1082] |
| 16 | -0.0019[-0.0049,+0.0010] | -0.0006[-0.0015,+0.0004] | +0.0808[+0.0673,+0.0953] |
| 24 | -0.0021[-0.0056,+0.0015] | -0.0002[-0.0011,+0.0008] | +0.0727[+0.0596,+0.0871] |

**SWAP FORENSICS (a6 arm -- a swap = the argmax V*LR pick differs from the pure-V s3-order continuation among unasked items).**

- Deviations from s3's order per user: mean **13.30**, median 14, max 21; users with >=1 swap **298/298**; 3962 swap events total. (a6-margin, LR>1.2: mean 1.04 deviations/user.)
- Realized answer rate of swapped-IN items = **0.207** vs displaced (s3-order) items = **0.102** (the K-map LR tilts toward items the user is MORE likely to answer).
- NDCG effect of swaps (per-user paired delta a6-s3): users with >=1 swap (n=298) any@24 **-0.0021**[-0.0056,+0.0015] (end@24 -0.0075); users with 0 swaps (n=0) any@24 +nan (identical to s3 by construction -- 0 deviations => same plan).

**Mechanism metric -- hit rate (mean answered turns / 24):** s3 = 3.4 (~14%); a3-blind = 4.4; a6 = 4.1 (~17%); a6-margin = 3.4; a6-table (PRIV) = 12.9; a3-table ceiling = 12.9. a6 BEATS s3 at any budget (CI excl 0)? **False**. a6-table (PRIV) beats s3 (policy CLASS viable)? **True**.

**First separation (a6 belief(t) vs s3 belief(t), CI excl 0):** turn 4 (sign -).

**NDCG@10(t) curves (t=1..24):**

| arm | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 | t24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3 popular-item | 0.172 | 0.197 | 0.216 | 0.230 | 0.244 | 0.254 | 0.262 | 0.268 | 0.274 | 0.280 | 0.285 | 0.289 | 0.293 | 0.296 | 0.299 | 0.304 | 0.308 | 0.310 | 0.312 | 0.314 | 0.316 | 0.319 | 0.322 | 0.324 |
| a3-blind | 0.172 | 0.176 | 0.186 | 0.197 | 0.207 | 0.214 | 0.213 | 0.217 | 0.218 | 0.223 | 0.227 | 0.235 | 0.247 | 0.253 | 0.260 | 0.267 | 0.274 | 0.278 | 0.283 | 0.285 | 0.287 | 0.289 | 0.290 | 0.293 |
| a6 | 0.172 | 0.197 | 0.217 | 0.226 | 0.242 | 0.247 | 0.257 | 0.266 | 0.273 | 0.278 | 0.285 | 0.286 | 0.292 | 0.294 | 0.299 | 0.301 | 0.307 | 0.310 | 0.314 | 0.313 | 0.315 | 0.315 | 0.316 | 0.316 |
| a6-margin | 0.172 | 0.197 | 0.216 | 0.229 | 0.243 | 0.252 | 0.260 | 0.268 | 0.274 | 0.280 | 0.284 | 0.288 | 0.292 | 0.295 | 0.302 | 0.302 | 0.308 | 0.312 | 0.313 | 0.315 | 0.316 | 0.320 | 0.323 | 0.324 |
| a6-table (PRIV) | 0.271 | 0.299 | 0.325 | 0.333 | 0.337 | 0.336 | 0.343 | 0.342 | 0.349 | 0.352 | 0.355 | 0.354 | 0.360 | 0.363 | 0.367 | 0.368 | 0.370 | 0.371 | 0.372 | 0.372 | 0.370 | 0.373 | 0.374 | 0.375 |

**VERDICT:** A6 TIES s3 within noise: warm-starting from s3 and tilting only by evidence neither adds nor destroys ranking value -- the confound in the prior arms explained their apparent losses, but blind answerability adaptivity still does not beat the popular-item static. a6-table PRIV vs s3 @T24 +0.0727[+0.0596,+0.0871] -> policy CLASS viable (recover-a3-table check). Branch B stands.

**ASSUMPTIONS / judgment calls (a6):**
1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3/a4/a5 draw from; item probes only.
2. V(j) REALIZATION = the s3 greedy ORDER as a monotone rank proxy V(j)=1/rank_s3(j). FA.build_greedy returns only the schedule (not per-position greedy marginal contributions), so the rank proxy is used; it is monotone in the s3 order, which is all the tie-by-construction argument requires (argmax V over unasked == s3's next item at LR=1). s3's 24 items get ranks 1..24 in schedule order; the remaining bank items get ranks 25.. by coverage descending (tie-break dense id).
3. Belief = the LEARNED K-map (kmap_emb.npz/kmap_intercepts.npz), online inference = kmap_validate.Kmap.infer (MAP-Newton, 12 it, priors tau(k)=1.0, tau_a(alpha)=2.0); imported UNMODIFIED. Event label = probe outcome (answered=1 iff the user rated the dense id in the known half; refused=0) -- the STRUCTURAL arena. No privileged features enter the a6/a6-margin policy.
4. LR CALIBRATION = P_kmap(j|events)/P_kmap(j|no events) with P_kmap=sigmoid(alpha+b_j+k.e_j+beta0), beta0 the a5 level shift (reused via A5.calibrate_beta0; base rate 0.1107 -> beta0 -1.8510). At t=0 alpha=k=0 => numerator==denominator => LR=1 exactly for every j (verified: all turn-1 picks == s3[0]); beta0 does not affect the t=0 tie (cancels in the ratio there) and only tilts the ratio once events accrue.
5. MARGIN (a6-margin) = 1.2: swap off the s3-base order only if the argmax's LR exceeds 1.2 (hysteresis against posterior noise); otherwise take the next s3-order item.
6. a6-table (PRIVILEGED, labelled) = same V*LR selection but LR = the certain-knowledge posterior (1.0 if the user rated j, 1e-6 else), so argmax V*LR = highest-V ANSWERABLE item each turn -- the tie-by-construction ceiling of the policy class (mirrors a3-table's restrict-to-answerable rule). Not deployable.
7. SWAP definition = actual pick != j_base, where j_base = the max-V unasked item (the pure s3-order continuation given what is already asked); swapped-IN = actual pick, displaced = j_base. Deterministic tie-breaks: max score, then higher coverage V, then lower dense id.
8. Refusal = no-op turn (belief unchanged for the fold; the K-map still records the refusal event), user retained; all 298 users in every mean (fair, inherited). Bootstrap paired per-user BOOT=5000 seed=0; all selectors deterministic.

