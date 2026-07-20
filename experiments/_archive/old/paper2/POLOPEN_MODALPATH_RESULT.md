# Paper D — POLOPEN modal-path ablation: is the anytime-asker win BRANCHING or SEQUENCING? (2026-07-03)

**Review item D-B5.** The POLOPEN belief-only anytime no-repeat asker beats the best hand-fixed order at turns 2–7
(NOREPEAT_RESULT.md, +0.013 tail @t5 seed-avg). Open question: is that genuine PER-USER BRANCHING, or just a better
learned STATIC SEQUENCE? Test: extract each trained policy's MODAL path (most frequent 8-turn question-type sequence
across the 304 test users) and evaluate that sequence AS A STATIC ORDER on the same ruler.

Ruler: seed-avg {1,2,3,7,11} eval profile-splits, te[300:] (304 users), anytime NDCG@10 curve t1–8, graded geometric
answers, frozen V1 encoder. Harness: POLOPEN block of `scripts/paper2/continuous_actor.py` run via an eval-only patched
copy (adds `POLLOAD` = load checkpoint, skip training AND skip the checkpoint save; `POLSAVE` = non-clobbering save;
zero change to any eval code path). NOTE these curves use 5 eval seeds; NOREPEAT_RESULT used {1,2,3}, so absolute
values shift slightly (hand-fixed t5 here 0.3990/0.1784 vs 0.3932/0.1769 there) — all comparisons below are within-run.

## Checkpoint provenance (blocker found + resolved)
- The POLOPEN save path `polopen_<types>_wf1.0wt1.0.pt` does NOT include POLSEED → the three training-seed checkpoints
  of NOREPEAT_RESULT **overwrote each other**; only ONE survived (`.cache/polopen_gem_..._whatdoyoulike_wf1.0wt1.0.pt`,
  sha e02b78ce). Its greedy usage tree reproduces the **POLSEED=0** tree of NOREPEAT_RESULT exactly (align-100% opener,
  turn2 genre 67%/fav 21%, turn3 actor 48%) → the survivor is s0.
- POLSEED 1 and 2 were **retrained** with the exact repro config (NOBC=1 EP=0 CONTMODE=cont POLOPEN=1 NOREPEAT=1
  CURVEREW=1 HORIZON=8 ENT=0.02 POLEP=90 POLSEED=k, best-val selection) and saved to the session scratchpad (canonical
  cache untouched). Signatures match the NOREPEAT per-seed table (ps1: gem opener, t1 0.362/0.173=fixed; ps2: strong
  adaptive opener whatdoyoulike, t1 0.3875/0.1882 ≈ s2's .387/.189) → faithful stand-ins for the lost s1/s2.

## Modal paths + branching stats (greedy walk, TEST users, eval seed 1)
| seed | opener | distinct paths /304 users | modal path (most frequent full sequence) | modal share |
|---|---|---|---|---|
| s0 | align | 116 | align,genre,hate,actor,gem,director,avoidgenre,fav | 4.9% |
| s1 | gem | 39 | gem,whatdoyoulike,fav,director,actor,genre,hate,align | 43.8% |
| s2 | whatdoyoulike | 81 | whatdoyoulike,align,director,genre,fav,avoidgenre,gem,actor | 16.4% |

Per-turn fraction of users DEVIATING from the modal path's type at that turn (s0 / s1 / s2, %):
t1 0/0/0 · t2 33/0/57 · t3 87/39/34 · t4 63/31/62 · t5 49/35/63 · t6 65/33/82 · t7 88/22/79 · t8 88/8/63.
Fraction still exactly ON the modal prefix by t5: s0 8.6%, s1 48.4%, s2 25.7%. So the policies DO branch heavily
per-user (the opener is fixed-by-construction; branching starts at t2). The question is whether the branching *pays*.

## Headline: adaptive vs ITS OWN modal-static vs hand-fixed (FULL/TAIL NDCG@10)

s0 (locked checkpoint):
| turn | ADAPTIVE | MODAL-STATIC | hand-FIXED | adaptive−modal |
|---|---|---|---|---|
| 2 | .3862/.1742 | .3889/.1793 | .3804/.1674 | −.003/−.005 |
| 5 | .3994/.1885 | .3990/.1923 | .3990/.1784 | +.000/−.004 |
| 6 | .4025/.1926 | .4052/.1929 | .3992/.1781 | −.003/−.000 |
| 8 | .4066/.1955 | .4043/.1884 | .4078/.1966 | +.002/+.007 |
(s0's prefix-modal order align,genre,actor,gem,hate,director,avoidgenre,fav is even stronger: it beats the adaptive
policy at t2–t7, e.g. t4 .3987/.1925 vs adaptive .3951/.1826.)

s1 (retrained):
| turn | ADAPTIVE | MODAL-STATIC | hand-FIXED | adaptive−modal |
|---|---|---|---|---|
| 2 | .3903/.1941 | .3903/.1941 | .3804/.1674 | .000/.000 |
| 3 | .3969/.1909 | .3950/.1864 | .3924/.1707 | +.002/+.005 |
| 5 | .4037/.1918 | .4038/.1912 | .3990/.1784 | −.000/+.001 |
| 8 | .4079/.1963 | .4079/.1963 | .4078/.1966 | .000/.000 |

s2 (retrained, branchiest + strongest adaptive):
| turn | ADAPTIVE | MODAL-STATIC | hand-FIXED | adaptive−modal |
|---|---|---|---|---|
| 2 | .3951/.1944 | .3920/.1916 | .3804/.1674 | +.003/+.003 |
| 4 | .4032/.1986 | .4057/.1968 | .3964/.1781 | −.002/+.002 |
| 5 | .4049/.1961 | .4018/.1880 | .3990/.1784 | +.003/+.008 |
| 6 | .4069/.1958 | .4014/.1878 | .3992/.1781 | +.005/+.008 |
| 8 | .4079/.1964 | .4078/.1966 | .4078/.1966 | +.000/−.000 |
(vs s2's prefix-modal order the residual shrinks further: t5 +.001/+.005, t4 +.004/+.008, others ≤.002.)

Training-seed average @t5 (the KEY operating point):
- ADAPTIVE 0.4027 / 0.1921 · OWN-MODAL-STATIC 0.4015 / 0.1905 · HAND-FIXED 0.3990 / 0.1784.
- adaptive − hand-fixed: **+0.004 full / +0.014 tail** (the NOREPEAT claim, reproduced on 5 seeds).
- adaptive − own-modal-static: **+0.001 full / +0.002 tail** (≈ zero).

## VERDICT: the win is a better learned SEQUENCE, not per-user branching — REFRAME the claim
- ~90% of the POLOPEN edge over the hand-fixed order survives when the policy is FROZEN into its own modal STATIC
  order. What REINFORCE actually discovered is a better questionnaire (front-load the single most informative open
  question — align/gem/wdyl — then genre/person questions, negatives last), not user-conditional routing.
- The residual per-user branching value is ≤ +0.005 full / +0.008 tail at mid turns for the branchiest seed (s2) and
  ~0 for s1; sign flips across turns/seeds for s0. Not separable from noise at the 5-seed level (seed SD ≈ 0.005).
- The branching that does exist is real behaviour (49–63% of users off the modal type by t5) but nearly value-neutral:
  consistent with the permutation-invariant encoder + linear-Gaussian-ish world where optimal design is non-adaptive
  (memory: adaptivity needs nonlinearity). The t8 endpoint even shows the one clear per-user effect: adaptively
  CHOOSING WHICH 8-of-9 subset to ask (s0 ends wdyl for 42% of users) recovers the modal order's t8 subset loss.
- Paper D framing: report the learned-asker win as **"RL discovers a better static questionnaire (+ anytime
  front-loading); per-user branching contributes ≈0 (isolated here for the first time via the modal-path ablation)"**.
  This is an honest, defensible claim and mirrors the Paper-C finding (policy ≈ static; the learning is elsewhere).

## Repro
Patched copy: scratchpad `continuous_actor_evalpatch.py` (POLLOAD/POLSAVE only). Modal-static eval = same POLOPEN
block with `NRORDER=<modal sequence>` (the printed FIXED curve), `POLLOAD=<ckpt>`, EVALSEEDS=1,2,3,7,11.
Path dumps: TREEOUT jsons (scratchpad polopen_paths{,_ps1,_ps2}.json). Retrained stand-ins: scratchpad polopen_ps{1,2}.pt.
RECOMMEND: add POLSEED to the polopen save filename in continuous_actor.py to stop future clobbering.
