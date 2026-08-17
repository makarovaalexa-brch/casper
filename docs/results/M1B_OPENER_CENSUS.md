# M1b — the REALISTIC opener census (what people actually lead with)

**Design sheet:** `docs/design/DS_M1_OPENER_FAMILIES.md` · **code:** `src/instrument/m1b_census.py`,
`src/instrument/m1b_map.py` · **artifacts:** `experiments/battery/m1b_census.json`,
`m1b_mapping.json`. Ran 2026-08-17 on the banked corpus only — **no new adjudication, no LLM calls,
no Reddit API**. Scope: all 4,431 ruler threads (each ≥5 in-catalogue GT films + request text),
typed slots extracted previously.

## ★ THE HEADLINE: 63% of real requests open by NAMING A FILM

| opener (highest-priority slot present) | threads | share |
|---|---|---|
| **`anchors_pos` — names a film** | **2,793** | **63.03%** |
| `genres_pos` — names a genre | 651 | 14.69% |
| `constraints` | 230 | 5.19% |
| `tone` | 146 | 3.29% |
| `era` | 86 | 1.94% |
| *(no priority slot present)* | 525 | 11.85% |

Marginal presence (a slot appears at all, not necessarily as the opener), which is the assumption-free
view — the priority ordering above is a declared modelling choice, this is not:

| slot | threads | share |
|---|---|---|
| `other` | 3,518 | 79.40% |
| **`anchors_pos`** | **2,793** | **63.03%** |
| `genres_pos` | 1,763 | 39.79% |
| `constraints` | 1,360 | 30.69% |
| `tone` | 946 | 21.35% |
| `anchors_neg` | 730 | 16.47% |
| `context` | 612 | 13.81% |
| `era` | 433 | 9.77% |
| `genres_neg` | 184 | 4.15% |

**What this means for the chapter, stated carefully.** The single most common thing a real user
volunteers is *a film they like* — which is **evidence the recommender folds, not a signature a
question elicits**. This is direct empirical support for the mechanism the chapter has already
measured: volunteered evidence pays through the fold (+0.0206 full / +0.0353 tail at k=8, R). It is
also a scope statement for the opener families: an anchor film is too granular to key a cohort on
(5,868 distinct anchor values over 2,793 threads), so anchors enter as evidence and the *routable*
openers are genre, tone and era.

## Step 2 — mapping opener values onto ML attribute signatures

Computed over each slot's census top-50 value list (stated limitation: that is the head of the tail,
not the whole tail). Unmapped values are reported, never dropped (HARD RULE 1).

| slot | signature space | mapped | share | leading targets |
|---|---|---|---|---|
| `genres_pos` | MovieLens genre | 924 / 972 | **95.1%** | Horror 254 · Thriller 146 · Comedy 121 · Sci-Fi 82 · Action 73 · Romance 67 · Drama 56 · Documentary 40 |
| `tone` | genome concept (1,128 tags) | 113 / 171 | **66.1%** | feel-good 30 · fun 14 · sad 7 · scary 7 · emotional 6 · dark 6 · disturbing 6 |
| `era` | release decade | 73 / 104 | **70.2%** | 1980s 28 · 1990s 17 · 2000s 14 · 1970s 8 |

**The unmapped residual is a measured price, and it is the interesting part.** What our attribute
vocabulary cannot express: `coming of age` (14), `Christmas` (9), `supernatural` (8), `dystopian`
(6), `suspense` (6) on the genre side; **`wholesome` (8), `uplifting` (6), `cozy` (4)** on the tone
side — note these are exactly the affect words with no genome tag; and relative eras (`newer`,
`recent`, `from the last decade`) which have no fixed decade at all.

## ★ The finding that reshapes the realistic family

**Real openers are dominated by Horror.** As a genre opener, Horror is 254 of 924 mapped occurrences
(27.5%) — against Horror being **51 of 10,000 validation users** by our fold-in genre rule
(`groupgreedy_genre15`: Drama 5,481 · Action 1,582 · Comedy 1,475 … Horror 51). **What people ask
for and what people have watched are close to inverted**: our ML population is Drama-heavy, real
requests are Horror/Thriller-heavy.

This is a genuine result and it must travel with every realistic-family number. Two consequences:

1. **The Horror cohort cannot meet the ≥1,200-user gate** on the ML side (51 users). The realistic
   family's viable cohorts are therefore the *intersection* of "people ask for it" and "our
   population has it" — Comedy (1,475), Action (1,582), Thriller (451), Sci-Fi (161) are the
   candidates, and only Comedy and Action clear 1,200.
2. The mismatch is itself reportable: **the openers people actually use address a population our
   corpus under-represents**, which is a scope condition on any ML-only elicitation result and a
   motivation for the mixed benchmark being weighted by observed frequencies rather than by ML
   prevalence.

## Gates and what carries forward to M2

- Cohorts must be ≥1,200 validation users. On genre, **Comedy and Action clear; Horror and Thriller
  do not**, despite being the most-requested. Reported, not merged.
- Tone cohorts are keyed on genome concepts (≥k rated members rule), where the ML side is far less
  sparse than the genre side — tone is the realistic family's best hope for a cohort that is both
  frequently requested and adequately populated. To be measured in M1b step 3.
- Era cohorts by decade, ML-side populous, requested in only 1.94% of threads as opener (9.77%
  marginal).

## Provenance

4,431 rows from 40 slot files, 4,431 distinct thread ids, 4,166 flagged `is_request`. Priority order
`anchors_pos > genres_pos > tone > era > constraints` declared in the design sheet before running.
No TEST data touched; no recommender or policy was loaded by either script.
