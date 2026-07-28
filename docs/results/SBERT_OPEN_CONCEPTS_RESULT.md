# RESULT — open-vocabulary concept directions via one SBERT adapter (R3 illustration)

> Ran 2026-07-28. `src/instrument/sbert_open_concepts.py` →
> `experiments/battery/sbert_open_concepts_fit{concepts,items}.json`.
> Frozen `all-MiniLM-L6-v2` (384-d) → frozen i25 latent (200-d), one closed-form ridge adapter.

## Configuration of record: fit on CONCEPT DIRECTIONS, hold out 20% of tags

`W = Dᵀ S (SᵀS + βI)⁻¹`, S = SBERT(tag name), D = `d_c` (whitened member centroids — the directions the
concept channel actually folds). Fit on 825 tags, evaluate on **206 never seen**, predicting each held-out
concept's direction **from its name alone**:

| | held-out (206 unseen tags) |
|---|---|
| cosine to true `d_c` | **0.638** (in-fit reference 0.815) |
| top-10 item overlap | **0.167** vs **0.0005** random = **345×** |
| paraphrase overlap@10 (10 pairs) | **0.25** |

## What is solid, and what is not

**Solid.** Generalisation to unseen concepts is real and large against chance (345×). A concept the adapter
was never fitted on still lands in roughly the right region of the catalogue, from its name alone, with no
per-concept training and no retraining — which is the R3 property we wanted to illustrate. Several
free-text probes (phrases that are not genome tags at all, so they have no `d_c` to fall back on) are
qualitatively convincing: *courtroom drama* → Indictment: The McMartin Trial, Primal Fear, ...And Justice
for All, The Verdict; *coming of age in the suburbs* → The Way Way Back, The Edge of Seventeen, Valley
Girl; *movies about grief* → Rabbit Hole, Mean Creek, The Woodsman; *spy during the cold war* → Funeral in
Berlin, Where Eagles Dare.

**Not solid.** The absolute numbers are modest — 0.167 overlap means a held-out concept shares under two
of ten items with its true direction — and **paraphrase invariance at 0.25 is weak**. It should NOT be
presented as a control that passes. Some probes are only adjacent rather than right: *dinosaurs* lands on
B-movie creature features (Food of the Gods, Land That Time Forgot) rather than Jurassic Park; *heist gone
wrong* returns generic thrillers.

## Two hypotheses tested and REFUTED (recorded so they are not retried)

1. **"1,031 pairs is too few — fit on items instead."** Fitting on 13,815 relevance-weighted item vectors
   made everything **worse**: cos 0.503 (vs 0.638), overlap 0.086 (vs 0.167), paraphrase 0.12 (vs 0.25).
   More data, worse result.
2. **"That's a target mismatch — an items fit lives in `d_raw`'s geometry, so score it against `d_raw`."**
   Also refuted: the items fit scores **worse** against the unwhitened centroid than the whitened one
   (cos 0.373 vs 0.503; overlap 0.055 vs 0.086).

The surviving explanation is the one the previous chapter already reached by a different route: **text
underdetermines collaborative taste.** Its item-level adapter alignment was 0.43 and it said so plainly.
Fitting on the concept directions themselves is the best configuration precisely because it targets the
object we want, not because more data or better-matched geometry would rescue it.

## How this should be used in the chapter

As a **half-page qualitative illustration** that the interface accepts an arbitrary continuous direction —
supporting the R3 claim that the channel set is open without bolt-ons and that a continuous elicitation
policy could drive it. Report the probes and the 345×-over-random generalisation; state the paraphrase
number as a limitation. Do **not** build a claim on it, and do **not** present it as a text-to-latent
mapping contribution — that ground is occupied (Balog et al. SIGIR 2021 §5.2 label-free retrieval-centroid;
Göpfert 2022 / Bıyık 2023 per-concept supervised probes).


## PROBE CONTAMINATION CHECK (2026-07-28) — several probes were genome tags

The author asked what the nearest genome tag to each probe was. Several probes ARE tags verbatim:
`courtroom drama` = tag 263, `dinosaurs` = 304, `coming of age` = 235, `cold war` = 226,
`heist` = 498, `post apocalyptic` = 802. The probe adapter was refitted on ALL tags, so those phrases
were in the fitting set — the results were recall of a fitted direction, not open-vocabulary work. A
sentence asserting the opposite had already been written into the chapter and was pulled.

`src/instrument/probe_nearest_tag.py` measures this: nearest tags by cosine, the top-10 overlap between
the phrase direction and its nearest tag CURATED direction, and a leave-the-neighbour-out refit dropping
every tag within cos 0.75.

**The four that survive** (nearest tag < 0.75, nothing dropped, so no contamination):

| phrase | nearest concept | cos | overlap w/ that concept | top films |
|---|---|---|---|---|
| movies about grief | *feel good movie* | 0.54 | 0.0 | Rachel Rachel; Mean Creek; The Woodsman |
| slow burn character study | *character study* | 0.52 | 0.0 | Spring; Let Me In; The Wailing; Let the Right One In |
| spy during the cold war | *cold war* | 0.67 | 0.1 | Where Eagles Dare; Funeral in Berlin; Sink the Bismarck |
| coming of age in the suburbs | *coming of age* | 0.69 | 0.4 | The Way Way Back; Manic; Dogfight; Edge of Seventeen |

**Excluded**: courtroom drama and dinosaurs (exact tags, cos 1.0); heist gone wrong (0.79), outer space
aliens invading earth (0.77), artificial intelligence robots (0.83), post apocalyptic survival (0.86).

The strongest single line: the nearest concept in the whole vocabulary to *movies about grief* is
*feel good movie* — tonally opposite — sharing NONE of its top ten with that concept curated direction,
yet the returned films are right. The phrase is placed on its own terms, not retrieved.
