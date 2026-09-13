# What each published elicitation method was COMPARED AGAINST (2026-09-13)

Prompted by the supervisor's comments 8 and 20 on the ECIR draft: *"big criticism -- no one used
strong comparator"* and *"none of the published method choose greedy as a baseline, their baselines
are weak"*. Paper B asserts this in Related Work and never demonstrates it. This file is the evidence.

Companion to `baseline_reported_metrics.md`, which audits what each paper *reported*; this one audits
what each paper *compared against*. `[V]` = verified from the primary full text in this pass.

## The two that could have refuted us — both checked, both do not

**DPE — Parapar & Radlinski, WSDM 2021.** Has a baseline literally named `Greedy`, which our archive
had flagged as *"Greedy tries to maximize recommendation performance"*. Full text, §4.3 `[V]`:

> *"As a first baseline we use the Greedy method from [13]. In that work, the authors compare several
> algorithms for non-diversified preference elicitation... We use the simpler Greedy approach as a
> baseline: At each time step, it solicits the relevance of the item with **highest estimated relevance
> by a recommendation algorithm** that uses the previously elicited items, i.e., **the top item in the
> recommendation list**."*

So it asks about the item the model currently predicts the user will like most: a **per-user,
model-estimated relevance**, not a cohort-level order maximising a **measured held-out** metric. Same
class as Lin et al.'s estimated NDCG (see `recency_check_2026-09.md`). **Our claim survives**, and the
distinction is now quotable rather than assumed. Their other arms: `Greedy+MMR`, `DPE-ILD`,
`DPE-Hellinger`, `DPE-Oracle`.

**Greedy SLIM — Proissl, Vatic & Waldschmidt, arXiv 2406.06061 (2024).** Our closest protocol sibling
(ML-25M, NDCG@10, the Cremonesi tail definition). Two findings `[V]`:

> *"We had to implement all the baselines we wanted to consider on our own. For Golbandi et al. [2011]
> and Sepliarskaia et al. [2018] we, unfortunately, obtained **poor results**. We therefore decided to
> **not include them in our official experiments** as there is always the possibility of a
> misinterpretation."*

Its official comparison is therefore against **QBandit alone**. The closest paper to ours dropped the
two most relevant baselines because it could not reproduce them — which is a reproducibility datum in
its own right, and the strongest external support for our replication contribution.

> *"The shortcoming of this approach [Golbandi's] is that the **RMSE is not a good measure for the
> performance of top-N recommender systems** as it was shown in Cremonesi et al. [2010]."*

⭐ A third party in our own literature making the objective-mismatch argument, citing the exact
reference Paper B now cites for the evaluation shift. Quotable.

## The comparator table

| method | compared against | any metric-optimised order? |
|---|---|---|
| Golbandi, WSDM'11 | *not yet extracted* | — |
| Rashid (entropy0/HELF/IGCN), 2002/2008 | *not yet extracted* | — |
| fMF, Zhou SIGIR'11 | *not yet extracted* | — |
| EDDI, ICML'19 | not a recsys paper; feature-acquisition curves | no |
| DRE, WWW'20 | MOSTPOP, RAN++, POP++, RBMF, RBMF++, RMVA, RMVA++ | no |
| DPE, WSDM'21 | Greedy (top predicted item), Greedy+MMR, DPE variants, DPE-Oracle | **no** — estimated, per-user |
| ConUCB, WWW'20 | *not yet extracted* (regret family) | no |
| ConLinUCB, AAAI'23 | LinUCB, Arm-Con, ConUCB, ConLinUCB-BS, ConLinUCB-MCR | no |
| CLiME, KDD'25 | LinUCB, Arm-Con, ConUCB, ConLinUCB-BS, ConLinUCB-MCR | no |
| NICF, SIGIR'20 | Random, Pop, MF, BPR, MLP, eps-greedy, TS, GLM-UCB, MeLU | no |
| Greedy SLIM, 2024 | QBandit only (Golbandi + Sepliarskaia excluded, poor results) | no |
| Lin et al., WWW'23 | MF, MaxEnt, CRM, EAR | **no** — estimated NDCG inside its own selector |
| Gharahighehi et al., 2025 | a HELF variant and their own single-item tree | no |
| PEBOL, RecSys'24 | declines ConvRec baselines | no |

**The pattern.** Comparators are random, popularity, a rating-prediction model, or other methods from
the same family. The two arms that come closest -- DPE's `Greedy` and Lin's greedy NDCG selector --
both maximise a quantity the model *estimates*, per user, at selection time. None of the fourteen
reports a comparison against a question order built by maximising the **measured held-out** value of
the metric it reports.

## Still to extract

Golbandi WSDM'11, Rashid 2002/2008, fMF SIGIR'11, ConUCB WWW'20, Sepliarskaia RecSys'18. The archive
notes Sepliarskaia *"never compares against"* something -- chase that line. The pattern is already
clear from nine of fourteen, but the paper should not print a count until all are checked.
