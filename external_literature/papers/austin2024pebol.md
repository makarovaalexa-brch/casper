# austin2024pebol — Austin et al. 2024, Bayesian Optimization with LLM Acquisition for NL Preference Elicitation (PEBOL)

- **Venue/year:** RecSys 2024
- **Link / DOI:** arXiv:2405.00981
- **Status:** skimmed (also digested in elicitation_and_channels findings; reconcile key with `austin2024bayesian` in bib)

## Essence
Natural-language preference elicitation as Bayesian optimization: maintain a Beta posterior per catalog item, use
LLM-extracted ≤3-word aspects + NLI entailment to softly up/down-weight items, and pick the next question by a BO
acquisition (Thompson / UCB). Strong cold-start: MRR@10 **0.27** vs 0.17 baseline, +131% MAP@10. Preferences 100% LLM-simulated.

## Method in one paragraph
Each turn the LLM proposes an aspect; NLI scores each item's description for entailment of that aspect, updating a
per-item Beta belief; a BO acquisition function selects the aspect that best reduces uncertainty / maximizes expected
reward. Uncertainty is PER-ITEM (independent Betas), NOT a shared user covariance; no full-profile CF ranking; no
monotonicity guarantee; cannot represent a NOVEL taste direction that no fixed aspect/item names.

## Relevance to CASPER
- **Papers:** C (load-bearing PE baseline), A (uncertainty-native contrast).
- **Taxonomy slot:** (vi) LLM-as-recommender / principled NL-PE.
- **Baseline candidate?** yes (Tier 3) — cold-start MRR@10 0.27 (LLM-sim users); public code.
- **Pre-empts / supports:** proves principled-selection + LLM-phrasing works; its per-item Beta (no shared covariance,
  no CF geometry) is exactly what our shared-belief frozen-tower instrument adds.

## Verdict
Cite as the strongest recent NL-PE system; its per-item Bayesian belief without a CF tower is the contrast that
motivates our shared-covariance-over-frozen-SOTA design.
