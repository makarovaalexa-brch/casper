---
name: unified-embedding-novelty-map
description: "June 2026 thorough lit review — novelty verdict for CASPER's unified items+attributes+open-concepts embedding as BOTH recommender and elicitation action space; what's established vs open; must-cite anchors + citation corrections"
metadata: 
  node_type: memory
  type: reference
  originSessionId: f896d008-8222-4f33-8f51-8f8389a9b8c1
---

CASPER's reunified thesis (June 2026): ONE shared embedding space for items + attributes + arbitrary concepts,
serving BOTH the recommender ("instrument") AND the elicitation policy's action space. Five-agent deep lit review
verdict (consistent across angles):

## OPEN / defensible novelty (the contribution) = the CONJUNCTION
A single continuous, text-derived (SBERT/two-tower) embedding space that is simultaneously (i) the recommender — a
FROZEN, policy-independent, PERMUTATION-INVARIANT set encoder over an arbitrary MIX of item-rating AND standalone
(non-item-anchored) attribute/concept reveals, partial-reveal-robust via masked training, scoring items as
popularity-floor + personalization-residual — AND (ii) the action space of a continuous-action (Wolpertinger-style,
proto-action + NN grounding) elicitation policy, spanning items + attributes + ARBITRARY OPEN concepts, LLM-verbalised.
No single published system occupies this intersection.

## ESTABLISHED (must CITE, must NOT claim as ours)
- MF/learned embeddings as interview SEEDS via representativeness/max-volume in the recommender's own latent space:
  Liu et al. RecSys 2011 (RBMF), Fonarev et al. ICDM 2016 (rectangular maxvol/RMVA), Shi/Zhao/Shen TOIS 2017 (Local
  RBMF), Anava et al. WWW 2015 (optimal design). [This is exactly the "MF embedding as seeds" idea — established.]
- Decision-tree interview whose leaves are latent vectors, cold user folded into item-factor space: Zhou et al.
  SIGIR 2011 (functional MF), Sun et al. WSDM 2013 (multi-question trees), Golbandi WSDM 2011 (decoupled, RMSE).
- AL acquisition computed in the MF embedding geometry: Karimi/Schmidt-Thieme ICTAI/IRI 2011, RecSys 2012; VOI
  Boutilier UAI 2003. Survey: Elahi/Ricci/Rubens 2016 (CSR).
- Shared space serving recommender AND a UNIFIED items+attributes DISCRETE policy action space: UNICORN (Deng SIGIR
  2021, dueling DQN over graph nodes), ConTS (Li et al. unified item+attribute Thompson arms, cold-start; TOIS 2021 /
  arXiv 2005.12979). EAR (Lei WSDM 2020) = FM fold-in over items+confirmed attributes (separate policy).
- Permutation-invariant set encoder over an arbitrary observed SUBSET + info-gain acquisition: EDDI/Partial-VAE (Ma
  et al. ICML 2019) — right architecture/idea, wrong domain (not recsys, not item+attribute). Mult-VAE (Liang WWW
  2018), EASE (Steck WWW 2019), CDAE (Wu WSDM 2016) = subset-robust fold-in, item-only.
- Items+attributes as concepts/directions, mixed feedback, but FIXED schema or Bayesian (not one frozen learned
  encoder + continuous RL): Biyik et al. 2023 (soft attributes via CAVs), Gopfert WWW 2022, Latent Linear Critiquing
  (Luo WWW 2020), CE-VAE/Deep Critiquing (Wu RecSys 2019), PEBOL (Austin RecSys 2024, NLI beliefs, NO shared embedding).
- Continuous-action RL: Wolpertinger (Dulac-Arnold arXiv:1512.07679, 2015, item selection, non-conversational);
  ECoC (arXiv:2408.08047, KBS 2025, items-only, non-conversational). Discrete-strategy RL+LLM: PPDPP (Deng ICLR
  2024, arXiv:2311.00262), RSO (arXiv:2509.26093, 2025) — closed strategy sets.

## NEAREST ANCHORS to cite-and-differentiate (lead related work with these)
ConTS (unified item+attr, cold-start, but FIXED categorical attributes) ; UNICORN (shared space + unified policy, but
DISCRETE graph-Q, no open concepts, not cold-start) ; EDDI (set-encoder over arbitrary subset + info-gain, but not
recsys/no attributes) ; Wolpertinger (continuous+NN action mechanism, but item selection, no recommender tie-in) ;
EAR + Latent Linear Critiquing (item+attribute scoring in one space, but policy-entangled / linear-fusion / critique-only) ;
Biyik soft-attributes + PEBOL (items+concepts but Bayesian / no shared dense embedding).

## CITATION CORRECTIONS (carry into the paper)
- DRE = Kweon, Kang, Hwang, Yu, WWW 2020 (NOT 2024; arXiv:2402.16327 is a reupload).
- There is NO "Liu et al. 2017": RBMF = Liu et al. RecSys 2011; Local-RBMF = Shi/Zhao/Shen TOIS 2017; RMVA = Fonarev
  et al. ICDM 2016.
- Functional MF = Zhou/Yang/Zha SIGIR 2011 (not KDD). Golbandi tree = WSDM 2011. Rashid "Getting to Know You" = IUI 2002.
- PPDPP arXiv id = 2311.00262.

## ATTRIBUTES — where they sit
Two paradigms barely talk: ITEM-only elicitation (RBMF/DRE/Golbandi/bot-play, our mf_foldin baseline) vs ATTRIBUTE-only
CRS over a FIXED schema (EAR/SCPR/UNICORN). CASPER's continuous semantic space is the bridge: probe items OR attributes
OR open concepts, all as points. CRITICAL architectural seam to solve: the recommender must INGEST attribute answers
(token = (attribute_embedding, response)); train with reveal-subset augmentation mixing item-rating + attribute-pref
tokens (derive GT attribute prefs from item ratings). Otherwise "entities" silently collapse back to items.

Related: [[replication-breakthrough-frozen-instrument]] (the mf_foldin baseline + the pop-floor+residual lesson).
