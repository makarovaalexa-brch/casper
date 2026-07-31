# The critiquing line is PROJECTION, not open input — verified, 8 systems, 0 exceptions

Research 2026-07-31. **Question asked:** does any critiquing-based recommender accept a critique that is
NOT an index into a pre-committed symbol set? This decides whether Paper A's **R2 (an open set-valued
input interface)** has a live counterexample in the critiquing line — the one family that superficially
looks like it takes free-form user input.

**VERDICT: ALL EIGHT ARE PROJECTION. No exceptions found.** In every system the utterance is resolved to
an index in a closed vocabulary *before* it reaches the model: a keyphrase id (1–6), a KG fact (7), or a
feature/attribute slot (8). Three of them represent the critique as a continuous vector *downstream*
(CE-VAE, M&Ms-VAE, BK-VAE) — but that vector is always a function of a one-hot over a closed set. The
continuity is entirely after the argmax.

**Scale worth quoting in the paper:** the whole Sanner lineage runs on **40 keyphrases (CDs&Vinyl) / 75
(BeerAdvocate)** — a 40-symbol critique language for a 4,395-item catalogue.

## The eight

| # | System | Critique object | Vocabulary | Verdict |
|---|--------|-----------------|-----------|---------|
| 1 | Wu/Luo/Sanner/Soh, **DLC**, RecSys'19 | one-hot keyphrase → embedded → fused into NCF/VNCF latent | **40 / 75**, counted from author CSVs | PROJECTION |
| 2 | Luo et al., **LLC**, WWW'20 | *"critiques … are encoded as **one-hot keyphrase indicators**"* | **40 / 75** (Table 3 caption) | PROJECTION |
| 3 | Luo et al., **CE-VAE**, SIGIR'20 | *"**A one-hot vector with keyphrase length s** whose sole positive value position indicates the index"* | 40 / 75 | PROJECTION |
| 4 | Yang/Shen/Sanner, **BK-VAE**, SIGIR'21 | keyphrase index → TCAV-style KAV direction in VAE latent; Gaussian conjugate posterior over user μ,Σ | fixed **MovieLens tag inventory** (`tag_id_dict.json`); **size UNVERIFIED** | PROJECTION |
| 5 | Antognini & Faltings, **M&Ms-VAE**, RecSys'21 | *"**A one-hot vector of length \|K\|.** The only positive value indicates the index"* | Beer 75 / CDs 40 / Yelp 234 / Hotel 141 | PROJECTION |
| 6 | Li/Majumder/McAuley, **Bot Play**, arXiv 2112.05197 | cumulative critique vector over a fixed aspect set | Books 75 / Beer 75 / Music 80 | PROJECTION |
| 7 | Toroghi et al., **BCIE**, SIGIR'23 | *"the user may either accept the recommendation or **critique a fact f ∈ F**"* — a KG triple | ML-20M **56,789 entities / 46 relations**; Amazon-Book 79,682 / 38 | PROJECTION (largest symbol set) |
| 8a | Burke et al., **FindMe/RENTME**, AAAI-96 | fixed "tweak" buttons (Cheaper, Nicer, Bigger) → ordered ops on a constraint set | fixed slot-value schema | PROJECTION |
| 8b | Chen & Pu survey, UMUAI 2012 | unit critique *"can only **constrain over a single feature** at a time"*; compound = vector of (attribute, tradeoff) pairs | fixed attribute schema | PROJECTION |

**The nearest thing to a counterexample is BCIE** — ~57k–80k entities × ~40 relations, three orders of
magnitude above the keyphrase lineage. If we need to hedge R2's strength, hedge it there. But it is still
strictly projection: the user must critique an existing `(h,r,t)` fact **the system itself surfaced** from
the candidate set F, and BCIE's own "Mapped items" baseline mines ≤10 KG items per critique, which only
works *because* the critique is a KG symbol. A vector the model never saw at fit time has no address.

## Second finding — the evaluation split, and who has no ruler

- **Held-out ranking metrics AND simulated critiquing:** DLC, CE-VAE, M&Ms-VAE, BCIE.
- **Simulated critiquing ONLY — no NDCG / Recall / MAP anywhere in the paper (grep-verified):**
  **LLC (WWW'20)** and **Bot Play**. LLC reports Success Rate@{1,5,10} + Average Session Length against an
  "Oracular Upper Bound"; Bot Play reports Success Rate@N, average turns, and an ACUTE-Eval human study,
  using AUC only for hyperparameter selection.
- **FindMe and the Chen & Pu survey:** no offline ranking evaluation at all (user studies only).

**The two papers making the strongest claims about *conversation* are exactly the two with no ranking
ruler.** That is directly usable in Paper A's protocol section and in Paper B's measurement framing.

## Corrections to our bibliography

- **DLC's repo is `wuga214/DeepCritiquingForRecSys`**, NOT `k9luo/DeepLanguageBasedCritiquing` (404s). Fix.
- BK-VAE's exact title is *"Bayesian Critiquing with Keyphrase Activation Vectors for VAE-based
  Recommender Systems"*, SIGIR 2021, DOI 10.1145/3404835.3463108.
- BCIE = *"Bayesian Knowledge-driven Critiquing with Indirect Evidence"*, SIGIR 2023, arXiv 2306.05636.
- Repo map: LLC → `k9luo/LatentLinearCritiquingforConvRecSys` · CE-VAE →
  `k9luo/DeepCritiquingForVAEBasedRecSys` · BK-VAE → `hojinYang/bayesian-critiquing-recommender` ·
  BCIE → `atoroghi/BCIE`.

## BCIE and our belief layer — read before task #67

BCIE puts a **Gaussian conjugate posterior over the user embedding**, like ours, but in two stages:
*"we introduce a new variant of SimplE in which the sigmoid likelihood is replaced by a Gaussian, forming
a conjugate pair with the Gaussian prior assumed over the user belief; this, in turn, enables tractable
closed-form belief updates using Gaussian Belief Propagation."* In information form the **item**
distribution `p(z_m|z_d) ~ N⁻¹(h_d+h_m, J_d+J_m)` is updated first, and that belief then passes to
`z_u ~ N⁻¹(h_u, J_u)` by marginalising the joint. **Their "Direct" ablation is precisely the version that
skips the item distribution and attributes the critique straight to the user belief** — i.e. our shape.
If we keep the belief layer, this is the prior art to position against, and their own ablation is the
comparison they already ran.

## Verification status — three cells are NOT quotable

1. **BK-VAE vocabulary size** — no accessible PDF (ACM closed, no preprint). Mechanism and source
   (MovieLens tag inventory) verified **from code only**. **Do not quote a number.**
2. **BK-VAE recommendation-performance claim** — abstract only.
3. **DLC — no sentence from the paper itself was verified.** ACM 403s behind Cloudflare, unpaywall
   `is_oa: false`, no arXiv version. Everything for DLC comes from the released code: the literal 40- and
   75-row `KeyPhrases.csv` files, the `BigramCollocationFinder` + `apply_freq_filter(100)` + PMI + POS
   extraction notebooks, and `return_keyphrase_index()`. That is arguably *stronger* evidence for the
   vocabulary than a paper sentence — but it is not a paper quote, and the chapter must not present it as
   one. To quote DLC or BK-VAE, get the PDF through institutional access first.

Everything else in the table is verified from the full PDF, most with the sentence quoted above.

## ⚠ RETRACTION NOTICE (added 2026-07-31, later the same day)

A parallel investigation reported that **two of nine sub-agents fabricated primary-source content before
self-correcting**, and the retracted list explicitly includes **the Burke (AAAI-96) and Chen & Pu
(UMUAI 2012) quotes** and **BK-VAE internals**. Therefore, in the table above:

- **Rows 8a and 8b: the QUOTED SENTENCES are provisional and must be re-verified from the PDFs before
  any use in the chapter.** The PROJECTION verdict for FindMe (fixed tweak buttons over a slot-value
  schema) and for the Chen & Pu survey (unit critique = one feature at a time) rests on well-known,
  independently-attested properties of those systems and is not in doubt — but do not print their
  sentences.
- **Row 4 (BK-VAE): mechanism description is provisional too**, on top of the vocabulary-size cell
  already marked unquotable.

Rows 2, 3, 5, 6, 7 were verified from full PDFs with sentences extracted, and row 1 from the authors'
released code and data files. Those stand. See `organic_open_request_novelty.md` for the full caveat.
