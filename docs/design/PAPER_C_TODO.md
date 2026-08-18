# Paper C — TODO (author brief, 2026-08-18)

**Author's framing:** *"continuous is what we can marry with LLM for a sexy star-of-the-show paper,
maybe more fun, less beating benchmarks … make it somewhat safe, like pebol, so fun idea, cool
techniques, no clear baselines to beat, more capability paper. should ideally be truly original.
i am counting on a and b to be thorough and serious ones, and this is just basically making the
whole thesis more current and fun and sexy."* · Publication is a **bonus, not a requirement**; no
deadline (`PUBLICATION_PLAN.md` roster reset).

Absorbs old Paper E (LLM-rendered dialogue). Old Paper D is absorbed into B.

---

## What is already SETTLED, and must not be re-litigated

1. **The one-hot distillation target forbids in-between emission.** Nothing in E1's loss rewards
   landing between two questions (`PAPER_C_EMISSION_DESIGN.md` blocker 1).
2. **The snap makes novelty unobservable to any reward.** Two emissions snapping to the same bank
   question give byte-identical interviews, so RL through the snap cannot see novelty (blocker 2).
   ⇒ "add RL" is a route to better *selection*, never to novelty.
3. **★ THE RENDERING HALF IS TAKEN — ELM, Tennenholtz et al., arXiv:2310.04475 (Google).** It decodes
   *"non-existent or hypothetical"* embeddings, interpolates movie AND user embeddings, extrapolates
   along CAV directions, and introduces **consistency metrics precisely because those points have no
   ground truth**. We CANNOT claim novelty for "an LLM verbalises a synthesised latent point."
   What ELM lacks: no interactive loop, no user, **no learned policy choosing which vector to decode**.
4. **An LLM renders ANY vector plausibly — including a scrambled one.** So fluency of a rendered
   question is evidence of NOTHING. Any claim must rest on a control that a gibberish or shuffled
   embedding fails.
5. **There is NO non-circular SIMULATED answerer for a continuous direction** (author, 2026-08-18).
   Geometric a = sign(u*·q) is circular (Jul-25 ruling). Projecting the user's real liked items onto
   q does not escape it — it still runs through tower item embeddings, and adds error. Non-circular
   answers exist only where a real person responds: the 300-user study, or real text.
   ⇒ **C cannot be an NDCG-beating claim from simulated answers.** Capability + controls, or nothing.
6. **CLARINET is not the template.** Fine-tuning an emitter on measured-best candidates emits a BANK
   question — the opposite of in-between (author, 2026-08-18). Keep it as a citation and as evidence
   that measured-utility-trained emitters work.
7. **All June-2026 Paper C numbers are DEMOTED** (pre-Jul-22 retired answer environment). Directions
   are credible priors; every magnitude re-establishes on the certified stack before it is cited.
   That includes the continuity headroom (+0.029/+0.049), the realizable continuous win
   (+0.018/+0.017), and ORADISTILL's failure — the last is a *direction* worth keeping: oracle
   selection did not generalise from a cold-start profile.

## What is CARRIED FORWARD from June (machinery, not numbers)

- **Wolpertinger structure** — actor emits a query point, kNN retrieves candidates, the existing
  scorer ranks them; reduces exactly to discrete when k = full pool. The near-lossless bridge.
- **Learning to predict embeddings** + **snap variants** (author: "important and necessary part").
- **DBU / "Breaking the Grid"** (arXiv:2602.08616) — regress the actor onto a Q-softmax-weighted
  BLEND of retrieved candidates. The one training signal that teaches landing BETWEEN phrases while
  every gradient still comes from executed, measured candidates.
- `new_chapters/paper4_casper/ASKABLE_QUESTIONS.md` — taxonomy of open vs closed question types with
  their embedding targets; reusable as C's action-space section.

**DROPPED from the D salvage (author, 2026-08-18):** the question-framing lever
("underrated/hidden-gem" vs "favourite"). It is measured against an answerer WE chose — the
"names their most distinctive favourite" simulator — so it characterises our assumption, not users.

## Citations to route

**→ Paper C:** ELM (2310.04475, the rendering precedent + the consistency-metric idea) · HyAR (Li
ICLR 2022) · Wolpertinger · DBU 2602.08616 · Gumbel-softmax (Jang 2017) · CLARINET (2405.15784) ·
ACING (2411.12736) · VDLM (2602.15870, RL in embedding space + vec2text renderer) · Vec2Text /
ZSInvert / vec2vec (inverters — and the reason the snap must be justified on ANSWERABILITY, not on
"no decoder exists") · Blair 2605.08360 (text embeddings capture preference only via a
semantic↔preferential correlation — a live objection to frozen SBERT).

**→ Paper B prose (positioning / neighbouring families):** KBRD · KGSF · Sanner 2023 ·
Montazeralghaem 2025 (2510.12015 — hard-codes coarse-to-fine, directly relevant to B's tree) ·
Nguyen 2024.

---

## ★ STEP A — DEEP RESEARCH (do this first)

Each question is phrased so a negative answer is as useful as a positive one.

1. **Has anyone built a learned policy that emits a latent point and has a USER answer it?** ELM
   decodes without a loop; PEBOL/GATE/BED-LLM select then phrase. Is the
   interactive-loop-over-synthesised-points cell genuinely empty?
2. **How do people validate that a rendered latent is FAITHFUL rather than merely fluent?** ELM's
   consistency metrics are the known instance. What else — round-trip re-encoding, contrastive
   controls, human judgement? This is the methodological core of any claim we make.
3. **Is there prior work using a scrambled/random latent as the CONTROL for a renderer?** If that
   design already exists we adopt it; if not, it is a contribution in itself.
4. **What answer models do continuous-action elicitation papers use, and how do they justify them?**
   We have concluded no simulated answerer is non-circular — is that the field's position too, or has
   someone solved it?
5. **Embedding-space RL for dialogue.** VDLM is the closest. Anyone doing it with a user in the loop?
6. **Feeding a latent to an LM directly** (adapter / soft-prompt into token space, as ELM does)
   versus prompting with retrieved concepts. Which is standard now, and what does it cost?

## ★ STEP B — BRAINSTORM (after A)

Seeds, not conclusions:

- **The falsifiable core.** If plausibility proves nothing and no simulated answerer is honest, what
  IS the claim? Candidate: *the emitted point carries information no bank question carries* —
  demonstrated by an answer channel that responds differently to it than to (a) its nearest bank
  neighbour and (b) a scrambled control. Everything else is decoration.
- **Where real answers could come from.** The 300 quarantined study users and the Reddit corpus are
  the only non-circular answer sources on the project. Does C need a small human round-trip? That is
  already flagged in `HARSH_REVIEW_POINTS.md` as blocking the strongest claims across A–E.
- **What the policy is FOR.** B will have shown whether opener-conditioned teachers beat the cold
  tree. C's emitter should aim where B's bank is thinnest — states B's screen showed are improvable
  but where no bank question fits well.
- **The honest-negative option.** "We can emit novel directions, an LLM renders them fluently, and we
  can show that fluency is not faithfulness" is a real capability paper with a real methodological
  contribution — and it survives even if the directions turn out not to help NDCG.

## ★ CIRCULARITY RULING (author, 2026-08-18): accept it, but only as a STATE GENERATOR

*"can we accept circularity, if clearly stated, if we are not trying to show ndcg improvement, but
want to discover hypothetical point in rec space to ask about? to have smth to verbalise?"* — **Yes,
with one line held firm.**

**Harmless use.** The simulated answerer's job is to put the policy at a plausible interview state so
it can emit a point. Nothing downstream is measured through it, so bias in it does not bias a result.
Declared plainly, this is a simulator used to EXPLORE a space, not to SCORE one.

**Fatal use, and it stays closed.** Validating the emitted point by simulating an answer TO THAT
POINT and showing the belief improves. That is the model answering its own question and grading
itself. No claim may route through it.

**⇒ Validation must be a property of the POINT, never an outcome of answering it.** Four that avoid
the answerer entirely:
1. **Genuinely off-catalogue** — distance to the nearest bank question, and the margin over the 2nd
   and 3rd. Pure geometry, and it is the check `PAPER_C_EMISSION_DESIGN.md` already specifies.
2. **Discriminative over real users** — do actual fold-in profiles project differently onto it?
   Computed from real ratings, once; a fact about the direction, not an answer on anyone's behalf.
3. **Renders to coherent text** — ELM's consistency-metric idea, reused.
4. **A scrambled control fails 1–3** — this is what stops fluency from doing the work.

**The consequence to state in the paper, up front:** we can show such a question EXISTS, is DISTINCT
from anything in the bank, and is MEANINGFUL — we cannot show it would HELP. That trade is what buys
the freedom to use a circular state generator at all.

## What ELM actually is (checked 2026-08-18, author asked)

**ELM = Embedding Language Models**, Tennenholtz et al., Google Research, arXiv:2310.04475.

- **Mechanism:** a two-layer MLP adapter maps embeddings from a FROZEN domain model (MovieLens CF
  item/user embeddings) into an LLM token-embedding space; the LLM then describes that vector.
- **Domain: recommendation. Task: NOT elicitation.** One-way embedding to text. No user is asked
  anything, no dialogue, no belief state, and nothing SELECTS which vector to decode - every vector
  is precomputed or analyst-supplied.
- **Why it binds us:** it decodes SYNTHESISED points, not only real ones - interpolates two movie
  embeddings at alpha 0.5, interpolates random pairs of USER embeddings, extrapolates along concept
  directions ("make this movie funnier") - and introduces CONSISTENCY METRICS precisely because such
  points have no ground truth. That is the manoeuvre Paper C needs, already published.
- **Verified negative (our own check):** ELM has NO text-to-embedding path. Its attribute directions
  are per-attribute CAVs pre-trained by the Gopfert et al. WWW 2022 procedure, so it CONSUMES a
  closed vocabulary of directions; it does not generate them from arbitrary text.
- **What stays open for us:** the interactive loop, a learned policy CHOOSING which point to decode
  from an interview state, and validating that the chosen point is worth asking about.

Related, same group: Montazeralghaem 2025 (arXiv:2510.12015) HARD-CODES coarse-to-fine question
ordering - route that citation to Paper B, where the tree makes the same ordering emerge.

## Constraints that bind

No LLM API calls without explicit author approval · never the Reddit API · the 173/300 study users
stay quarantined until deliberately spent · every magnitude re-established on the certified stack ·
a capability claim needs a control that a scrambled input fails.
