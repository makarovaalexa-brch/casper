# DS_C3_POC — LLM rendering proof-of-concept on the C1 pilot points

**Status: PRE-REGISTERED 2026-08-21. Author asked for "some llm proof of concept on these"** —
interpreted and bounded as: Claude-in-session as the LLM (renderer + blinded judge subagents),
NO external LLM API, no training. This is a PoC of **variant (a) (retrieval-anchored prompting)**,
which `PAPER_C_DESIGN.md` designates the BASELINE arm — the claim-bearing renderer is the trained
adapter (variant b), out of scope today. Nothing here may be quoted as the paper's result; it
demonstrates machinery and the control design.

## Questions (each with a registered reading)

1. **Does the pipeline run end to end?** point → nearest-bank anchors + interview context → rendered
   natural-language question. (Existence demo.)
2. **Does fluency discriminate?** A blinded judge, seeing rendered TEXT ONLY, classifies each
   question as from a real policy direction vs a scrambled control. **Registered prediction:
   accuracy ≈ chance** — fluency proves nothing (the audit's premise, on our own instrument).
3. **Does the CONTROL discriminate, and through what?** Re-encode each rendered question through the
   frozen SBERT→concept open-vocab path (Paper A's, independently fitted) and compare its direction
   to the source point. Prediction: true-point texts land closer to their source than
   scrambled-correspondence texts do — **but under variant (a) this tests RETRIEVAL, not rendering**
   (the LLM never sees the vector). The write-up must say so in those words; showing this
   limitation empirically is itself the PoC's value (it is why variant (b) exists).

## Materials

- **Points**: from the C1 pilot's banked artifacts — for ~10 states chosen to span depth, channel
  mix, and one low-coherence dislike-path state: the m=5 blend target AND the fine-tuned emission ẑ
  (full-policy arm, seed 1). Bias-free space.
- **Controls, one per real point**: (i) scrambled correspondence — the packet keeps state A's
  interview context but carries state B's point (derangement); (ii) spectrum-matched random vector.
- **Packet** per item (id-keyed, labels in a SEPARATE key file the renderer/judge never see):
  top-8 nearest bank phrases with cosines (mean-removed), the blend ingredients are NOT named as
  such (the renderer must not know which packets are real), the asked-so-far path in plain words,
  and the instruction: write ONE askable interview question capturing the direction, not reusing
  any anchor phrase verbatim.
- **Renderer**: one fresh Sonnet subagent, sees packets only. **Judge**: a different fresh Sonnet
  subagent, sees rendered questions only (not packets), guesses real vs control + rates
  answerability 1–5. Blinding = separation of agents + label file quarantine.

## Re-encode bridge (settled 2026-08-21, before the stage ran)

The SBERT recipe ends in the 200-d tower latent; the source points live in the 128-d bias-free
policy-head space. **The comparison happens in the head space:** rendered text → SBERT sims over
the 1,031 tag names (`mode='sent'`, topn 4, mincos 0.30 — the smoke-trio constants of record) →
the selected concepts' own bank embeddings `k(φ)` → their **uniform mean direction** (primary;
selection-cosine-weighted mean as a secondary row) → **mean-removed cosine** to the packet's source
vector, same convention as every other geometry number today. Column alignment between the packets'
concept channel and the SBERT tag list is ASSERTED, not assumed. Texts whose SBERT pass selects
zero concepts above threshold are reported as non-encodable, not dropped.

## Registered readings

- Judge accuracy on real-vs-scrambled from text alone: report with exact n; ≈ chance ⇒ the fluency
  point stands; ≫ chance ⇒ investigate WHAT leaks (anchors' topical coherence is the suspect —
  that finding would sharpen the paper's control design and is welcome).
- Re-encode gap: mean cos(re-encoded text, source point) for true vs scrambled vs random. Any gap
  is attributed to RETRIEVAL, never to rendering, in every sentence that reports it.
- Failure to render (LLM refuses/waffles on incoherent anchors) is REPORTED per packet — Baum–Lang
  answerability in action, not a bug.

## What may NOT be claimed

No faithfulness claim (that is C4 with variant (b)) · no novelty claim · no "the renderer
understands the latent" — variant (a)'s LLM never sees it · nothing from this PoC enters the
chapter as a result; at most the Setup/C3 section may cite it as a pipeline demonstration and the
limitation demo, labelled PoC.

## Cost

Extraction: seconds of numpy (banked arrays only). SBERT encoding of ~30 short strings: seconds on
CPU. Rendering/judging: in-session subagents. No training, no API, no test users, nothing heavy.
Outputs: `experiments/paperC/c3_poc/` (packets, key, renders, judgments, re-encode table) +
`docs/results/C3_POC.md`.
