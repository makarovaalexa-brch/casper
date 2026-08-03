# SHELVED (author ruling 2026-08-03: no emails — likely ignored, too slow to gate anything)
# Plan of record instead: ZhankuiHe/reddit_movie_raw + LLM-assisted linking (see memory
# paperb-kickoff-artifacts-2026-08-03). Draft kept only in case the parallel-shot option is revived.

# DRAFT — email to Lukas Eberhard (TU Graz), task #68 — NOT SENT

Subject: Request texts for your r/MovieSuggestions dataset (IUI 2019)

Dear Dr. Eberhard,

I am a PhD student working on cold-start conversational recommendation, and your IUI 2019 paper
"Evaluating Narrative-Driven Movie Recommendations on Reddit" is the closest published artifact to
the evaluation we need. Thank you for releasing the annotated dataset on OSF — we have already
verified it fits our setup well: the IMDb identifiers in your 1,480 annotated threads cover 98.9%
of mentions in our MovieLens-based catalogue, and every thread retains at least five linked items.

The OSF release, as far as we can tell, ships the annotations and identifiers but not the text of
the original submission (the requester's narrative). For our study — measuring what an adaptive
follow-up interview adds *after* a real, organically formulated request — the request text is the
essential input. Would you be able to share the submission texts for the annotated threads, or the
submission IDs paired with an archived text source you used, under whatever terms you consider
appropriate? We would be glad to sign a data-use agreement, and we do not intend to redistribute
the texts or rehydrate anything through the Reddit API.

For context on how we would use it: your dataset would serve as the evaluation ruler (request as
given evidence, your annotated replies as ground truth, OP-named items excluded as in your setup),
with our contribution being the follow-up interview measured downstream of the request. We would of
course cite the dataset and paper as the foundation of the evaluation, and we would be happy to
share our results, including a Most-Popular baseline number on your corpus, which to our knowledge
has not been published.

Thank you for considering this, and for making the dataset public in the first place — it is the
only artifact of its kind we could find.

Best regards,
Aleksandra Makarova
[affiliation / signature to add]

---
Notes for the author (not part of the email):
- Recipient: Lukas Eberhard, TU Graz (verify current affiliation/address before sending — he may
  have moved since 2019; check the OSF project contact and his homepage).
- If no reply in ~2 weeks: fallback is ZhankuiHe/reddit_movie_raw (has text; 61.3%/75.1% catalogue
  fit; ~3–5 weeks re-linking) — recorded in `reddit-mined-open-request-opener.md`.
- Do not promise anything about votes; we use them at most as a coarse inclusion filter, as he did.
- Sending is outward-facing: author sends, not the assistant.
