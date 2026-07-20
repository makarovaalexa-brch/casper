# Answerer / fold — TODO (author notes)

- **Favourite-movie recall samples the HIGH-AFFINITY REGION, not a random highly-rated item**
  (author note, Jul 10). When a user names a favourite in free recall, it's drawn from the tight
  cluster of their strongest taste, not any 4.5-star item. Implication: the answerer's free-recall
  channel (Paper D) should model "favourite" as a sample from the user's high-affinity region, and
  the fold should treat a volunteered favourite as a high-affinity anchor. TEST on the 173 LLM-judged
  data (do named favourites cluster tighter than random high-rated items?). Relevant to Paper D
  (open recall) + the answerer value model.
