# Paper D foundation — NAMING the continuous policy (snap queries to genome concepts)
policy_phase2_cont_v1_best, te[300:] 304 users. Snap each emitted query to nearest genome concept (cosine), read strategy.
mean cos(query, nearest concept) = 0.772 (NEAR but OFF-manifold => names are APPROXIMATE; an LLM should realize the
fuzzy query as a natural question, not a literal concept).

LEARNED STRATEGY (named):
 turn0 (FIXED opener, all 304 users): "complicated plot"
 turn1 branch: "intelligent sci-fi"(153) vs "end of the world"(151)
 turn2+: gruesome, car chase, disaster, christmas, hitchcock, superhero, ... (per-user adaptive)
 top first-3 paths (tree):
   complicated plot -> intelligent sci-fi -> gruesome      (119 users)
   complicated plot -> end of the world  -> car chase      (112)
   complicated plot -> end of the world  -> end of world   (39)
   complicated plot -> intelligent sci-fi-> intelligent scifi(34)

=> The off-manifold continuous policy, when NAMED, is a coherent adaptive decision tree (fixed opener -> taste-region
branch -> refine). This is the bridge to PAPER D: wrap the fuzzy continuous policy in a deployable LLM agent that asks
MEANINGFUL questions (LLM realizes each off-manifold query as natural language; bot-play with a learned answerer).
NOVELTY (Paper D) = continuous latent question policy made conversational/deployable. Code: INTERP=1 in continuous_actor.py.
NEXT (Paper D): mine a richer descriptive-phrase vocab (reviews+genome), better snap, LLM phrasing, bot-play.
