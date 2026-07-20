### Slate 1 (top-100 movies)  (prior=0.700)

| Policy | AUAC@5 | AUAC@15 | turns→random@15 | speedup | %info vs random |
|---|---|---|---|---|---|
| PPO+ans | 0.7251 | 0.7524 | 7 | 2.1× | 144% |
| Bot-play | 0.7236 | 0.7516 | 7 | 2.1× | 139% |
| SCPR-entropy | 0.7265 | 0.7510 | 9 | 1.7× | 149% |
| PPO | 0.7243 | 0.7492 | 9 | 1.7× | 133% |
| Bot-play+ans | 0.7296 | 0.7473 | 11 | 1.4× | 119% |
| Greedy info-gain | 0.7260 | 0.7472 | 10 | 1.5× | 134% |
| Bot-play v2 | 0.7253 | 0.7422 | 12 | 1.2× | 113% |
| DQN | 0.7268 | 0.7397 | — | — | 93% |
| Popularity | 0.7254 | 0.7385 | 14 | 1.1× | 101% |
| llm_vanilla_gpt-4o-mini | 0.7123 | 0.7375 | 12 | 1.2× | 123% |
| Random | 0.7162 | 0.7351 | 15 | 1.0× | 100% |
| llm_strategist_gpt-4o-mini | 0.7090 | 0.7303 | 14 | 1.1× | 111% |
| Thompson | 0.7116 | 0.7271 | — | — | 86% |
| llm_gate_gpt-4o-mini | 0.7075 | 0.7252 | — | — | 84% |