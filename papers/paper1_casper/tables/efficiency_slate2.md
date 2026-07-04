### Slate 2 (mid-popularity + genome tags)  (prior=0.638)

| Policy | AUAC@5 | AUAC@15 | turns→random@15 | speedup | %info vs random |
|---|---|---|---|---|---|
| PPO+ans | 0.6851 | 0.7077 | 6 | 2.5× | 123% |
| PPO | 0.6854 | 0.7069 | 6 | 2.5× | 119% |
| Bot-play | 0.6862 | 0.7058 | 6 | 2.5× | 114% |
| SCPR-entropy | 0.6832 | 0.7053 | 7 | 2.1× | 117% |
| DQN | 0.6871 | 0.7053 | 6 | 2.5× | 112% |
| Bot-play+ans | 0.6878 | 0.7051 | 9 | 1.7× | 115% |
| Greedy info-gain | 0.6874 | 0.7019 | 11 | 1.4× | 103% |
| Popularity | 0.6783 | 0.6942 | — | — | 99% |
| Random | 0.6614 | 0.6859 | 15 | 1.0× | 100% |
| Thompson | 0.6477 | 0.6577 | — | — | 41% |