# i26c ep31 — the certified instrument tower

`t2i26c_ep31.pt`, sha256 `cdb74693616cb75f5e3a1796bc35054124b924c81be3fb7e7398358aa73062a0`, 32.3 MB.

Committed to the repository deliberately. Every earlier anchor on this project became unreproducible
because the weights lived only in `.cache/` while the code that scored them moved on
(`anchor-provenance-hole-pbc-recvae`). This file is the one artifact the chapter's numbers depend on,
so it lives in git next to the forward that scores it.

## Provenance — ONE hop from a published anchor

```
RecVAE (hidden 600, latent 200, own val NDCG@10 0.3512)   <- published, frozen decoder W+b
   └── i26c                                                <- this checkpoint
```

`t2final_best.pt` (the i25 tower) is **not** read. Earlier i26 runs warm-started from it, giving
RecVAE → i25 → i26 and leaving "how much of this is really i25?" unanswerable. `--init recvae` removes
that question.

## Exact training command

```
python -u src/instrument/train_i26.py \
    --init recvae \
    --mix 0.25,0.30,0.35,0.05,0.05 \
    --epochs 60 --tag t2i26c --steps_per_epoch 800
```

Batch 128. `warm_lr_scale` 0.1 (WARM taste path 3e-5, NEW exposure branch + z0 3e-4). Trained to a
pre-declared plateau — six epochs with no new `sel` maximum — reached after 35 epochs; ep31 selected.
Code at commit `93fbea6` and later; the curriculum is `src/instrument/interview_curriculum.py`.

## Curriculum — five regimes, sampled fresh per example

| share | regime | what it supervises |
|---|---|---|
| 25% | full-profile dropout, keep U(50%,100%) | ranking from a complete history |
| 30% | **dense small set**, k~U{1..8} from the user's real rated history, all answered | the small-set fold the interview lives in |
| 35% | interview: k~(1,2,4,8,16,32) strategy-selected asks; unrated asks return as **refusal** tokens | sparse, refusal-dominated sets |
| 5% | k=0 (empty input) | the popularity prior, learned not hard-coded |
| 5% | k=1 | minimum evidence |

The dense-small-set bucket was **missing** from earlier i26 curricula and that was the defect: without
it the tower converged at sel 0.1370 and trailed on dense k=2/4/8 fold-ins by 0.010–0.014. Dense sets
are sampled with probability ∝ `cnt^-0.5` so rare titles actually appear — under uniform sampling the
median tail item reached only ~25 dense sets across a run and 10.7% of the catalogue fewer than 5.

## Selection — pre-registered, val only

`argmax(sel + val_full)` over all 35 checkpoints, where `sel = mean(val NDCG@10 at k=2, at k=8)`.
Never touches test. ep31 wins at 0.4873; the author independently chose ep31 from the Pareto front, so
rule and judgement agree.

```
ep30  sel 0.1408  val_full 0.3427  score 0.4835
ep31  sel 0.1407  val_full 0.3466  score 0.4873   <- selected
ep06  sel 0.1357  val_full 0.3479  score 0.4836
```

Recorded val metrics inside the file: `sel 0.140674`, `val_full 0.346629`, `val_tail 0.248468`,
`k0`, `k1`, `k2`, `k8`, `epoch 31`, `arch i26`.

## How to load

```python
from i26_encoder import build_i26
from train_tower_t2 import load_recvae_teacher, apply_sign_prior
src = load_recvae_teacher(n_items, hidden=600, latent=200)
enc, dec, _, _ = build_i26(n_items, src, ma, log=print)   # ma.arch = "i26"
apply_sign_prior(enc)
blob = torch.load("models/i26c_ep31/t2i26c_ep31.pt", map_location="cpu")
enc.load_state_dict(blob["enc"])      # strict: must fully populate
dec.load_state_dict(blob["decoder"])
```

`--arch i26` is required by `arm_n_tower.py`, `arm_n_paired.py` and `run_interview_table.py`; an i26
state dict cannot be loaded into an i25 model and those scripts refuse rather than partially load.

## Test numbers

Filled in from the battery run on this checkpoint — see `docs/results/` and the commit that adds them.
