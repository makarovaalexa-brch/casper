# CASPER Training Time Estimates

## Computational Analysis

### Bottleneck: GPT API Calls (not computation!)

**Per episode breakdown (10 turns):**
- Question generation: 10 × GPT calls (~0.5s each) = 5s
- User simulator: 10 × GPT calls (~0.8s each, function calling) = 8s
- Preference extraction: 10 × GPT calls (~0.5s each) = 5s
- Neural network ops (actor, recommender, FAISS): 10 × 70ms = 0.7s
- Actor-critic training: 1 × 200ms = 0.2s

**Total per episode: ~19 seconds**
- 18s GPT API calls (95% of time)
- 1s neural network computation (5% of time)

**Implication:** CPU vs GPU makes almost no difference - GPT API latency dominates.

---

## Training Time Estimates

### Full Training (1000 episodes as configured)

**Components:**
1. Two-tower recommender training (20 epochs, 500 users)
2. RL training (1000 episodes, 10 turns each)
3. Baseline evaluation (100 episodes × 3 agents)

| Component | CPU (Laptop) | GPU (Colab) | Notes |
|-----------|--------------|-------------|-------|
| Recommender | 2 min | 30 sec | Small model, minimal difference |
| RL training (1000 ep) | 5.3 hours | 5.2 hours | GPT API bottleneck |
| Baseline eval (300 ep) | 1.6 hours | 1.5 hours | GPT API bottleneck |
| **TOTAL** | **~7 hours** | **~6.8 hours** | Barely any GPU benefit |

**Verdict:** GPU provides minimal speedup (<10%) due to GPT API bottleneck.

---

## Recommended Training Configurations

### Quick Test (Paper 1 feasibility)
- RL: 100 episodes
- Eval: 30 episodes per agent

| Time | CPU | GPU |
|------|-----|-----|
| Total | ~45 min | ~42 min |

**Good for:** Sanity check, debugging, quick results

---

### Medium Training (Paper 1 submission)
- RL: 500 episodes
- Eval: 100 episodes per agent

| Time | CPU | GPU |
|------|-----|-----|
| Total | ~3.5 hours | ~3.3 hours |

**Good for:** Paper 1 results, reasonable training

---

### Full Training (best results)
- RL: 1000 episodes
- Eval: 100 episodes per agent

| Time | CPU | GPU |
|------|-----|-----|
| Total | ~7 hours | ~6.8 hours |

**Good for:** Final paper results, best performance

---

## Optimization Strategies

### 1. Reduce Episodes (Recommended)
```python
# In config.yaml, change:
reinforcement:
  num_episodes: 500  # instead of 1000
```
**Saves:** ~2.5 hours
**Impact:** Minimal (500 episodes likely sufficient for Paper 1)

### 2. Reduce Conversation Turns
```python
# In config.yaml, change:
environment:
  max_turns: 5  # instead of 10
```
**Saves:** ~50% time per episode
**Impact:** Moderate (fewer questions per conversation)

### 3. Batch GPT Calls (Future work)
Currently: Sequential API calls
Could: Batch requests with async/await
**Potential savings:** 30-40%
**Effort:** Moderate code changes

### 4. Use Faster Model
Currently: gpt-5-nano (~0.5-0.8s per call)
Alternative: gpt-3.5-turbo (~0.3-0.5s per call)
**Saves:** ~30% time
**Impact:** Potentially lower quality responses

---

## Practical Recommendations

### For Your Laptop (CPU)
**Best config for Paper 1:**
- RL: 500 episodes
- Eval: 50 episodes per agent
- Turns: 10

**Estimated time:** ~2.5 hours
**Run overnight or during work**

### For Google Colab (GPU)
**Best config for Paper 1:**
- RL: 1000 episodes (use the free GPU!)
- Eval: 100 episodes per agent
- Turns: 10

**Estimated time:** ~6.8 hours
**Fits within Colab's ~12 hour session limit**

**Strategy:**
1. Start training in evening
2. Let it run overnight
3. Check results in morning
4. Colab saves checkpoints to Google Drive every 100 episodes

---

## Memory Requirements

### RAM:
- Local: ~4-6 GB (MovieLens + SentenceBERT + models)
- Colab: 12 GB available (plenty)

### GPU VRAM (if using):
- Models: ~500 MB (small neural nets)
- SentenceBERT: ~400 MB
- Batch processing: ~200 MB
- **Total: ~1.2 GB** (any modern GPU works)

### Disk:
- MovieLens dataset: ~1 GB
- Checkpoints: ~100 MB per checkpoint
- Total: ~2 GB

---

## Actual Recommendation

**For Paper 1 deadline (Jan 29):**

### Option A: Laptop overnight (recommended)
```yaml
# config.yaml
reinforcement:
  num_episodes: 500
environment:
  max_turns: 10
```
- Start before bed
- Finishes in ~3 hours
- Wake up to results
- No Colab dependency

### Option B: Colab full training
```yaml
# config.yaml
reinforcement:
  num_episodes: 1000
environment:
  max_turns: 10
```
- Start in evening
- Runs ~7 hours
- Better results for paper
- Automatic Drive backup

**Verdict:** Use Colab for convenience (auto-saves, no laptop battery drain), not for speed.

---

## Time Breakdown (1000 episodes)

```
Component                    Time        % of Total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPT API calls               ~5.5 hrs     82%
Neural network ops          ~0.6 hrs     9%
Data loading/processing     ~0.4 hrs     6%
Checkpoint saving           ~0.2 hrs     3%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                       ~6.7 hrs     100%
```

**Key insight:** GPU only speeds up 9% of the work (neural nets). The other 91% is GPT API calls and I/O, which are the same on CPU/GPU.

---

## Summary

**TL;DR:**
- **CPU vs GPU:** Almost no difference (~6% speedup)
- **Bottleneck:** GPT API latency, not computation
- **Recommended:** 500 episodes on laptop (3 hours) or 1000 episodes on Colab (7 hours)
- **Paper 1:** Either config gives sufficient results
- **Best strategy:** Start on Colab before bed, wake up to results
