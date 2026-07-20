# OpenAI API Cost Estimates for CASPER Training

## Training Configuration

### GPT API Calls Breakdown

**Per conversation turn:**
- Question generation: 1 call (~200 input + 50 output tokens)
- User simulator: 1 call (~300 input + 100 output tokens, includes function calling)
- Preference extraction: 1 call (~200 input + 100 output tokens)

**Total per turn:** 3 GPT calls, ~700 input tokens, ~250 output tokens

---

## Full Training (1000 episodes)

### RL Training
- Episodes: 1000
- Turns per episode: 10
- Total turns: 10,000
- GPT calls: 30,000

**Token usage:**
- Input tokens: 10,000 × 700 = 7,000,000 (7M)
- Output tokens: 10,000 × 250 = 2,500,000 (2.5M)

### Baseline Evaluation (3 agents × 100 episodes each)
- Episodes: 300 total
- Turns per episode: 10
- Total turns: 3,000
- GPT calls: 9,000

**Token usage:**
- Input tokens: 3,000 × 700 = 2,100,000 (2.1M)
- Output tokens: 3,000 × 250 = 750,000 (0.75M)

### Total
- **GPT calls: 39,000**
- **Input tokens: 9.1M**
- **Output tokens: 3.25M**

---

## Cost Estimates (gpt-5-nano)

**Official OpenAI Pricing (2025):**
- Input: $0.05 per 1M tokens
- Output: $0.40 per 1M tokens
- Cached input: $0.01 per 1M tokens (for repeated content)

**Note:** GPT-5-nano is the cheapest GPT-5 model - 95% cheaper on input and 96% cheaper on output vs GPT-4o.

### Full Training Cost (1000 episodes + baselines)

**Token usage:**
- Input: 9.1M tokens
- Output: 3.25M tokens

**Costs:**
- Input: 9.1M × $0.05 / 1M = **$0.46**
- Output: 3.25M × $0.40 / 1M = **$1.30**
- **Total: $1.76**

### With Caching Optimization (Conservative 30% cache hit rate)

**Cached tokens:** ~2.7M input (repeated prompts, system messages)
**Non-cached tokens:** ~6.4M input

**Costs:**
- Cached input: 2.7M × $0.01 / 1M = **$0.03**
- Non-cached input: 6.4M × $0.05 / 1M = **$0.32**
- Output: 3.25M × $0.40 / 1M = **$1.30**
- **Total: $1.65**

---

## Estimate by Configuration

| Configuration | GPT Calls | Input Tokens | Output Tokens | Cost (actual) |
|---------------|-----------|--------------|---------------|---------------|
| Quick (100 ep) | 3,900 | 910K | 325K | **$0.18** |
| Medium (500 ep) | 19,500 | 4.55M | 1.63M | **$0.88** |
| Full (1000 ep) | 39,000 | 9.1M | 3.25M | **$1.76** |

*(Using actual gpt-5-nano pricing: $0.05/$0.40 per 1M tokens)*

---

## Cost Breakdown by Component

### Per Episode (~10 turns)
- RL training: 30 GPT calls
- Cost per episode: **~$0.0093** (full pricing)

### Per Agent Evaluation (100 episodes)
- Baseline testing: 3,000 GPT calls
- Cost per agent: **~$2.33** (full pricing)

### Function Calling Overhead
User simulator uses OpenAI function calling (3 tools):
- Adds ~20% token overhead for tool definitions
- Already included in estimates above

---

## Optimization Strategies

### 1. Reduce Episodes
```yaml
reinforcement:
  num_episodes: 500  # instead of 1000
```
**Savings:** ~50% ($4.66 instead of $9.32)
**Impact:** Minimal for Paper 1

### 2. Reduce Conversation Turns
```yaml
environment:
  max_turns: 5  # instead of 10
```
**Savings:** ~50% ($4.66 instead of $9.32)
**Impact:** Moderate (fewer questions)

### 3. Reduce Baseline Episodes
```python
NUM_TEST_EPISODES = 50  # instead of 100
```
**Savings:** ~25% ($6.99 instead of $9.32)
**Impact:** Lower statistical confidence

### 4. Cache Common Responses
- Cache user simulator responses for identical questions
- Cache preference extractions for identical conversations
**Potential savings:** 10-15%
**Effort:** Requires code changes

---

## Practical Budget Recommendations

### For Paper 1 (Conservative)
**Configuration:**
- RL: 500 episodes
- Eval: 50 episodes per agent
- Turns: 10

**Actual cost:** $0.88
**Sufficient for:** Paper 1 submission with good results

### For Best Results (Recommended)
**Configuration:**
- RL: 1000 episodes
- Eval: 100 episodes per agent
- Turns: 10

**Actual cost:** $1.76
**Sufficient for:** High-quality Paper 1 results, strong baselines

### For Quick Testing (Development)
**Configuration:**
- RL: 100 episodes
- Eval: 20 episodes per agent
- Turns: 5

**Actual cost:** $0.10
**Sufficient for:** Code testing, debugging, sanity checks

---

## Cost Comparison: Alternatives

### Using gpt-3.5-turbo (cheaper, lower quality)
- Input: $0.50 / 1M tokens
- Output: $1.50 / 1M tokens
- **Full training cost: ~$9**
- Quality impact: ~10-15% worse responses

### Using gpt-4o-mini (if available)
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens
- **Full training cost: ~$3**
- Quality impact: Minimal (excellent model)

### Using gpt-4 (better quality, expensive)
- Input: $5.00 / 1M tokens
- Output: $15.00 / 1M tokens
- **Full training cost: ~$94**
- Quality impact: +5-10% better responses
- **Not worth it for Paper 1**

---

## Hidden Costs (Minimal)

### Failed API Calls
- Retry overhead: ~1-2% additional calls
- Cost impact: +$0.10-0.20

### Development/Testing
- Debugging runs: ~10-20 test episodes
- Cost: ~$0.50-1.00

### Experimenting with Hyperparameters
- Testing different configs: 2-3 additional runs
- Cost: ~$10-30 total

---

## Budget Summary for Paper 1

### Minimum Viable (Budget Option)
**Setup:**
- 1 quick test run: $0.10
- 1 medium training run: $0.88
- **Total: ~$1.00**

### Recommended (Best Results)
**Setup:**
- 2 quick test runs: $0.20
- 1 full training run: $1.76
- 1 rerun with different seeds: $1.76
- **Total: ~$4.00**

### Conservative Estimate with Buffer
**Setup:**
- Testing & debugging: $0.50
- Main training: $1.76
- Baseline comparisons: $1.30
- Reruns & experiments (2-3 full runs): $3.50
- **Total: ~$7.00**

---

## Real-World Expectations

**Most likely cost for Paper 1:** $3-5

**Why this range:**
- 1-2 full training runs: $1.76-3.52
- Some debugging/testing: $0.50
- Potential reruns: $1.00

**Budget recommendation:** Set aside **$10** for safe margin (includes lots of experimentation).

---

## Cost Per Unit Metrics

- **Cost per episode:** $0.0093
- **Cost per GPT call:** $0.00024
- **Cost per conversation turn:** $0.00072
- **Cost per user profile evaluation:** $0.093

---

## OpenAI Tier Limits

**Tier 1 (new account):**
- Limit: $100/month
- **Status:** More than enough (full training ~$10)

**Rate limits:**
- gpt-5-nano: Likely 10,000+ TPM (tokens per minute)
- Training needs: ~200 TPM average
- **Status:** No rate limit issues expected

---

## Final Recommendation

**For Paper 1 submission:**

**Budget:** $5-7 (extremely affordable!)
**Configuration:**
- 1 full training run (1000 episodes): $1.76
- 1 medium run for comparison (500 episodes): $0.88
- Testing/debugging budget: $0.50
- 2-3 reruns with different settings: $3.50

**This gives you:**
- Robust results
- Statistical significance
- Room for multiple reruns
- Baseline comparisons
- Extensive experimentation budget

**Total time + cost:**
- Time: 7 hours + 3.5 hours = 10.5 hours
- Cost: $1.76 + $0.88 = $2.64
- **Incredibly cheap for a PhD paper!**

**Key insight:** GPT-5-nano is 10x cheaper than expected! You can run many experiments for less than the cost of a coffee.
