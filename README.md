# CASPER

**Continuous Action Space Preference Elicitation via Reinforcement**

**Paper:** "CASPER: Learning What to Ask through Continuous Action Space RL for Preference Elicitation in CRS"

Conversational movie recommendation system using RL to learn optimal question-asking strategies. The RL agent predicts continuous embeddings in semantic space that guide GPT question generation.

---

## Key Innovation

Instead of selecting from fixed question templates, CASPER:
1. Predicts **continuous embeddings** (384-dim SentenceBERT space)
2. Maps embeddings to movie entities (titles, actors, directors, genres)
3. Generates natural language questions via GPT
4. Learns from recommendation success using actor-critic RL

---

## Architecture

**Per-Turn Flow:**

1. **User responds** → "I love Christopher Nolan films, especially Inception"

2. **Preference Extraction** (LLM)
   - Full conversation → GPT-5-nano → Structured preferences: `{"liked": ["Nolan", "Inception", "sci-fi"]}`

3. **State Encoding** (SentenceBERT)
   - Preferences text → SentenceBERT encoder → 384-dim state vector

4. **RL Prediction** (Actor Network)
   - State vector → Actor (DDPG) → 384-dim action embedding

5. **Entity Mapping** (FAISS)
   - Action embedding → Cosine similarity search → Top-5 nearest entities: ["Tenet", "Interstellar", "cerebral", "mind-bending", "Denis Villeneuve"]

6. **Question Generation** (LLM)
   - Entities + conversation context → GPT → Natural question: "Since you enjoy Nolan's mind-bending films, have you seen Tenet?"

7. **[Background] Recommendation** (Two-Tower)
   - State vector → User tower → User embedding (128-dim)
   - Movie metadata → Item tower → Item embeddings (128-dim)
   - Dot product scores → Top-10 movies (NOT shown to user)

8. **[Background] Reward Calculation**
   - Top-10 vs held-out test set → NDCG@10
   - Reward = NDCG@10(t) - NDCG@10(t-1)

9. **Actor-Critic Update** (after episode)
   - Experiences + per-turn rewards → Update actor & critic networks

**Components:**
- **Preference Extractor**: GPT extracts structured facts from conversation
- **State Encoder**: SentenceBERT (`all-MiniLM-L6-v2`) - 384-dim
- **RL Actor**: DDPG-style actor-critic for continuous actions
- **Embedding Space**: SentenceBERT (384-dim) - handles multi-word entities, no vocab limits
- **Entity Mapper**: FAISS-powered cosine similarity for fast concept lookup
- **Question Generator**: GPT with entity-guided prompts
- **Recommender**: Two-tower architecture with BPR loss

### Conversation → Extraction → Encoding → RL Pipeline

**How user preferences flow from conversation to RL:**

**1. User Responds**
```
User: "I love Nolan films, especially Inception and Interstellar"
```

**2. Preference Extraction (LLM-based)**
- PreferenceExtractor analyzes FULL conversation history (all turns)
- Extracts structured preferences using GPT-5-nano with JSON mode
- Categories: `liked`, `neutral`, `disliked`
- Output:
```json
{
  "liked": ["Nolan", "Inception", "Interstellar", "sci-fi"],
  "neutral": [],
  "disliked": []
}
```

**3. State Encoding (SentenceBERT)**

State = **Extracted preferences only** (clean, structured)

Process:
1. LLM extracts: `{"liked": ["Nolan", "Inception"], "disliked": ["horror"]}`
2. Convert to text: `"likes: Nolan, Inception | dislikes: horror"`
3. Encode via SentenceBERT: Text → 384-dim vector

Why not include raw conversation?
- Preferences already extracted by LLM from conversation
- Avoids duplication ("Nolan" would appear in both conversation and preferences)
- Cleaner signal for RL and recommender

**4. RL Prediction**
- Actor network receives state vector
- Predicts 384-dim action embedding (continuous semantic space)
- Maps to nearest movie concepts for question generation

**Key Design:**
- Preferences are **cumulative** across all turns
- LLM extraction uses full conversation history for context
- State encoding uses only extracted preferences (no raw conversation)
- RL sees complete, clean preference state at every turn

### Embedding-to-Concept Mapping

**How RL predictions become questions:**

1. **RL predicts 384-dim embedding** in semantic space (e.g., somewhere between "thriller" and "cerebral")
2. **Entity Mapper finds top-5 nearest concepts** using FAISS similarity search:
   - Entity space includes: **1128 MovieLens Genome tags** + movie titles
   - Examples: `["thriller", "plot twist", "cerebral", "suspenseful", "thought-provoking"]`
3. **GPT generates natural question** using these concepts as guidance:
   - Prompt: "Generate a question using these concepts: thriller, plot twist, cerebral..."
   - GPT output: "Do you enjoy cerebral thrillers with unexpected plot twists?"

**Key design choices:**
- **Multiple concepts per question** (top-5): Provides rich context to GPT, but GPT decides how to combine them naturally
- **Concepts include movie titles**: RL can predict embeddings near specific movies ("Inception"), allowing questions like "Have you seen Inception?"
- **No filtering by entity type**: All entities (genres, moods, themes, movies) are available - GPT chooses what makes sense
- **FAISS for efficiency**: Handles 63K+ entities (1128 tags + 62K movies) efficiently

**Why this works:**
- GPT is good at naturally combining concepts without cramming
- If concepts are unrelated, GPT picks the most coherent subset
- RL learns which concept combinations lead to good questions through reward signal

---

## Key Design Principles

### Agent Blindness to User Profiles

**CRITICAL:** The CASPER agent is **completely blind** to user rating profiles.

**What the agent knows:**
- ✅ Only what the user **explicitly mentions** in the current conversation
- ✅ Conversation history (agent's questions + user's responses)
- ✅ Preferences extracted from user's natural language (e.g., "I love Nolan films")

**What the agent does NOT know:**
- ❌ User's rating history (even training profile)
- ❌ User's favorite movies/genres (unless mentioned in conversation)
- ❌ Any metadata about the user

**Who sees what:**
```
Agent:         BLIND to all profiles. Only sees conversation text.
User Simulator: Sees training_profile to generate realistic responses.
Test Set:      Hidden from BOTH agent and simulator.
```

**Why this matters for evaluation:**
- Agent must **discover** preferences through conversation (not cheat by reading profile)
- Simulates real-world scenario where agent knows nothing about new user
- Makes the task realistic and challenging

**Recommendation categories:**
1. **Held-out discovery** (in test set) → **Best outcome** ⭐⭐⭐ - Agent discovered hidden preference
2. **Preference discovery** (in training profile, not mentioned in conversation) → **Good outcome** ⭐⭐ - Agent inferred from conversation
3. **Already mentioned** (user said "I love Inception", agent recommends Inception) → **Redundant** ❌
4. **Not in profile** (bad recommendation) → **Failure** ❌❌

### Background-Only Recommendations

**Agent only asks questions.** Recommendations are computed in **background** for reward calculation.

**Design:**
- Agent NEVER shows recommendations to user
- After each user response, compute top-10 recommendations in background
- Calculate NDCG@10 improvement as RL reward
- Conversation continues indefinitely (no forced ending)

**Paper focus:** "Learning What to Ask" not "Learning When to Recommend"

**Per-turn flow:**
```
1. Agent asks question
2. User responds
3. [BACKGROUND] Compute top-10 recommendations
4. [BACKGROUND] Calculate NDCG@10 → RL reward
5. Repeat (recommendations never shown)
```

---

## Training

### Supervised Pretraining ✅ IMPLEMENTED
Train RL actor to predict embeddings in correct semantic space.

**Data:** Reddit conversations (r/MovieSuggestions, r/movies)
**Process:** Extract concepts from posts → Train actor to predict concept embeddings
**Loss:** MSE between predicted and target embeddings

### End-to-End RL ✅ IMPLEMENTED

Maximize recommendation success with held-out evaluation using per-turn reward calculation.

**Process:** Split user profile 70/30 → Agent asks questions → Recommend movies → Reward based on held-out movie discovery

#### Per-Turn Reward Mechanism

CASPER implements the reward mechanism from the paper (Section II.C, Equation 2):

```
r_t = NDCG@10(t) - NDCG@10(t-1)
```

**At each conversational turn:**
1. User responds to agent's question
2. Agent updates conversation state with user's response
3. **Recommendations computed** (not shown to user - for reward only)
4. **NDCG@10 calculated** against ground truth held-out movies
5. **Reward = improvement in NDCG@10** from previous turn

**Why per-turn rewards (not end-of-episode)?**
- ✅ **Immediate feedback**: Agent learns which questions improve recommendations
- ✅ **Credit assignment**: Identifies helpful vs unhelpful questions
- ✅ **Faster learning**: Don't wait until episode end for signal
- ✅ **Aligns with paper**: Matches published methodology

**Example episode:**
```python
Turn 1: User: "I love Nolan films"
        → NDCG@10 = 0.3, Reward = +0.3

Turn 2: User: "Especially Inception"
        → NDCG@10 = 0.5, Reward = +0.2 (good question!)

Turn 3: User: "I don't like horror"
        → NDCG@10 = 0.5, Reward = 0.0 (no improvement)

Turn 4: User: "I prefer cerebral sci-fi"
        → NDCG@10 = 0.7, Reward = +0.2 (good question!)
```

The RL agent learns to ask questions like Turn 2 & 4 that maximize NDCG improvement.

**Implementation:**
- `casper_agent.py:176-224` - Per-turn reward calculation in `process_user_response()`
- `casper_agent.py:262-299` - NDCG@10 calculation with DCG/IDCG
- `casper_agent.py:301-362` - Training with per-turn rewards in `train_from_episode()`

---

## User Simulator

Uses MovieLens ratings + GPT to generate realistic user responses for training/evaluation.

**Features:**
- Samples real user profiles (liked/disliked movies, genres)
- Infers personality from rating patterns (generous vs critical, enthusiast vs casual)
- GPT generates natural responses matching profile + personality
- Supports held-out evaluation (hide 30% of movies, test if agent discovers them)

**Integration:**
```python
user = simulator.sample_user()
question = agent.generate_question(state)
response = simulator.simulate_response(question, user)
```

---

## Evaluation

**Primary Metric:** Held-out evaluation
- Hide 30% of user's liked movies
- Agent converses using 70% training movies
- Success = Recommending hidden movies in top-k

**Metrics:**
- **Success@5, Success@10**: % held-out movies in top-k recommendations
- **Questions to Success**: Avg questions before first successful recommendation
- **Efficiency Score**: Balances quality + conversation length

**Baselines:**
- **RandomAgent** - Random entity selection (floor performance)
- **PureLLMAgent** - GPT without RL actor (shows value of strategic learning)
- For publication: Compare to published CRS papers (KBRD, UNICORN, etc.)

---

## Quick Start

### 1. Install
```bash
cd casper
poetry install
```

### 2. Train on Colab (GPU)
Upload `train_colab.ipynb` to Google Colab.

### 3. Local Training
```bash
# Sample data
poetry run python src/casper/data/reddit_scraper.py --sample --output-dir data/reddit

# Train RL actor (supervised pretraining)
poetry run python src/casper/training/rl_actor_pretraining.py --reddit-data data/reddit --epochs 10

# Test
poetry run python src/casper/models/embedding_casper.py
```

---

## Project Structure

```
casper/
├── src/casper/
│   ├── models/
│   │   ├── embedding_casper.py            # RL actor-critic + question generation
│   │   └── two_tower_recommender.py       # Recommendation model
│   ├── agents/
│   │   └── user_simulator.py              # MovieLens-based user simulation
│   ├── utils/
│   │   └── preference_extractor.py        # GPT preference extraction
│   ├── data/
│   │   ├── reddit_scraper.py              # Reddit data collection
│   │   ├── reddit_loader.py               # Reddit data loader
│   │   └── movielens_downloader.py        # MovieLens dataset downloader
│   ├── training/
│   │   └── rl_actor_pretraining.py        # Supervised pretraining
│   └── evaluation/
│       └── conversation_evaluator.py      # Held-out evaluation
├── data/
│   ├── movielens/                         # MovieLens 25M dataset
│   ├── reddit/                            # Scraped conversations
│   └── processed/                         # Preprocessed data
├── experiments/                            # Experiment results
├── train_colab.ipynb                      # GPU training notebook
└── docs/                                  # Additional documentation
```

---

## Implementation Status

### ✅ Completed
- SentenceBERT embedding space (384-dim)
- Actor-critic RL agent (DDPG-style)
- GPT question generator with entity guidance
- Two-tower recommender with BPR loss
- MovieLens user simulator with personality inference
- Held-out evaluation framework
- Reddit data scraper
- Supervised pretraining
- Colab training notebook

### ⏳ In Progress
- End-to-end RL training
- Large-scale experiments (100+ users)
- Baseline implementations

### 📋 Planned
- User study with real participants
- Multi-domain extension (books, music)
- Cross-session preference tracking
- Production deployment

---

## Research Contribution

**Research Gap Addressed:**
No prior work combines RL continuous action spaces + LLM question generation + preference elicitation. Current systems use discrete question selection or pure LLM without strategic learning.

**First system to:**
- Use continuous RL in semantic embedding space for conversational recommendation
- Bootstrap RL with pretrained SentenceBERT knowledge
- Train on real question-asking patterns from Reddit
- Combine actor-critic RL + GPT generation in shared semantic space

**vs. Discrete Approaches:**
- Smoother optimization landscape
- Novel concept combinations via embedding interpolation
- Better generalization (no vocabulary limits vs 80 fixed concepts)
- Still interpretable (maps back to entities)

**vs. Pure LLM:**
- Strategic question selection learned via RL
- Learns from recommendation feedback
- Consistent improvement over time

---

## Citation

```bibtex
@software{casper_2025,
  title={CASPER: Continuous Action Space Preference Elicitation via Reinforcement for Conversational Recommendation},
  author={Makarova, Aleksandra},
  year={2025},
  url={https://github.com/makarovaalexa-brch/casper}
}
```

---

## License

MIT License - see LICENSE file for details.
