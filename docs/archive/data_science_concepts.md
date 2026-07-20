# Data Science Concepts in CASPER

Quick reference for understanding the ML/DS techniques used in CASPER.

---

## Embeddings & Semantic Space

### What are Embeddings?

**Embeddings** convert text/concepts into numerical vectors that capture semantic meaning.

**Example:**
```
"action movie"     → [0.2, -0.5, 0.8, ..., 0.3]  (384 numbers)
"thriller film"    → [0.3, -0.4, 0.7, ..., 0.4]  (similar vector!)
"romantic comedy"  → [-0.5, 0.8, -0.2, ..., -0.1] (very different)
```

**Key insight:** Similar concepts have similar vectors. We can measure similarity with **cosine similarity**.

```python
cosine_similarity("action movie", "thriller") = 0.85  # Very similar
cosine_similarity("action movie", "romance")  = 0.12  # Very different
```

### SentenceBERT

**SentenceBERT** is a pre-trained neural network that converts any text into a 384-dimensional vector.

**In CASPER:**
- Model: `all-MiniLM-L6-v2` (configured in `config.yaml`)
- Embedding dimension: 384
- Usage:
  - Encode conversation state → 384-dim vector
  - Encode movie entities → 384-dim vectors
  - RL predicts embeddings in this same space

**Why SentenceBERT?**
- Pre-trained on billions of text pairs
- Captures semantic meaning automatically
- No manual feature engineering
- Handles any text (no vocabulary limits)

**Example in code:**
```python
encoder = SentenceTransformer('all-MiniLM-L6-v2')
embedding = encoder.encode("I love Christopher Nolan films")
# Returns: np.array([0.23, -0.41, 0.67, ...])  # 384 numbers
```

**Configuration:**
```yaml
# config/config.yaml
models:
  embedding_space:
    model_name: all-MiniLM-L6-v2  # Can be changed to other models
    embedding_dim: 384
```

---

## Reinforcement Learning

### What is RL?

**Reinforcement Learning** trains an agent to make good decisions by learning from rewards.

**Components:**
- **State**: Current situation (e.g., conversation history)
- **Action**: What to do (e.g., which question to ask)
- **Reward**: Feedback signal (e.g., NDCG improvement)
- **Policy**: Strategy for choosing actions

**Training loop:**
```
1. Agent observes state
2. Agent takes action
3. Environment gives reward
4. Agent learns from reward
5. Repeat
```

### Actor-Critic Architecture

Two neural networks that work together:

**Actor Network:**
- Learns **which actions to take**
- Input: State (384-dim conversation embedding)
- Output: Action (384-dim concept embedding)
- "What should I ask about?"

**Critic Network:**
- Learns **how good actions are**
- Input: State + Action
- Output: Q-value (expected future reward)
- "Is this a good question to ask?"

**Training:**
```
Critic: "That question would get reward of 0.7"
Actor:  "I'll learn to ask more questions like that!"

Critic: "That question would get reward of 0.1"
Actor:  "I'll avoid questions like that"
```

### DDPG (Deep Deterministic Policy Gradient)

**Problem:** Most RL algorithms work with **discrete actions** (e.g., pick option 1, 2, or 3).

CASPER needs **continuous actions** (a 384-dimensional embedding vector with infinite possibilities).

**DDPG Solution:**
- Designed for continuous action spaces
- Actor directly outputs action vector (not probabilities)
- Critic evaluates continuous actions
- Uses gradient ascent to improve actor

**In CASPER:**
```python
# Actor predicts embedding
state = encode_conversation()  # 384-dim
action_embedding = actor(state)  # 384-dim (continuous!)

# Critic evaluates it
q_value = critic(state, action_embedding)  # scalar

# Training
actor_loss = -q_value.mean()  # Maximize Q-value
critic_loss = (q_value - target_q)^2  # Predict accurate Q-values
```

**Why DDPG?**
- Handles continuous actions (embeddings)
- Stable training with replay buffer
- Exploration via noise addition
- Well-suited for semantic space navigation

**CASPER-specific details:**
```python
# rl_actor_critic.py
class EmbeddingActorCritic:
    def __init__(self):
        # Actor: State → Embedding
        self.actor = nn.Sequential(
            nn.Linear(384, 512),  # Hidden layer
            nn.ReLU(),
            nn.Linear(512, 384),  # Output embedding
            nn.Tanh()  # Normalize to [-1, 1]
        )

        # Critic: State + Embedding → Q-value
        self.critic = nn.Sequential(
            nn.Linear(384 + 384, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Q-value
        )
```

**Training flow:**
```
Turn 1: Ask about "Nolan" → User responds → NDCG = 0.3 → Reward = +0.3
Turn 2: Ask about "sci-fi" → User responds → NDCG = 0.5 → Reward = +0.2
Turn 3: Ask about "horror" → User responds → NDCG = 0.5 → Reward = 0.0

Actor learns: "Asking about Nolan and sci-fi was good! Avoid horror."
```

---

## Recommendation Systems

### Two-Tower Recommender

**Two-Tower** is a neural architecture that learns to match users to items by encoding them in the same embedding space.

**Architecture:**
```
User Tower:                    Item Tower:
Conversation State (384-dim)   Movie Features (384-dim)
       ↓                              ↓
   Neural Net                     Neural Net
       ↓                              ↓
  User Embedding (128-dim)      Movie Embedding (128-dim)
       ↓                              ↓
       └──────── Dot Product ─────────┘
                     ↓
                 Score (scalar)
```

**Why "Two-Tower"?**
- User and item are processed **separately** (two independent towers)
- Both output embeddings in the **same 128-dim space**
- At inference, compute dot product between user and all movies
- Higher score = better match

**Key Advantage:**
Can pre-compute movie embeddings once, then just encode user state at runtime. Fast inference.

**In CASPER:**

**Input to User Tower:**
```python
# Conversation state = Extracted preferences only (no raw conversation)
# LLM extracts: {"liked": ["Nolan", "Inception"], "disliked": ["horror"]}
# Convert to text:
state_text = "likes: Nolan, Inception | dislikes: horror"
state_vector = encoder.encode(state_text)  # 384-dim from SentenceBERT
user_embedding = user_tower(state_vector)  # 128-dim
```

**Input to Item Tower:**
```python
# Movie metadata
movie_text = "Inception: Action, Sci-Fi, Thriller"
movie_vector = encoder.encode(movie_text)  # 384-dim from SentenceBERT
movie_embedding = item_tower(movie_vector)  # 128-dim
```

**Recommendation:**
```python
# Score all movies
scores = user_embedding @ movie_embeddings.T  # Dot product
top_k = argsort(scores)[:10]  # Top-10 recommendations
```

**Training: BPR Loss**

**BPR (Bayesian Personalized Ranking)** trains the model to rank liked movies higher than random movies.

**Loss formula:**
```
L = -log(sigmoid(score_positive - score_negative))
```

**Training batch:**
```python
user_state = "User likes Nolan films"
positive_movie = "Inception" (user liked this)
negative_movie = "Random movie" (not in user's likes)

pos_score = dot(user_tower(state), item_tower(positive_movie))  # Should be high
neg_score = dot(user_tower(state), item_tower(negative_movie))  # Should be low

# Loss encourages: pos_score > neg_score
loss = -log(sigmoid(pos_score - neg_score))
```

**Example:**
```
User state: "I love Christopher Nolan films"
Positive movie: "Inception" → score = 0.8
Negative movie: "Romantic Comedy XYZ" → score = 0.2

BPR loss = -log(sigmoid(0.8 - 0.2)) = -log(sigmoid(0.6)) = 0.38

After training, gap increases:
Positive movie: "Inception" → score = 0.95
Negative movie: "Romantic Comedy XYZ" → score = 0.05
BPR loss = -log(sigmoid(0.9)) = 0.15 (lower is better)
```

**Used in CASPER for:**
- Background reward calculation only (NOT shown to users)
- Computes top-10 recommendations after each turn
- NDCG@10 calculated against ground truth
- Reward = NDCG improvement guides RL training

**Training data:**
- Simulated conversation preferences from MovieLens users
  - User simulator: Takes a MovieLens user profile (their ratings)
  - Simulates what they might say: "I love action films, especially Nolan"
  - Paired with their actual liked movies from profile
- Example training pair:
  - Input: "likes: Nolan, action films | dislikes: romance"
  - Positive movie: Movie rated 5★ by this user
  - Negative movie: Random movie not in their likes
- Trained from scratch (not pretrained)

---

## Evaluation Metrics

### NDCG (Normalized Discounted Cumulative Gain)

Measures **ranking quality** for recommendations.

**Intuition:** Good recommendations should put relevant items at the top.

**Example:**
```
Recommended:        [Movie A, Movie B, Movie C, Movie D, Movie E]
User likes (truth): [Movie B, Movie D]

Position 1 (A): Not liked → 0 relevance
Position 2 (B): Liked! → 1 relevance (discounted by log₂(3) = 1.58)
Position 3 (C): Not liked → 0 relevance
Position 4 (D): Liked! → 1 relevance (discounted by log₂(5) = 2.32)
Position 5 (E): Not liked → 0 relevance

DCG = 0 + 1/1.58 + 0 + 1/2.32 + 0 = 0.63 + 0.43 = 1.06

Ideal ranking:     [Movie B, Movie D, Movie A, Movie C, Movie E]
IDCG = 1/1 + 1/1.58 = 1.63

NDCG = DCG / IDCG = 1.06 / 1.63 = 0.65
```

**Interpretation:**
- NDCG = 1.0: Perfect ranking (all relevant items at top)
- NDCG = 0.65: Decent (some relevant items near top)
- NDCG = 0.0: Terrible (no relevant items in list)

**Why NDCG@10?**
- Only top-10 recommendations matter
- Position matters (item at #1 is more valuable than #10)
- Standard metric for recommendation systems

**In CASPER:**
```python
# Compute NDCG after each turn
recommendations = recommender.recommend(state, top_k=10)
ndcg = calculate_ndcg(recommendations, user_target_movies, k=10)

# Reward = improvement
reward = ndcg_current - ndcg_previous
```

---

## FAISS (Fast Similarity Search)

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional spaces.

**Problem:** Finding nearest neighbors in 384-dim space with 60,000+ entities is slow with brute force.

**FAISS Solution:** Optimized algorithms + GPU acceleration

**In CASPER:**
```python
# Build index once
entity_embeddings = encoder.encode(all_entities)  # (60000, 384)
faiss_index = faiss.IndexFlatIP(384)  # Inner product (cosine similarity)
faiss_index.add(entity_embeddings)

# Fast search at runtime
predicted_embedding = rl_actor.predict(state)  # (384,)
scores, indices = faiss_index.search(predicted_embedding, k=5)
nearest_entities = [all_entities[i] for i in indices]
# Returns: ["Inception", "Nolan", "sci-fi", "cerebral", "plot twist"]
```

**Speed:**
- Brute force: ~100ms for 60K entities
- FAISS: ~1ms for 60K entities

---

## CASPER Training Paradigm

### Overview: What Gets Trained?

CASPER uses a mix of **pretrained** and **trained-from-scratch** models:

| Component | Status | Training Data | When Trained |
|-----------|--------|---------------|--------------|
| **SentenceBERT** | PRETRAINED ✓ | Billions of text pairs (HuggingFace) | Never (frozen) |
| **GPT (Question Gen)** | PRETRAINED ✓ | OpenAI training data | Never (API call) |
| **GPT (Pref Extraction)** | PRETRAINED ✓ | OpenAI training data | Never (API call) |
| **RL Actor-Critic** | TRAINED | Stage 1: Reddit conversations<br>Stage 2: RL with NDCG rewards | During training |
| **Two-Tower Recommender** | TRAINED | MovieLens user profiles | Before RL training |

### Training Pipeline

**Phase 1: Recommender Training (One-time)**
```
MovieLens data → Two-Tower Recommender
  - Input: User profiles (ratings)
  - Output: Trained recommender model
  - Used for: NDCG reward calculation during RL
```

**Phase 2: Stage 1 - Supervised Pretraining (RL Actor)**
```
Reddit conversations → RL Actor pretraining
  - Extract concepts from posts
  - Train actor to predict concept embeddings
  - Why: Start with reasonable predictions (not random)
```

**Phase 3: Stage 2 - RL Training (End-to-End)**
```
User Simulator → Agent → Recommender → Rewards → RL Update
  - Agent asks questions guided by RL
  - User responds (simulator)
  - Extract preferences
  - Compute recommendations (background)
  - Calculate NDCG → Reward
  - Update RL policy
```

### Training Stages in Detail

#### Stage 1: Supervised Pretraining (RL Actor)

Train RL actor to predict embeddings from conversation context **before** RL training.

**Why?** Start with reasonable predictions instead of random noise.

**Data:** Reddit movie conversations
**Loss:** MSE between predicted and target embeddings

```python
# Extract concept from Reddit post
post = "I love Nolan films, especially Inception"
target_embedding = encoder.encode("Inception")

# Train actor
predicted_embedding = actor(encode_conversation(post))
loss = MSE(predicted_embedding, target_embedding)
```

**Status:** Implemented in `src/casper/training/stage1_supervised.py`

#### Stage 2: End-to-End RL

Train agent to ask questions that maximize recommendation success.

**Per-Turn Rewards:**

Give RL feedback **after every turn**, not just at end of conversation.

**Traditional RL:** Reward only at episode end
```
Turn 1, 2, 3, ... → Final reward = 0.7 → ??? Which turn was good?
```

**Per-Turn RL (CASPER):**
```
Turn 1 → NDCG = 0.3 → Reward = +0.3  ✓ Good question!
Turn 2 → NDCG = 0.5 → Reward = +0.2  ✓ Good question!
Turn 3 → NDCG = 0.5 → Reward = 0.0   ✗ Didn't help
```

**Benefits:**
- Faster learning (immediate feedback)
- Better credit assignment (know which questions helped)
- Aligns with paper methodology

**Full Episode Flow:**
```python
# Initialize
agent = CASPERAgent()
user = simulator.sample_user()  # MovieLens profile split 70/30

# Episode loop
for turn in range(max_turns):
    # 1. Agent asks question
    question = agent.ask_question()

    # 2. User responds
    response = user.respond(question)

    # 3. Extract preferences
    agent.process_user_response(response, user.target_movies)

    # 4. [BACKGROUND] Compute recommendations
    recommendations = agent.recommend_movies(top_k=10)

    # 5. [BACKGROUND] Calculate NDCG
    ndcg = calculate_ndcg(recommendations, user.target_movies)

    # 6. [BACKGROUND] Compute reward
    reward = ndcg - previous_ndcg

    # 7. Store experience for RL
    agent.rl_agent.store_experience(state, action, reward, next_state, done)

# 8. Train RL after episode
agent.train_from_episode()
```

**Status:** Implemented in `src/casper/agents/casper_agent.py`

### Model Dependencies

```
SentenceBERT (frozen)
    ↓ (encodes conversations & movies)
    ├─→ RL Actor-Critic (trained)
    ├─→ Two-Tower Recommender (trained)
    ├─→ Preference Extractor (uses GPT-5-nano)
    └─→ Question Generator (uses GPT-5-nano)

Training order:
1. SentenceBERT: Already pretrained ✓
2. Two-Tower: Train on MovieLens data
3. RL Actor: Stage 1 pretraining on Reddit
4. RL Actor: Stage 2 RL training with recommender
```

### Joint Training?

**No joint training.** Models are trained separately:

1. **Two-Tower** trained independently on MovieLens
   - Frozen during RL training
   - Only used for reward calculation

2. **RL Actor-Critic** trained with Two-Tower feedback
   - Does NOT update Two-Tower weights
   - Only updates actor/critic networks

**Why separate?**
- Two-Tower learns general movie recommendations
- RL learns question-asking strategy
- Simpler training, more stable
- Can swap out recommender without retraining RL

---

## Common Patterns in CASPER

### State Encoding Pattern
```python
# Clean and simple: Extracted preferences only
# LLM extracts from conversation → Structured data → Text for encoding
preferences = {"liked": ["Nolan", "Inception"], "disliked": ["horror"]}
state_text = "likes: Nolan, Inception | dislikes: horror"
state_vector = encoder.encode(state_text)  # 384-dim

# Why not include raw conversation?
# - Preferences already extracted by LLM
# - Avoids duplication ("Nolan" would appear twice)
# - Cleaner signal for RL
```

### Preference Extraction Pattern
```python
# LLM extracts structured preferences
conversation = ["Agent: ...", "User: I love Nolan", ...]
preferences = preference_extractor.extract(conversation)
# Returns: {"liked": ["Nolan"], "neutral": [], "disliked": []}
```

### Question Generation Pattern
```python
# RL → Entities → GPT → Question
state = encode_conversation()
embedding = rl_actor.predict(state)
entities = find_nearest(embedding, top_k=3)  # ["Nolan", "sci-fi", "Inception"]
question = gpt.generate(entities, conversation, preferences)
# Returns: "Since you enjoy Nolan, have you seen Tenet?"
```

---

## Configuration is Key

All model hyperparameters live in `config/config.yaml`:

```yaml
models:
  embedding_space:
    model_name: all-MiniLM-L6-v2  # SentenceBERT model
    embedding_dim: 384

  actor_critic:
    state_dim: 384
    embedding_dim: 384
    hidden_dim: 512
    learning_rate: 0.001
    exploration_noise: 0.2

  question_generator:
    model_name: gpt-5-nano
    temperature: 0.7

  preference_extractor:
    model_name: gpt-5-nano
    temperature: 0.3
```

**Never hardcode!** Load from config for easy experimentation.
