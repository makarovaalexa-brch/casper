# CASPER PhD TODO

**Paper 1 Deadline:** UMAP 2026 (Jan 22 abstract / Jan 29 paper)
**Innovation:** Continuous RL action space for preference elicitation (vs discrete question selection)

---

## ✅ COMPLETED - Paper 1 Implementation

**Core Systems:**
- RL actor-critic (continuous action space, DDPG-style)
- Entity space (~6,200: MovieLens Genome tags + top 5K movies + 50 people)
- Preference extraction (LLM with JSON mode)
- Question generation (entity-guided GPT)
- Two-tower recommender (background-only)
- Held-out evaluation (70/30 split, NDCG@10 delta rewards)
- User simulator with function calling (OpenAI tools)
- Baselines (RandomAgent, PureLLMAgent)
- Semantic Reddit filtering (SentenceBERT, threshold 0.6)
- End-to-end RL training notebook (Colab-ready)
- Documentation (README with Per-Turn Flow)

**Ready for experiments and paper writing!**

---

## 🔥 URGENT - Before Training

### Critical Fixes Applied ✅
- ✅ Fixed RandomAgent bug (entity_embeddings.keys() → entity_list)
- ✅ Added FAISS to dependencies (pyproject.toml)
- ✅ Confirmed gpt-5-nano is valid model (no change needed)
- ✅ Deleted dead code (user_simulator_tools_example.py)
- ✅ Standardized all prompts to .jinja format
- ✅ **GPT-5-nano compatibility fixes (CRITICAL):**
  - ✅ Changed `max_tokens` → `max_completion_tokens` (all GPT calls)
  - ✅ Removed `temperature` parameter for GPT-5 models (only supports default=1)
  - ✅ Fixed in: question_generator.py, preference_extractor.py, user_simulator.py
- ✅ Removed unused langchain imports and deleted prompts.yaml reference

### Pre-Training Tasks
- ✅ Reviewed and rewrote training notebook (now `train_casper.ipynb`)
- ✅ **Fixed PyTorch DLL issues (downgraded to 2.2.0 CPU)**
- ✅ **Created MovieLensLoader class**
- ✅ **Fixed module import structure (__init__.py files)**
- ✅ **Fixed JSON parsing in PreferenceExtractor**
- ✅ API key already set in .env file
- ✅ Dependencies installed (poetry install completed)
- [ ] Test simulator UI (chat_ui.html) (USER - optional)
- [ ] Download MovieLens 25M dataset → data/movielens/ (USER)

---

## 🎯 CRITICAL - Implementation

(All critical items complete for Paper 1)

---

## ⚙️ OPTIONAL - Future Enhancements

### User Simulator
- [ ] Review simulator quality (test with 30 diverse questions, iterate on prompts)

---

## 💡 FUTURE - Research Extensions

**Paper 2:** "Cooperative Bot-Play Training for Adaptive CRS"
- ABot-QBot cooperative training with shared reward
- Joint optimization of question asking + user simulation
- Expertise-aware response generation

**UI:**
- [ ] Agent vs agent conversation mode
- [ ] Human vs agent interactive testing
- [ ] Recommendation visualization (top-k with scores)
