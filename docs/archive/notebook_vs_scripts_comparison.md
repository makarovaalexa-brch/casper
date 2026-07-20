# Comprehensive Comparison: Original CSMAI-19 Notebook vs Training Scripts

This document details all differences between the original CSMAI-19 notebook (`CSMAI_19_MovieLens_Dataset_+_attributes.ipynb`) and the current training scripts in `casper/scripts/`.

## 1. Data Configuration

| Parameter | Original Notebook | Scripts | Match? |
|-----------|------------------|---------|--------|
| Movies | 100 (top by rating count) | 100 (top by rating count) | Yes |
| Users | ~37,139 (>=25 ratings on top 100) | ~64,639 (>=25 ratings on top 100) | **No** |
| Actors | **50** | **50** | Yes |
| Directors | **10** | **20** | **No** |
| Genres | ~19 (from data) | ~19 (from data) | Yes |
| Keywords | **Yes** (>=20 movie appearances) | **No** | **No** |
| Tags | No | No | Yes |

### Key Differences:

1. **Directors**: Notebook uses 10 directors, scripts use 20. Line in notebook:
   ```python
   popular_directors = get_popular_names(credits=credits, target_col='crew', n=10)
   ```

2. **Keywords**: Notebook includes keywords (terms appearing in >=20 movie descriptions) as separate items. Scripts don't use keywords at all:
   ```python
   # Notebook
   popular_keywords = get_popular_keywords(keywords)  # Items appearing in >=20 movies
   keyword_features = ohe_popular_attributes(...)
   ```

3. **User Count Discrepancy**: The notebook quotes 37,139 users in the paper, but my scripts are finding ~64,639 users meeting the criteria. This suggests:
   - Either the filtering criteria differs slightly
   - Or the MovieLens dataset version is different (25M vs 20M)

## 2. Model Architecture

### One-Hot Model (ExtrapolationModel)

| Component | Original Notebook | Scripts | Match? |
|-----------|------------------|---------|--------|
| Embedding dimension | `n_items // 2` | `n_items // 2` | Yes |
| LSTM input | `embedding_dim + 3` | `embedding_dim + 3` | Yes |
| LSTM hidden | `n_items // 2` | `n_items // 2` | Yes |
| LSTM layers | 1 | 1 | Yes |
| LSTM bidirectional | No | No | Yes |
| Dense1 | `hidden -> n_items` | `hidden -> n_items` | Yes |
| Dense2 | `n_items -> hidden` | `n_items -> hidden` | Yes |
| Attention | MultiheadAttention (1 head) | MultiheadAttention (1 head) | Yes |
| **Output dimension** | **`n_items * 2`** | **`n_items`** | **No** |

### Output Layer Difference (Critical):

**Notebook (line 869)**:
```python
self.output = nn.Linear(y_n, y_n*2)  # explicit and implicit model outputs stacked together
```

**Scripts**:
```python
self.output = nn.Linear(hidden_dim * 2, n_items)  # Only n_items outputs
```

The notebook outputs **explicit + implicit** predictions:
- First `n_items` outputs: explicit rating predictions
- Second `n_items` outputs: implicit (whether item was seen or not)

The scripts only output explicit predictions. This is a significant architectural difference that affects:
- Loss calculation (notebook masks both explicit and implicit)
- Model capacity (2x output neurons)

### Rating Encoding

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Rating threshold | >= 4.0 = positive | >= 4.0 = positive | Yes |
| Encoding format | One-hot [disliked, liked, not_seen] | One-hot [disliked, liked, not_seen] | Yes |
| Encoding dimension | 3 | 3 | Yes |

## 3. Training Configuration

| Parameter | Original Notebook | Scripts | Match? |
|-----------|------------------|---------|--------|
| **Batch size** | **1** | **32** | **No** |
| Learning rate | 0.001 | 0.001 | Yes |
| Optimizer | Adam | Adam | Yes |
| Epochs | 100 | 100 | Yes |
| Train/Val split | 80/20 | 80/20 | Yes |

### Batch Size Impact:

The notebook uses batch_size=1, meaning each user is processed individually. This is extremely slow but may have different gradient behavior:
- Notebook: Pure stochastic gradient descent per user
- Scripts: Mini-batch gradient descent (32 users averaged)

Line in notebook:
```python
BATCH_SIZE = 1
```

## 4. Loss Function

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Base loss | BCEWithLogitsLoss | BCEWithLogitsLoss | Yes |
| NaN masking | Yes (both explicit & implicit) | Yes (explicit only) | Partial |

The notebook masks NaN values in both the explicit and implicit output dimensions, while scripts only handle single n_items output.

## 5. Data Preprocessing

### Movie Selection

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Selection criteria | Most frequently rated | Most frequently rated | Yes |
| Count | 100 | 100 | Yes |

### Attribute Processing

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Actor selection | Top 50 by appearance count | Top 50 by appearance count | Yes |
| Actor limit per movie | Not specified in visible code | 5 actors per movie | Unclear |
| Director selection | Top 10 by appearance | Top 20 by appearance | **No** |
| Keyword selection | >=20 movie appearances | **Not used** | **No** |
| Genre processing | One-hot encoded | One-hot encoded | Yes |

## 6. Input/Output Format

### Input

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Item indices | Integer indices | Integer indices | Yes |
| Rating encoding | 3-dim one-hot | 3-dim one-hot | Yes |
| Sequence order | Shuffled per user | Shuffled per user | Yes |

### Output

| Aspect | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Format | **`(seq_len, n_items * 2)`** | `(seq_len, n_items)` | **No** |
| Content | Explicit + Implicit stacked | Explicit only | **No** |

## 7. Evaluation Metrics

| Metric | Original Notebook | Scripts | Match? |
|--------|------------------|---------|--------|
| Loss tracking | BCE loss | BCE loss | Yes |
| NDCG calculation | Manual implementation | Custom implementation | Similar |
| Accuracy | Masked accuracy (ignores NaN) | Not computed during training | **No** |

## 8. Total Items Comparison

| Component | Original Notebook | Scripts | Difference |
|-----------|------------------|---------|------------|
| Movies | 100 | 100 | 0 |
| Genres | ~19 | ~19 | 0 |
| Actors | 50 | 50 | 0 |
| Directors | **10** | **20** | **+10** |
| Keywords | **~X** (varies) | **0** | **-X** |
| **Total Items** | **~137** (estimated) | **~189** | **~-52** |

The paper mentions 137 items, which suggests roughly:
- 100 movies + 19 genres + 10 directors + ~8 keywords = ~137

Our scripts have:
- 100 movies + 19 genres + 50 actors + 20 directors = ~189

## 9. Summary of Critical Differences

### Must Fix (Significant Impact):

1. **Output Dimension**: Notebook outputs `n_items * 2` (explicit + implicit), scripts output `n_items`. This fundamentally changes the model's learning objective.

2. **Keywords Missing**: Notebook uses keywords as items, scripts don't. This changes the item space.

3. **Directors Count**: 10 vs 20 - affects item space size.

### Consider Fixing (Moderate Impact):

4. **Batch Size**: 1 vs 32 - affects gradient updates and possibly convergence behavior.

### Acceptable Differences:

5. **User count variations**: May be due to dataset version differences.

## 10. Recommendations

1. **For exact replication**:
   - Change output to `n_items * 2`
   - Add implicit prediction handling
   - Reduce directors to 10
   - Add keyword items
   - Set batch_size=1 (very slow)

2. **For practical comparison**:
   - Current setup is reasonable for comparing models against each other
   - All models use same configuration, so relative comparisons are valid
   - The RL signal measurement should still work as intended

3. **For the paper**:
   - Note the configuration differences in methodology
   - Emphasize that all models were trained with identical setups for fair comparison
