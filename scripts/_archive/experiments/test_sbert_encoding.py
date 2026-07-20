"""
Sanity check for SBERT encoding of movie concepts.
Tests whether SBERT can meaningfully encode different concept types.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# Test 1: Different movie title formats
# =============================================================================
print("\n" + "="*80)
print("TEST 1: Movie Title Encoding Formats")
print("="*80)

title_formats = [
    "The Matrix",
    "The Matrix (1999)",
    "movie: The Matrix",
    "movie: The Matrix (1999)",
    "film: The Matrix (1999)",
    "I like The Matrix",
]

embeddings = model.encode(title_formats)
sim_matrix = cosine_similarity(embeddings)

print("\nSimilarity matrix for different formats of 'The Matrix':")
for i, fmt in enumerate(title_formats):
    print(f"  [{i}] {fmt}")
print("\n     " + "  ".join([f"[{i}]" for i in range(len(title_formats))]))
for i, row in enumerate(sim_matrix):
    print(f"[{i}]  " + "  ".join([f"{v:.2f}" for v in row]))

# =============================================================================
# Test 2: Similar movies should be close
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Similar Movies Should Be Close")
print("="*80)

# Sci-fi action movies
scifi_movies = [
    "movie: The Matrix (1999)",
    "movie: Inception (2010)",
    "movie: Blade Runner (1982)",
    "movie: The Terminator (1984)",
]

# Romance movies
romance_movies = [
    "movie: The Notebook (2004)",
    "movie: Titanic (1997)",
    "movie: Pride and Prejudice (2005)",
    "movie: When Harry Met Sally (1989)",
]

# Comedy movies
comedy_movies = [
    "movie: The Hangover (2009)",
    "movie: Superbad (2007)",
    "movie: Bridesmaids (2011)",
    "movie: Anchorman (2004)",
]

all_movies = scifi_movies + romance_movies + comedy_movies
labels = ["SciFi"]*4 + ["Romance"]*4 + ["Comedy"]*4

embeddings = model.encode(all_movies)
sim_matrix = cosine_similarity(embeddings)

print("\nSimilarity matrix (SciFi [0-3], Romance [4-7], Comedy [8-11]):")
print("         " + " ".join([f"{i:5d}" for i in range(len(all_movies))]))
for i, row in enumerate(sim_matrix):
    label = labels[i][:3]
    print(f"[{i:2d}]{label} " + " ".join([f"{v:5.2f}" for v in row]))

# Calculate within-group vs between-group similarity
scifi_emb = embeddings[:4]
romance_emb = embeddings[4:8]
comedy_emb = embeddings[8:12]

def avg_sim(emb1, emb2=None):
    if emb2 is None:
        emb2 = emb1
    sim = cosine_similarity(emb1, emb2)
    if emb1 is emb2:
        # Exclude diagonal for within-group
        mask = ~np.eye(len(sim), dtype=bool)
        return sim[mask].mean()
    return sim.mean()

print("\nWithin-group similarity:")
print(f"  SciFi:   {avg_sim(scifi_emb):.3f}")
print(f"  Romance: {avg_sim(romance_emb):.3f}")
print(f"  Comedy:  {avg_sim(comedy_emb):.3f}")

print("\nBetween-group similarity:")
print(f"  SciFi-Romance: {avg_sim(scifi_emb, romance_emb):.3f}")
print(f"  SciFi-Comedy:  {avg_sim(scifi_emb, comedy_emb):.3f}")
print(f"  Romance-Comedy: {avg_sim(romance_emb, comedy_emb):.3f}")

# =============================================================================
# Test 3: Genre encoding
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Genre Encoding")
print("="*80)

genres = [
    "genre: Science Fiction",
    "genre: Action",
    "genre: Romance",
    "genre: Comedy",
    "genre: Horror",
    "genre: Drama",
    "genre: Thriller",
    "genre: Animation",
]

genre_embeddings = model.encode(genres)
sim_matrix = cosine_similarity(genre_embeddings)

print("\nGenre similarity matrix:")
for i, g in enumerate(genres):
    print(f"  [{i}] {g}")
print("\n     " + "  ".join([f"[{i}]" for i in range(len(genres))]))
for i, row in enumerate(sim_matrix):
    print(f"[{i}]  " + "  ".join([f"{v:.2f}" for v in row]))

# =============================================================================
# Test 4: Concept type separation (movie vs actor vs genre)
# =============================================================================
print("\n" + "="*80)
print("TEST 4: Concept Type Separation")
print("="*80)

mixed_concepts = [
    # Movies
    "movie: The Matrix (1999)",
    "movie: Inception (2010)",
    "movie: The Godfather (1972)",
    # Actors
    "actor: Keanu Reeves",
    "actor: Leonardo DiCaprio",
    "actor: Al Pacino",
    # Genres
    "genre: Science Fiction",
    "genre: Drama",
    "genre: Crime",
    # Directors
    "director: Christopher Nolan",
    "director: Francis Ford Coppola",
    "director: The Wachowskis",
]

labels = ["Mov"]*3 + ["Act"]*3 + ["Gen"]*3 + ["Dir"]*3

embeddings = model.encode(mixed_concepts)
sim_matrix = cosine_similarity(embeddings)

print("\nMixed concept similarity (Mov [0-2], Act [3-5], Gen [6-8], Dir [9-11]):")
print("         " + " ".join([f"{i:5d}" for i in range(len(mixed_concepts))]))
for i, row in enumerate(sim_matrix):
    print(f"[{i:2d}]{labels[i]} " + " ".join([f"{v:5.2f}" for v in row]))

# =============================================================================
# Test 5: Like vs Dislike distinction (what we're trying to fix!)
# =============================================================================
print("\n" + "="*80)
print("TEST 5: Like vs Dislike Encoding (Current Problem)")
print("="*80)

like_dislike = [
    "likes: Science Fiction",
    "dislikes: Science Fiction",
    "likes: The Matrix",
    "dislikes: The Matrix",
    "likes: Action movies",
    "dislikes: Action movies",
]

embeddings = model.encode(like_dislike)
sim_matrix = cosine_similarity(embeddings)

print("\nLike vs Dislike similarity (THIS IS THE PROBLEM WE'RE FIXING):")
for i, concept in enumerate(like_dislike):
    print(f"  [{i}] {concept}")
print("\n     " + "  ".join([f"[{i}]" for i in range(len(like_dislike))]))
for i, row in enumerate(sim_matrix):
    print(f"[{i}]  " + "  ".join([f"{v:.2f}" for v in row]))

print("\n>>> Note: likes/dislikes pairs have ~0.95+ similarity!")
print(">>> This is why we need EXPLICIT one-hot rating encoding!")

# =============================================================================
# Test 6: Best format for movie titles
# =============================================================================
print("\n" + "="*80)
print("TEST 6: Movie-Genre Correlation")
print("="*80)

# Test if movies are close to their genres
test_pairs = [
    ("movie: The Matrix (1999)", "genre: Science Fiction"),
    ("movie: The Matrix (1999)", "genre: Action"),
    ("movie: The Matrix (1999)", "genre: Romance"),
    ("movie: Titanic (1997)", "genre: Romance"),
    ("movie: Titanic (1997)", "genre: Drama"),
    ("movie: Titanic (1997)", "genre: Horror"),
    ("movie: The Godfather (1972)", "genre: Crime"),
    ("movie: The Godfather (1972)", "genre: Drama"),
    ("movie: The Godfather (1972)", "genre: Comedy"),
]

for movie, genre in test_pairs:
    emb = model.encode([movie, genre])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    print(f"  {movie:35s} <-> {genre:25s} = {sim:.3f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
KEY FINDINGS:
1. Movie title format: "movie: Title (Year)" works well (0.99 sim with "film: Title (Year)")
2. Similar genre movies have WEAK clustering (within: 0.38-0.43, between: 0.37-0.39)
3. Genres cluster better than movies (within-genre sim: 0.55-0.80)
4. PROBLEM: "likes: X" vs "dislikes: X" have 0.75-0.78 similarity
   -> This confirms we MUST use explicit one-hot rating encoding!
5. Movie-genre correlation is LOW (0.10-0.30) - SBERT doesn't know Matrix = SciFi

RECOMMENDED FORMAT:
- Movies:    "movie: Title (Year)"
- Genres:    "genre: Genre Name"
- Actors:    "actor: Actor Name"
- Directors: "director: Director Name"
- Themes:    "theme: theme keyword"

For LSTM input: (SBERT_embedding, rating_one_hot) where:
- SBERT_embedding = encode("movie: Title (Year)") or encode("genre: X")
- rating_one_hot = [1,0,0] for liked, [0,1,0] for disliked, [0,0,1] for unknown
""")
