"""LLM answerability FUEL-GATE micro-pilot.
Question: does GPT-5.4-mini, reading HALF a user's ML-1M profile, judge answerability in a way that
(a) varies across users and (b) tracks taste? Cheap eyeball read (~15 users x ~20 q, <$1).
Run: python scripts/llm_answerability_pilot.py
"""
import os, re, json, random, collections
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("PILOT_MODEL", "gpt-5.4-mini")
random.seed(0)
ML = "data/movielens/ml-1m"

# ---- load ML-1M ----
movies = {}   # id -> (title, [genres])
for line in open(f"{ML}/movies.dat", encoding="latin-1"):
    mid, title, genres = line.rstrip("\n").split("::")
    movies[int(mid)] = (title, genres.split("|"))
pop = collections.Counter()
uratings = collections.defaultdict(list)  # uid -> [(mid,rating)]
for line in open(f"{ML}/ratings.dat", encoding="latin-1"):
    u, m, r, _ = line.split("::"); u, m, r = int(u), int(m), int(r)
    uratings[u].append((m, r)); pop[m] += 1
maxpop = max(pop.values())
def poptier(mid):
    p = pop.get(mid, 0)/maxpop
    return "famous" if p > 0.3 else "moderate" if p > 0.05 else "obscure"

# ---- pick ~15 users with DIVERSE dominant genres (so taste-tracking is visible) ----
def domgenre(uid):
    g = collections.Counter()
    for m, r in uratings[uid]:
        if r >= 4:
            for gg in movies.get(m, ("", []))[1]: g[gg] += 1
    return g.most_common(1)[0][0] if g else None
cand = [u for u in uratings if 40 <= len(uratings[u]) <= 300]
random.shuffle(cand)
by_g = collections.defaultdict(list)
for u in cand:
    dg = domgenre(u)
    if dg: by_g[dg].append(u)
users = []
for g in ["Horror", "Romance", "Children's", "Film-Noir", "Documentary", "Action", "Comedy", "Sci-Fi"]:
    if by_g.get(g): users.append((by_g[g][0], g))
users = users[:8]  # keep pilot tiny/cheap
print(f"pilot users ({len(users)}):", [(u, g, len(uratings[u])) for u, g in users])

# ---- fixed question bank: broad -> niche concepts + tiered items + pairs ----
CONCEPTS = [  # (text, breadth)
    ("comedies", "broad"), ("horror films", "broad"), ("romance films", "broad"),
    ("action movies", "broad"), ("documentaries", "mid"), ("film noir", "niche"),
    ("Scandinavian crime dramas", "niche"), ("silent films", "niche"),
    ("slow-burn arthouse cinema", "niche"), ("1970s political thrillers", "niche"),
]
# items across popularity tiers (fixed, well/less known)
def pick_items():
    fam = [m for m in pop if poptier(m) == "famous"]
    obs = [m for m in pop if poptier(m) == "obscure"]
    random.shuffle(fam); random.shuffle(obs)
    return [(m, "famous") for m in fam[:4]] + [(m, "obscure") for m in obs[:4]]
ITEMS = pick_items()

SYS = ("You simulate judging what a specific movie viewer could answer in a cold-start interview. "
       "You see PART of their rating history. For each question, decide whether THIS viewer could give "
       "a meaningful answer (not whether they'd like it). Answer strictly as JSON list, one object per "
       'question: {"q":<idx>,"can_answer":"yes"|"no"|"maybe","conf":0-1}. No prose.')

def profile_text(uid):
    # show HALF the profile (random), titles+ratings, no ids
    rs = uratings[uid][:]; random.shuffle(rs)
    half = rs[:len(rs)//2]
    liked = [f"{movies[m][0]} ({r}/5)" for m, r in half if m in movies]
    return "; ".join(liked[:40])

def build_questions():
    qs = []
    for c, b in CONCEPTS: qs.append(("concept", c, b, f"Do you like {c}?"))
    for m, t in ITEMS: qs.append(("item", movies[m][0], t, f"Did you like the film '{movies[m][0]}'?"))
    return qs
QS = build_questions()

results = {}
for uid, g in users:
    prof = profile_text(uid)
    qlist = "\n".join(f"{i}: {q[3]}" for i, q in enumerate(QS))
    msg = f"Viewer's partial history (liked/rated films):\n{prof}\n\nQuestions:\n{qlist}"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
            response_format={"type": "json_object"} if "5.4" not in MODEL else None,
        )
    except Exception as e:
        # some models want max_completion_tokens / no temperature; retry minimal
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
        )
    txt = resp.choices[0].message.content
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        arr = json.loads(m.group(0)) if m else json.loads(txt)
    except Exception:
        print(f"  PARSE FAIL user {uid}: {txt[:200]}"); continue
    ans = {int(o["q"]): o for o in arr if "q" in o}
    results[uid] = {"genre": g, "ans": ans}
    yes = sum(1 for o in ans.values() if o.get("can_answer") == "yes")
    print(f"user {uid} [{g}]: {yes}/{len(ans)} yes")

# ---- eyeball summary: does niche-concept answerability track taste? ----
print("\n=== NICHE-CONCEPT answerability by user (the taste-tracking read) ===")
niche_idx = [i for i, q in enumerate(QS) if q[0] == "concept" and q[2] == "niche"]
hdr = "user/genre".ljust(22) + " ".join(QS[i][1][:14].ljust(15) for i in niche_idx)
print(hdr)
for uid, g in users:
    if uid not in results: continue
    row = f"{uid}/{g}".ljust(22)
    for i in niche_idx:
        o = results[uid]["ans"].get(i, {})
        row += (o.get("can_answer", "?")[:3]).ljust(15)
    print(row)

print("\n=== ITEM answerability by popularity tier (sanity: famous>obscure) ===")
for tier in ["famous", "obscure"]:
    idxs = [i for i, q in enumerate(QS) if q[0] == "item" and q[2] == tier]
    rate = []
    for uid, g in users:
        if uid not in results: continue
        a = results[uid]["ans"]
        yes = sum(1 for i in idxs if a.get(i, {}).get("can_answer") == "yes")
        rate.append(yes/len(idxs) if idxs else 0)
    print(f"  {tier}: mean yes-rate {sum(rate)/len(rate):.2f}" if rate else f"  {tier}: -")

json.dump({str(u): results[u] for u in results}, open("experiments/llm_pilot_results.json", "w"), indent=1)
print("\nsaved experiments/llm_pilot_results.json ; model:", MODEL)
