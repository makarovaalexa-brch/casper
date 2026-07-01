"""Blind geometric-answer helper for the adaptive LLM asker. Usage:
  python llm_answer.py <user_idx> "<entry>"   (entry = 'CONCEPT: x' or 'MOVIE: Title (Year)')
Returns the user's like/dislike/haven't-seen answer (the agent never sees the underlying taste).
SAVES EVERYTHING: every query (answerable or not) -> llm_log_<tag>_<user>.jsonl ; answerable folds (emb,val)
-> llm_traj_<tag>_<user>.jsonl (the authoritative record used for scoring; the agent's self-report is ignored)."""
import sys, os, pickle, json
base='C:/dev/phd/casper/data/movielens'
d=pickle.load(open(f'{base}/.cache/llm_data.pkl','rb'))
ui=int(sys.argv[1]); entry=' '.join(sys.argv[2:]).strip().strip('"').strip()
u=d['users'][ui]; tag=os.environ.get("TAG","haiku")
trajf=f'{base}/.cache/llm_traj_{tag}_{ui}.jsonl'; logf=f'{base}/.cache/llm_log_{tag}_{ui}.jsonl'
def emit(answerable,answer,entry_norm,emb=None,val=None,like=None):
    rec={'entry':entry_norm,'raw_entry':entry,'answerable':bool(answerable),'answer':answer}
    if answerable: rec.update({'val':float(val),'like':bool(like)})
    with open(logf,'a',encoding='utf-8') as ff: ff.write(json.dumps(rec)+'\n')          # FULL log: every query
    if answerable:
        with open(trajf,'a',encoding='utf-8') as ff: ff.write(json.dumps({'entry':entry_norm,'emb':emb.tolist(),'val':float(val),'like':bool(like)})+'\n')
    print(answer); sys.exit(0)
low=entry.lower(); kind=None; name=entry
if low.startswith('concept:'): kind='c'; name=entry[8:].strip()
elif low.startswith('movie:'): kind='m'; name=entry[6:].strip()
if kind in (None,'c') and name.lower() in d['concept_name2c']:
    c=d['concept_name2c'][name.lower()]
    if c in u['cans']:
        val=u['cans'][c]; like=val>0
        emit(True,f"ANSWER: the user {'LIKES' if like else 'DISLIKES'} {name} movies.",f'CONCEPT: {name}',d['Ec'][c],val,like)
    emit(False,f"ANSWER: the user can't say -- too few of their films are tagged '{name}'. [no information gained]",f'CONCEPT: {name}')
if kind in (None,'m') and name in d['title2iid']:
    iid=d['title2iid'][name]
    if iid in set(u['known']):
        val=u['resid'][iid]; like=val>0
        emit(True,f"ANSWER: the user has seen '{name}' and {'LIKED' if like else 'DISLIKED'} it.",f'MOVIE: {name}',d['Q'][iid],val,like)
    emit(False,f"ANSWER: the user has NOT seen '{name}'. [no information gained]",f'MOVIE: {name}')
emit(False,f"ERROR: '{entry}' not found. Copy an exact line from unified_menu.txt including the CONCEPT:/MOVIE: prefix.",f'UNRESOLVED: {entry}')
