"""
WOLPERTINGER replication (Dulac-Arnold et al. 2015, "Deep RL in Large Discrete Action Spaces"), mechanism-level,
self-contained (no gym). Validates the machinery BEFORE applying to CASPER, so a later CASPER failure is
attributable to the task, not a broken implementation.

Task: contextual bandit with a LARGE discrete action set. N actions are fixed embeddings a_i in R^d. For state s the
target direction is t=normalize(W s); reward(s,a_i)=a_i . t. Optimal = max_i a_i.t (full enumeration = ceiling).
Wolpertinger: actor f(s)->proto p in R^d; kNN = the k actions nearest p (cosine); critic Q(s,a) scores them; pick
argmax-Q. DDPG-style training (critic regresses reward; actor maximizes Q(s,actor(s))). epsilon-greedy exploration.

SUCCESS (reproduces paper's claims): (1) learns -> reward >> random, approaches the full-enumeration ceiling;
(2) larger k -> closer to ceiling (the accuracy/speed trade-off); (3) scales to large N without enumerating in the
actor. Reports normalized score = (reward-random)/(ceiling-random) in [0,1].
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); rng=np.random.default_rng(0)
d=8; N=int(2000); STEPS=4000; B=256
A=torch.tensor(rng.standard_normal((N,d)),dtype=torch.float32); A=A/A.norm(dim=1,keepdim=True)
W=torch.tensor(rng.standard_normal((d,d)),dtype=torch.float32)
def batch_states(n): s=torch.tensor(rng.standard_normal((n,d)),dtype=torch.float32); return s
def target(s): t=s@W.t(); return t/t.norm(dim=1,keepdim=True)
def rewards_all(s):                     # (n,N) reward of every action for every state = ceiling oracle
    return target(s)@A.t()
actor=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,d))
critic=nn.Sequential(nn.Linear(d+d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
oa=torch.optim.Adam(actor.parameters(),1e-3); oc=torch.optim.Adam(critic.parameters(),1e-3)
def knn(p,k):                           # k nearest actions to proto p (cosine), per row
    pn=p/(p.norm(dim=1,keepdim=True)+1e-9); sims=pn@A.t(); return sims.topk(k,dim=1).indices  # (n,k)
def select(s,k,eps=0.0):
    with torch.no_grad():
        p=actor(s); cand=knn(p,k)                                  # (n,k)
        sc=s.unsqueeze(1).expand(-1,k,-1); ac=A[cand]              # (n,k,d)
        q=critic(torch.cat([sc,ac],-1)).squeeze(-1)               # (n,k)
        pick=q.argmax(1)
        if eps>0:
            rndk=torch.randint(0,k,(len(s),)); m=torch.rand(len(s))<eps; pick=torch.where(m,rndk,pick)
        chosen=cand[torch.arange(len(s)),pick]
    return chosen
def evaluate(k,n=2000):
    s=batch_states(n); ch=select(s,k); ra=rewards_all(s)
    r=ra[torch.arange(n),ch]; ceil=ra.max(1).values; rand=ra.mean(1)  # random action = mean reward
    return ((r-rand)/(ceil-rand+1e-9)).mean().item(), r.mean().item(), ceil.mean().item()
K_TRAIN=10
print(f"Wolpertinger replication: N={N} actions, d={d}. normalized score in [0,1] (0=random,1=full-enum ceiling).",flush=True)
for step in range(STEPS):
    s=batch_states(B); ch=select(s,K_TRAIN,eps=max(0.3*(1-step/STEPS),0.05))
    a=A[ch]; r=rewards_all(s)[torch.arange(B),ch].detach()
    q=critic(torch.cat([s,a],-1)).squeeze(-1); lc=((q-r)**2).mean()
    oc.zero_grad(); lc.backward(); oc.step()
    la=-critic(torch.cat([s,actor(s)],-1)).mean()
    oa.zero_grad(); la.backward(); oa.step()
    if (step+1)%1000==0:
        ns,rm,cl=evaluate(K_TRAIN); print(f"  step{step+1}: norm-score={ns:.3f} (reward {rm:.3f} / ceiling {cl:.3f})",flush=True)
print("\nk-tradeoff (more neighbours -> closer to full-enumeration ceiling):",flush=True)
for k in [1,5,10,50,200]:
    ns,rm,cl=evaluate(k); print(f"  k={k:<4} norm-score={ns:.3f}",flush=True)
ns1,_,_=evaluate(N)  # k=N == full enumeration through the critic (upper bound the critic can express)
print(f"  k=N({N}) norm-score={ns1:.3f}  (critic over ALL actions)",flush=True)
