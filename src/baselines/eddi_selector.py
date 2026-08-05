r"""eddi_selector.py -- EDDI's own myopic information-gain acquisition, as an interview question
selector over our trained Partial-VAE checkpoint.

PAPER: Ma, Tschiatschek, Palla, Hernandez-Lobato, Nowozin, Zhang, "EDDI: Efficient Dynamic Discovery of
High-Value Information with Partial VAE", ICML 2019. The consumer half (the Partial-VAE itself) is
`eddi_pvae.py`, trained on the Liang ML-25M split (full-profile 0.2153 / 0.1077, best epoch 31). This
file is the half an active-learning reviewer looks for first: EDDI is not a recommender, it is an
ACQUISITION FUNCTION, and until now our harness could not express one.

THE ACQUISITION, and exactly which one we implement.
Ma et al. define, for observed set x_O and candidate variable i,

    single-order (their Eq. 2/3, the "information gain about the latent state"):
        R(i, x_O) = E_{x_i ~ p(x_i | x_O)} [ KL( p(z | x_i, x_O) || p(z | x_O) ) ]

    EDDI proper (their Eq. 4) subtracts the part of that gain that is NOT about the target x_phi:
        R_EDDI(i, x_O) = E_{x_i}[ KL(p(z|x_i,x_O) || p(z|x_O))
                                  - E_{x_phi}[ KL(p(z|x_i,x_phi,x_O) || p(z|x_phi,x_O)) ] ]

We implement the SINGLE-ORDER form, and say why in one sentence a reviewer can check: EDDI's target
x_phi is a designated held-out VARIABLE, and in cold-start ranking there is no such variable -- the
"target" is the user's entire held-out like set, so p(z | x_phi, x_O) is not a quantity our protocol
lets us form without leaking the evaluation targets into the question selection. The single-order
reward is Ma et al.'s own reduction when no target is designated, it is the acquisition their
"PartialVAE + info-gain" ablation uses, and it keeps the firewall intact. Stated as a documented
deviation, not glossed.

THE CF INSTANTIATION (all four steps exact under the binary implicit contract):
  1. p(z | x_O). The Partial-VAE encoder is a permutation-invariant SUM over the observed coordinates:
     c_O = sum_{i in O} Phi_i with Phi = phi(E), then (mu_O, logvar_O) = enc_head(c_O). So conditioning
     on one more observed item is EXACTLY c_O + Phi_i -- no re-encoding of the history, no
     approximation. (This is the same identity `eddi_pvae.encode` computes as the sparse product
     X @ Phi.)
  2. The expectation over x_i. Under the binary likes contract the answer to "have you liked i?" has
     two outcomes. A LIKE sets x_i = 1 and moves the code to c_O + Phi_i. Anything else -- a dislike,
     or an item the user has not seen -- leaves the observed set unchanged (a Partial-VAE conditions
     on observed coordinates only), so its posterior is p(z | x_O) and its KL term is exactly ZERO.
     The expectation therefore collapses, with no approximation, to
         R(i) = p_i * KL( q(z | c_O + Phi_i) || q(z | c_O) )
     where p_i = P(the user likes i | x_O).
  3. p_i comes from the model's own predictive, as EDDI requires -- p(x_i | x_O) = E_z[ p(x_i | z) ]
     evaluated at z = mu_O, which under the Mult-VAE^PR likelihood of `eddi_pvae` is the multinomial
     probability softmax(decoder(mu_O))_i. Note this is a shared-normaliser monotone transform of the
     decoder logits, so it changes the acquisition only through the RELATIVE answerability weighting,
     which is precisely the role p(x_i|x_O) plays in Ma et al.
     ==> the acquisition is literally "expected posterior movement, weighted by how likely the user is
     able to say yes" -- the answerability-weighted information gain. That is a nice, honest sentence
     for the paper and it is a consequence of the method, not a choice we made.
  4. Gaussian KL in closed form, per latent dimension:
         KL = 0.5 * sum_d [ log(s0_d^2 / si_d^2) + (si_d^2 + (mi_d - m0_d)^2) / s0_d^2 - 1 ]

CANDIDATE POOL: all n_items minus the items already asked of that user -- the same universe the static
global arms walk. `cand_topm > 0` restricts scoring to the top-M items by predictive probability; it
is OFF by default because it is a data reduction that needs sign-off, and it stamps
CANDIDATE_POOL_TRUNCATED into the run's JSON when used.

COST. The encoder head must be evaluated for every candidate item at every distinct interview STATE:
(n_items x feat) -> hidden -> latent is ~11 GFLOP per state. States are cached on the observed LIKE
SET (the only thing c_O depends on), so turn 0 costs one evaluation for the whole cohort and the bill
grows with how much the policy branches. The harness prints distinct-state counts and per-turn wall
time; read them off the smoke before committing to the k=16 cell.
"""
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_harness import Selector

LIKE_LEVEL = 7          # level >= 7 == r >= 4.0 == the binary "like" cut used by answers_to_csr
NEG = -np.inf


class EDDIInfoGainSelector(Selector):
    """EDDI single-order info-gain over a trained `eddi_pvae.PartialVAE`."""
    name = "eddi_ig"

    def __init__(self, model, cand_topm=0, item_chunk=4096):
        self.model = model
        self.model.eval()
        self.cand_topm = int(cand_topm or 0)
        self.item_chunk = int(item_chunk)
        self._acq = {}
        self._n_states = []
        with torch.no_grad():
            self.Phi = self.model._phi().detach()            # (n_items x feat), fixed
        self.n_items = self.Phi.shape[0]

    def reset(self, ctx):
        super().reset(ctx)
        self._acq, self._n_states = {}, []
        if self.cand_topm:
            ctx["log"](f"[{self.name}] CANDIDATE POOL TRUNCATED to top-{self.cand_topm} by predictive "
                       f"probability -- a data reduction; not citable without sign-off")

    # ---------------------------------------------------------------- model heads
    def _enc_head(self, C):
        """pooled code(s) -> (mu, logvar). C: (b x feat)."""
        h = torch.tanh(self.model.enc1(C))
        return self.model.enc_mu(h), self.model.enc_lv(h)

    def _acquisition(self, like_key):
        """R(i) for every catalogue item, given the observed like set. Cached on the set."""
        if like_key in self._acq:
            return self._acq[like_key]
        with torch.no_grad():
            c0 = torch.zeros(self.Phi.shape[1], dtype=self.Phi.dtype)
            if like_key:
                c0 = self.Phi[torch.as_tensor(like_key, dtype=torch.long)].sum(dim=0)
            mu0, lv0 = self._enc_head(c0[None, :])
            mu0, lv0 = mu0[0], lv0[0]
            var0 = torch.exp(lv0)
            # p(x_i | x_O): the Mult-VAE^PR predictive at z = mu_O (EDDI's E_z[p(x_i|z)])
            p = torch.softmax(self.model.decode(mu0[None, :])[0], dim=0)          # (n_items,)
            R = torch.empty(self.n_items, dtype=torch.float32)
            for s in range(0, self.n_items, self.item_chunk):
                e = min(s + self.item_chunk, self.n_items)
                Ci = c0[None, :] + self.Phi[s:e]                                   # exact conditioning
                mui, lvi = self._enc_head(Ci)
                vari = torch.exp(lvi)
                kl = 0.5 * torch.sum((lv0[None, :] - lvi)
                                     + (vari + (mui - mu0[None, :]) ** 2) / var0[None, :]
                                     - 1.0, dim=1)
                R[s:e] = p[s:e] * kl
            out = R.numpy().astype(np.float64)
            if self.cand_topm and self.cand_topm < self.n_items:
                keep = np.argpartition(-p.numpy(), self.cand_topm)[:self.cand_topm]
                masked = np.full(self.n_items, NEG)
                masked[keep] = out[keep]
                out = masked
        self._acq[like_key] = out
        return out

    # ---------------------------------------------------------------- selector API
    def select(self, turn, users, asked, answered):
        groups = {}
        for u in users.tolist():
            key = tuple(sorted(i for (i, lv) in answered[u] if lv >= LIKE_LEVEL))
            groups.setdefault(key, []).append(int(u))
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        for key, us in groups.items():
            R = self._acquisition(key)
            for u in us:
                s = R.copy()
                s[np.asarray(asked[u], dtype=np.int64)] = NEG
                out[idx_of[u]] = int(np.argmax(s))
        return out

    def stats(self):
        return {"distinct_states_per_turn": self._n_states}


# =============================================================================================
# THE THREE-OUTCOME VARIANT (added 2026-08-05 by the EDDI native snap, task #3 of the snap ladder)
# =============================================================================================
r"""WHY A SECOND SELECTOR RATHER THAN A FIX. `EDDIInfoGainSelector` above is a two-outcome model: the
answer to "have you liked i?" is LIKE or not-LIKE, and everything that is not a like -- a dislike the
user actually gave us, and an item they have never seen -- collapses into the same zero-KL branch.
Our harness does NOT have that ambiguity: `adaptive_harness.run_interview` records `answered[u]` as
(item, half-star level) and an unrated item is simply absent, so the interview has THREE observable
outcomes, and the two-outcome acquisition cannot express, and therefore cannot price, the one that
costs us the most -- NO ANSWER ARRIVES. Ma et al. treat partial observation explicitly (the Partial
VAE conditions on the observed set; an unobserved variable is not a value, it is an absence), so
the faithful instantiation of their acquisition under OUR contract is the three-outcome one.

Both selectors stay, because Act 1 needs both rows: the two-outcome number is what a naive port of
EDDI to implicit CF gives, and the delta between them is the measurement.

THE THREE OUTCOMES AND THEIR POSTERIORS.
    y_i = UNANSWERED : the user has not rated i. The observed set does not change, so the posterior
                       is q(z | c_O) and the KL term is EXACTLY zero. Probability 1 - a_i.
    y_i = LIKE       : c_O + Phi_i, exactly as above.
    y_i = DISLIKE    : the user rated i below the like cut. The frozen Partial-VAE was trained on
                       binary LIKES only, so it has no learned token for a disliked observation
                       (there is no Phi^(0)); `dislike_mode` names the two honest options:
                         "zero"   -- a dislike moves nothing (KL 0). Conservative: the third outcome
                                     then enters ONLY through the answerability weight a_i.
                         "signed" -- c_O - Phi_i, the negative-evidence convention arm N already
                                     uses downstream. A DOCUMENTED APPROXIMATION, not the paper.
    R3(i) = a_i * [ q_i * KL(c_O + Phi_i || c_O) + (1 - q_i) * KL_dislike ]

a_i, THE ANSWERABILITY. Estimated from the GRADED TRAIN matrix only -- the fraction of the 140,768
train users who RATED item i at all, at any star level, Laplace-smoothed. Train-only, so there is no
test leakage; structural, so it is not an LLM-derived artifact (the Jul-22 audit forbids those).
This is the quantity the two-outcome model has no slot for: its `p` is a LIKE probability.

q_i, P(LIKE | ANSWERED). The base rate is the item's own train like-rate among its raters; the model
then tilts it per user, so the acquisition stays personalised as EDDI requires:
    q_i = sigmoid( logit(base_i) + beta * zscore_i ),  zscore_i = (logit_i - mean) / std
where logit_i is the frozen decoder's logit at z = mu_O. beta = 1.0, NOT tuned.

WHY THIS IS ALSO THE FIX FOR A SECOND, MEASURED DEFECT. On the frozen checkpoint the two-outcome
acquisition is numerically dominated by its weight, not by its information term: at the empty state
log p spans 7.4 decades across the catalogue while log KL spans 1.7, and corr(log p, log R) = 0.965.
That is a consequence of the Mult-VAE^PR decoder: `softmax` normalises ACROSS items, so p is a
relative next-draw probability, not the per-variable marginal p(x_i | x_O) that Ma et al.'s
factorised likelihood (their Eq. 2, p(x|z) = prod_i p_i(x_i|z)) requires. a_i and q_i are both
proper per-item marginals in [0,1], which puts the weight and the information term back on
comparable scales.

STATE. The two-outcome selector keys its cache on the LIKE set only, discarding the dislikes the
user actually gave. This one keys on (likes, dislikes) and, under "signed", conditions on both.
"""


def structural_answerability(g_train, n_items, prior_strength=5.0):
    """(a_i, base_like_rate_i) from the GRADED TRAIN matrix. Train-only; no test data is touched.

    a_i    = P(a random user has rated item i)          -- Laplace-smoothed toward 0
    base_i = P(the rating is a LIKE | the item is rated) -- shrunk toward the global like rate
    """
    import scipy.sparse as sp
    g = sp.csr_matrix(g_train)
    n_users = g.shape[0]
    # g_train stores RAW HALF-STARS (0.5..5.0), not the integer levels `answered` carries. The like
    # cut LIKE_LEVEL=7 is level = round(2r)-1 >= 7, i.e. r >= 4.0 -- the same cut answers_to_csr uses.
    lv = np.clip(np.rint(g.data * 2).astype(np.int64) - 1, 0, None)
    gl = sp.csr_matrix(((lv >= LIKE_LEVEL).astype(np.float64), g.indices, g.indptr), shape=g.shape)
    rated = np.asarray((g > 0).sum(axis=0)).ravel().astype(np.float64)
    liked = np.asarray(gl.sum(axis=0)).ravel().astype(np.float64)
    assert liked.sum() > 0, "no likes found in g_train -- check the star->level convention"
    a = (rated + 1.0) / (n_users + 2.0)
    glob = float(liked.sum() / max(rated.sum(), 1.0))
    base = (liked + prior_strength * glob) / (rated + prior_strength)
    return a.astype(np.float64), base.astype(np.float64)


class EDDIThreeOutcomeSelector(Selector):
    """EDDI info-gain with a three-outcome response model (unanswered / like / dislike)."""
    name = "eddi_ig3"

    def __init__(self, model, g_train, dislike_mode="signed", beta=1.0, item_chunk=4096):
        assert dislike_mode in ("zero", "signed"), dislike_mode
        self.model = model
        self.model.eval()
        self.dislike_mode = dislike_mode
        self.beta = float(beta)
        self.item_chunk = int(item_chunk)
        self._acq = {}
        self._n_states = []
        with torch.no_grad():
            self.Phi = self.model._phi().detach()
        self.n_items = self.Phi.shape[0]
        a, base = structural_answerability(g_train, self.n_items)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.base_logit = torch.tensor(np.log(base / (1.0 - base)), dtype=torch.float32)
        self.name = f"eddi_ig3_{dislike_mode}"

    def reset(self, ctx):
        super().reset(ctx)
        self._acq, self._n_states = {}, []
        ctx["log"](f"[{self.name}] three-outcome response model: P(unanswered)=1-a_i from graded "
                   f"train ({float(self.a.mean()):.5f} mean answerability), dislike branch="
                   f"{self.dislike_mode}, beta={self.beta}")

    def _enc_head(self, C):
        h = torch.tanh(self.model.enc1(C))
        return self.model.enc_mu(h), self.model.enc_lv(h)

    def _kl_against(self, c0, mu0, lv0, var0, sign):
        """KL( q(z | c0 + sign*Phi_i) || q(z | c0) ) for every item i."""
        out = torch.empty(self.n_items, dtype=torch.float32)
        for s in range(0, self.n_items, self.item_chunk):
            e = min(s + self.item_chunk, self.n_items)
            Ci = c0[None, :] + sign * self.Phi[s:e]
            mui, lvi = self._enc_head(Ci)
            out[s:e] = 0.5 * torch.sum((lv0[None, :] - lvi)
                                       + (torch.exp(lvi) + (mui - mu0[None, :]) ** 2) / var0[None, :]
                                       - 1.0, dim=1)
        return out

    def _acquisition(self, key):
        if key in self._acq:
            return self._acq[key]
        likes, dislikes = key
        with torch.no_grad():
            c0 = torch.zeros(self.Phi.shape[1], dtype=self.Phi.dtype)
            if likes:
                c0 = c0 + self.Phi[torch.as_tensor(likes, dtype=torch.long)].sum(dim=0)
            if dislikes and self.dislike_mode == "signed":
                c0 = c0 - self.Phi[torch.as_tensor(dislikes, dtype=torch.long)].sum(dim=0)
            mu0, lv0 = self._enc_head(c0[None, :])
            mu0, lv0 = mu0[0], lv0[0]
            var0 = torch.exp(lv0)
            logit = self.model.decode(mu0[None, :])[0]
            z = (logit - logit.mean()) / (logit.std() + 1e-8)
            q = torch.sigmoid(self.base_logit + self.beta * z)      # P(like | answered, x_O)
            kl_pos = self._kl_against(c0, mu0, lv0, var0, +1.0)
            if self.dislike_mode == "signed":
                kl_neg = self._kl_against(c0, mu0, lv0, var0, -1.0)
            else:
                kl_neg = torch.zeros_like(kl_pos)
            R = self.a * (q * kl_pos + (1.0 - q) * kl_neg)
            out = R.numpy().astype(np.float64)
        self._acq[key] = out
        return out

    def select(self, turn, users, asked, answered):
        groups = {}
        for u in users.tolist():
            likes = tuple(sorted(i for (i, lv) in answered[u] if lv >= LIKE_LEVEL))
            dislikes = tuple(sorted(i for (i, lv) in answered[u] if lv < LIKE_LEVEL))
            groups.setdefault((likes, dislikes), []).append(int(u))
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        for key, us in groups.items():
            R = self._acquisition(key)
            for u in us:
                s = R.copy()
                s[np.asarray(asked[u], dtype=np.int64)] = NEG
                out[idx_of[u]] = int(np.argmax(s))
        return out

    def stats(self):
        return {"distinct_states_per_turn": self._n_states,
                "response_model": "three_outcome", "dislike_mode": self.dislike_mode,
                "beta": self.beta}
