"""E0f shared machinery — the NON-CIRCULAR real-rating item answer model.

E0d/E0e answered EVERY question geometrically (a = cos(z*, q)); real star ratings never entered
the answer value. E0f wires in the study-design §3 item answer:

  For an ITEM the user RATED (in the KNOWN portion of the pinned seed-123 answerer split — NEVER a
  held-out target): the answer is the user's real rating r, CENTERED on the user's own known-profile
  mean rating, RESCALED to the fold's answer scale, plus Gaussian fidelity noise sigma=0.70 stars
  (the sigma is rescaled by the SAME factor as the rating).

RESCALE CHOICE (stated once, used everywhere): a single global scalar BETA that matches the cohort RMS
of the real-rating-centered signal to the cohort RMS of the GEOMETRIC item answers over the identical
(user, known-rated-item) token set — the eta-matching approach of Paper C's fidelity-boundary section.
This puts the real-rating item channel at exactly the energy the geometric item channel would have on
the same tokens, so the ONLY thing that changes between E0d and E0f is the item answer VALUE (its sign
and graded magnitude come from data, not from cos(z*,.)), never the fold step budget.

  a_real(u, j) = BETA * ( (r_uj - mean_known_u) + 0.70 * eps_uj ),   eps_uj ~ N(0,1) seeded per (u,j).

Item DIRECTION is unchanged: q = Wn[j] (unit RecVAE decoder row), folded z' = z + ETA * a_real * q,
exactly as the geometric harness folds. Concepts are untouched (geometric), so any E0f vs E0d change
isolates the item VALUE channel.

The E0f "item channel" candidate pool = the user's KNOWN-portion rated items (trivially answerable per
study-design §3: "Item — RATED -> answerability yes (rated)"). The E0d bank's designed/taste-adjacent
items are almost never in a user's rating history, so they cannot carry a real rating; using the
known-rated items is the faithful realization of "answerable AND rated". (This makes the rated-item
descent an OPEN-RECALL-grade channel — the user supplies a real opinion on a film they have seen.)

NO LLM calls. Reuses e0_gonogo (E) + llm_answerability_gate (G) machinery exactly.
"""
import numpy as np


SIGMA_STARS = 0.70          # fidelity noise in STAR units (study-design §3 / fidelity sigma)


def augment_users(D, split, U):
    """Attach per-user known-rated item info: known_rated=[(j,rating)], known_mean, and a dir cache.
    Every KNOWN item has a rating (the split is a shuffle of the user's RATED items), so known ⊂ rated;
    none is a held-out target (targets live in the other half). Returns U (mutated in place)."""
    for rec in U:
        u = rec["u"]
        kn = split[u][0]                                  # set of known dense ids (answerer sees these)
        rat = dict(D["rat_by_u"][u])
        kr = [(int(j), float(rat[j])) for j in kn if j in rat]
        rec["known_rated"] = kr
        rec["known_mean"] = float(np.mean([r for _, r in kr])) if kr else 0.0
    return U


def compute_beta(FI, U):
    """Global BETA = RMS(geometric item answers) / RMS(centered real ratings), over all
    (user, known-rated-item) pairs. Also returns the two RMS values for the record."""
    Wn = FI["Wn"]
    geo_sq, rat_sq, npair = 0.0, 0.0, 0
    for rec in U:
        zstar, nz, km = rec["zstar"], rec["nz"], rec["known_mean"]
        for j, r in rec["known_rated"]:
            a_geo = float((Wn[j] @ zstar) / nz)           # cos(z*, Wn[j]) (Wn[j] unit)
            raw = r - km
            geo_sq += a_geo * a_geo
            rat_sq += raw * raw
            npair += 1
    rms_geo = float(np.sqrt(geo_sq / max(npair, 1)))
    rms_rat = float(np.sqrt(rat_sq / max(npair, 1)))
    beta = rms_geo / rms_rat if rms_rat > 0 else 0.0
    return beta, dict(rms_geo=rms_geo, rms_rating_centered=rms_rat, beta=beta, n_pairs=npair)


def _noise(u, j):
    """Deterministic per-(user,item) standard-normal fidelity draw."""
    seed = (int(u) * 100003 + int(j)) % (2 ** 32)
    return float(np.random.default_rng(seed).standard_normal())


def real_answer(rec, j, r, beta, sigma=SIGMA_STARS, add_noise=True):
    """study-design §3 real-rating item answer, rescaled to the fold's geometric answer scale."""
    raw = (r - rec["known_mean"])
    if add_noise:
        raw = raw + sigma * _noise(rec["u"], j)
    return beta * raw


def geo_answer(rec, q):
    """The OLD geometric answer a = cos(z*, q) (q assumed unit-norm)."""
    return float((q @ rec["zstar"]) / rec["nz"])
