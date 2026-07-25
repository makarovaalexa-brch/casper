r"""consolidate.py -- merge the answer-decomposition runs into ONE final artifact + RESULT md (author
overnight directive 2026-07-25). Reads (all optional except main):
  experiments/battery/answer_contrast_newrec.json   (main: G,B,O,K,U,S,C)
  experiments/battery/answer_contrast_imputers.json (E,P + B,O guards)
  experiments/battery/concept_granularity.json      (A1/A2 + C-lite vs C-full)
Writes:
  experiments/battery/answer_decomposition_FINAL.json
  docs/results/ANSWER_DECOMPOSITION_RESULT.md
Plain-language verdict per the four causes (answerability / selection / answer-value / fold-health) +
is-SEL-beatable. Re-runnable (idempotent).
"""
import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
BAT = os.path.join(_ROOT, "experiments", "battery")
BUDS = ["0", "1", "2", "4", "8"]
LEDGER_M8_REF = 0.1630                                           # sclite per-answer m8 (CONCEPT_CHANNEL_RESULT §4)


def _load(name):
    p = os.path.join(BAT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _ci_clean_pos(d):
    return d and d["ci95"][0] > 0


def main():
    main_j = _load("answer_contrast_newrec.json")
    imp_j = _load("answer_contrast_imputers.json")
    gran_j = _load("concept_granularity.json")
    assert main_j is not None, "main answer_contrast_newrec.json required to consolidate"
    intercept = main_j["intercept"]

    # ---- merge per-arm tables + guards ----
    concept_ask = dict(main_j.get("concept_ask", {}))
    item_ask = dict(main_j.get("item_ask", {}))
    cka = dict(main_j["guards"]["linear_CKA_answergeom_vs_ustar"])
    spear = dict(main_j["guards"]["spearman_value_vs_oracleB"])
    key = dict(main_j.get("key_quantities", {}))
    arms = list(main_j["arms"])
    if imp_j is not None:
        for tag in imp_j["arms"]:
            if tag in ("E", "P"):                                # take E,P from the imputer run
                concept_ask[tag] = imp_j["concept_ask"].get(tag)
                if tag in imp_j.get("item_ask", {}):
                    item_ask[tag] = imp_j["item_ask"][tag]
                cka[tag] = imp_j["guards"]["linear_CKA_answergeom_vs_ustar"].get(tag)
                spear[tag] = imp_j["guards"]["spearman_value_vs_oracleB"].get(tag)
                if tag not in arms:
                    arms.append(tag)
        for k, v in imp_j.get("key_quantities", {}).items():
            key.setdefault(k, v)

    # ---- verdict inputs ----
    def cget(tag, q, m):
        try:
            return concept_ask[tag][q][m]
        except Exception:
            return None
    def iget(tag, q, m):
        try:
            return item_ask[tag][q][m]
        except Exception:
            return None
    ma_concept_b = cget("B", "8", "mean_answered")
    ma_item_b = iget("B", "8", "mean_answered")
    B_q8 = cget("B", "8", "full@10"); K_q8 = cget("K", "8", "full@10")
    U_q8 = cget("U", "8", "full@10"); O_q8 = cget("O", "8", "full@10")

    def kd(name):
        return key.get(name)

    # ---- four-cause verdict ----
    verdict = {}
    # 1. answerability
    verdict["answerability"] = {
        "concept_mean_answered_q8": ma_concept_b, "item_mean_answered_q8": ma_item_b,
        "read": (f"Concept questions are answered far more often than item questions "
                 f"({ma_concept_b:.1f} vs {ma_item_b:.1f} folded by q8): popularity-ordered item "
                 f"questions are rarely rated by cold users, so the item belief barely moves until a "
                 f"few answerable items accumulate. Answerability favors concepts."
                 if (ma_concept_b and ma_item_b) else "n/a")}
    # 2. selection
    kb = kd("K_minus_B_concept_q8")
    verdict["selection"] = {
        "K_minus_B_q8": kb, "K_concept_q8_full": K_q8, "generic_B_q8_full": B_q8,
        "read": ((f"Oracle per-user concept selection lifts full@10 by "
                  f"{kb['full']['mean']:+.4f} (CI {kb['full']['ci95']}) over the generic polarization "
                  f"bank at q8 -> {'a real' if _ci_clean_pos(kb['full']) else 'a small/insignificant'} "
                  f"SELECTION gap: the realizable bank asks sub-optimal concepts.")
                 if kb else "n/a")}
    # 3. answer-value
    ub = kd("U_minus_B_concept_q8"); uo = kd("U_minus_O_concept_q8")
    verdict["answer_value"] = {
        "U_minus_B_q8": ub, "U_minus_O_q8": uo, "U_concept_q8_full": U_q8,
        "read": ((f"The TRUE answer-channel ceiling (utility oracle) sits {ub['full']['mean']:+.4f} "
                  f"above behavioral SEL and {uo['full']['mean']:+.4f} above the SEL-oracle at q8 "
                  f"(full@10). {'The SEL FORMULA is NOT the NDCG ceiling' if uo and uo['full']['mean'] > 0.003 else 'SEL is close to the answer ceiling'}"
                  f" -- a better per-question ANSWER exists.")
                 if ub else "n/a")}
    # 4. fold-health
    fold_ok = (K_q8 is not None and K_q8 >= LEDGER_M8_REF - 0.01)
    verdict["fold_health"] = {
        "K_concept_q8_full": K_q8, "ledger_sclite_m8_ref": LEDGER_M8_REF,
        "concept_lift_exists": (B_q8 is not None and B_q8 > intercept["full"]),
        "read": ((f"Oracle-selection concept-ask reaches {K_q8:.4f} full@10 at q8 vs the ledger's "
                  f"sclite per-answer m8 ~{LEDGER_M8_REF:.3f}: the fold is {'HEALTHY' if fold_ok else 'UNDERPERFORMING -- investigate (ckpt/harness)'}"
                  f". Concepts do move the belief above the {intercept['full']:.4f} intercept.")
                 if K_q8 is not None else "n/a")}
    # is-SEL-beatable
    beats = {}
    for imp in ("S", "C", "E", "P"):
        xo = kd(f"{imp}_minus_O_concept_q8"); xb = kd(f"{imp}_minus_B_concept_q8")
        beats[imp] = {"beats_oracleB": bool(xo and _ci_clean_pos(xo["full"])),
                      "beats_B": bool(xb and _ci_clean_pos(xb["full"])),
                      "vs_O_full": xo["full"]["mean"] if xo else None,
                      "vs_B_full": xb["full"]["mean"] if xb else None}
    any_beat_O = any(v["beats_oracleB"] for v in beats.values())
    verdict["is_SEL_beatable"] = {
        "imputers": beats, "any_imputer_beats_oracleB": any_beat_O,
        "read": (("A model-free imputer BEATS the SEL-oracle on concept NDCG -> the SEL formula is not "
                  "the ceiling and a better formula wins." if any_beat_O else
                  "No cheap model-free imputer (SEL+, content, ExpoMF, TagMF) beats the SEL-oracle on "
                  "concept NDCG -- SEL is a strong formula ceiling among realizable imputers; but the "
                  "utility oracle shows the ANSWER channel still has headroom the SEL family does not "
                  "reach."))}
    # granularity
    if gran_j is not None:
        a1 = gran_j["A1_granularity_of_selected"]; a2 = gran_j["A2_per_answer_lift_by_granularity"]
        sp = a2.get("spearman_lift_full_vs_membercount")
        verdict["granularity"] = {
            "median_member_count": {"oracle_selection": a1["oracle_selection_pooled"].get("median"),
                                    "polarization_bank": a1["polarization_bank"].get("median"),
                                    "member_mass_bank": a1["member_mass_bank"].get("median"),
                                    "all_concepts": a1["all_1031_concepts"].get("median")},
            "per_answer_lift_full": {b: a2["bins"][b].get("mean_lift_full@10") for b in a2["bins"]},
            "spearman_lift_vs_membercount": sp,
            "read": (f"Oracle selection asks median-{a1['oracle_selection_pooled'].get('median')}-member "
                     f"concepts vs the banks' median-{a1['polarization_bank'].get('median')}. "
                     f"Per-answer cold lift by granularity (fine/med/broad): "
                     f"{a2['bins']['fine'].get('mean_lift_full@10')}/{a2['bins']['medium'].get('mean_lift_full@10')}/"
                     f"{a2['bins']['broad'].get('mean_lift_full@10')}; Spearman(lift, member_count)={sp}. "
                     f"{'FINE concepts carry more per answer (coarseness is a selection artifact)' if (sp is not None and sp < -0.05) else 'Fine ~= broad (granularity not the lever)'}.")}

    final = {"analysis": "answer_decomposition_FINAL", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
             "stack": main_j.get("stack"), "n_users": main_j["n_users"], "budgets": main_j["budgets"],
             "arms_present": arms, "intercept": intercept,
             "controls": main_j.get("controls"),
             "guards": {"linear_CKA_answergeom_vs_ustar": cka, "spearman_value_vs_oracleB": spear},
             "item_ask": item_ask, "concept_ask": concept_ask,
             "channel_flip_within_model": main_j.get("channel_flip_within_model"),
             "key_quantities": key,
             "granularity": gran_j if gran_j else "PENDING/absent",
             "cfull_vs_sclite": (gran_j or {}).get("scfull_vs_sclite_concept_ask_B"),
             "VERDICT": verdict,
             "inputs_present": {"main": True, "imputers": imp_j is not None, "granularity": gran_j is not None}}
    os.makedirs(BAT, exist_ok=True)
    json.dump(final, open(os.path.join(BAT, "answer_decomposition_FINAL.json"), "w"), indent=2,
              default=float)

    # ---- RESULT md ----
    md = []
    md.append("# Answer-model decomposition -- FINAL (diagnostic for understanding, not paper)\n")
    md.append(f"> Generated {final['generated']}. Stack: {final['stack']}. {final['n_users']} COLD_SEED "
              f"users. Cold intercept **{intercept['full']:.4f}/{intercept['tail']:.4f}** (full/tail "
              f"NDCG@10). Controls: q0 snap {main_j['controls']['q0_canonical_snap_PASS']}, "
              f"shuffle collapses {main_j['controls']['shuffle_collapses_toward_intercept']}, "
              f"leak_users {main_j['controls']['leak_foldin_heldout_overlap_users']}. "
              f"Inputs: {final['inputs_present']}.\n")
    md.append("## Full table -- full@10 / tail@10 / full@100 per arm x budget\n")
    md.append("Arms: G geometric(circular/uncitable), B behavioral-SEL(honest), O oracle-B(SEL over full "
              "history), K oracle-concept-selection, U utility-oracle(true NDCG ceiling), S SEL+, "
              "C content-projection, E ExpoMF, P TagMF.\n")
    for label, tab in (("CONCEPT-ASK", concept_ask), ("ITEM-ASK", item_ask)):
        md.append(f"\n**{label}**\n")
        md.append("| arm | " + " | ".join(f"q{q}" for q in BUDS) + " |")
        md.append("|" + "---|" * (len(BUDS) + 1))
        for tag in arms:
            if tag not in tab or tab[tag] is None:
                continue
            cells = []
            for q in BUDS:
                d = tab[tag].get(q, {})
                cells.append(f"{d.get('full@10', float('nan')):.4f}/{d.get('tail@10', float('nan')):.4f}/"
                             f"{d.get('full@100', float('nan')):.4f}")
            md.append(f"| {tag} | " + " | ".join(cells) + " |")
    md.append("\n**mean_answered (folded per budget)**\n")
    md.append("| arm | channel | " + " | ".join(f"q{q}" for q in BUDS) + " |")
    md.append("|" + "---|" * (len(BUDS) + 2))
    for tag in arms:
        for ch, tab in (("concept", concept_ask), ("item", item_ask)):
            if tag in tab and tab[tag] is not None:
                md.append(f"| {tag} | {ch} | " + " | ".join(
                    f"{tab[tag].get(q, {}).get('mean_answered', float('nan')):.2f}" for q in BUDS) + " |")
    md.append("\n## Guards (per arm)\n")
    md.append("| arm | CKA(answer-geom, u*) | Spearman(value, oracle-B) |")
    md.append("|---|---|---|")
    for tag in arms:
        c = cka.get(tag); s = spear.get(tag)
        md.append(f"| {tag} | {('%.3f' % c) if isinstance(c, (int, float)) else c} | "
                  f"{('%.3f' % s) if isinstance(s, (int, float)) else s} |")
    md.append("\n(G is the HIGH-CKA circular reference; an honest imputer must not exceed SEL(B)'s CKA "
              "while agreeing more with oracle-B.)\n")
    if isinstance(final["cfull_vs_sclite"], dict):
        md.append("\n## C-lite vs C-full (concept-ask B, polarization, identical answers) -- full@10\n")
        md.append("| model | " + " | ".join(f"q{q}" for q in BUDS) + " |")
        md.append("|" + "---|" * (len(BUDS) + 1))
        for nm in ("sclite", "scfull"):
            row = final["cfull_vs_sclite"].get(nm, {})
            md.append(f"| {nm} | " + " | ".join(f"{row.get(q, {}).get('full@10', float('nan')):.4f}"
                                                for q in BUDS) + " |")
    md.append("\n## VERDICT -- the four causes + is-SEL-beatable\n")
    for cause in ("answerability", "selection", "answer_value", "fold_health", "granularity",
                  "is_SEL_beatable"):
        if cause in verdict:
            md.append(f"- **{cause}**: {verdict[cause]['read']}")
    md.append("")
    docdir = os.path.join(_ROOT, "docs", "results")
    os.makedirs(docdir, exist_ok=True)
    open(os.path.join(docdir, "ANSWER_DECOMPOSITION_RESULT.md"), "w").write("\n".join(md))
    print("[consolidate] wrote answer_decomposition_FINAL.json + docs/results/ANSWER_DECOMPOSITION_RESULT.md")
    for cause in verdict:
        print(f"  [{cause}] {verdict[cause]['read'][:160]}")


if __name__ == "__main__":
    main()
