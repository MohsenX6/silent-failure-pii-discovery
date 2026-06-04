"""
score.py
Entity-level scoring of every detector against the labelled corpus.
Produces results.json with P/R/F1, FNLR, SWLR, robustness gaps and Wilson CIs.
"""
import json
import math
from collections import defaultdict

from detectors import DETECTORS, _GAZ_LOC, _norm, _TOK

CORPUS = "corpus.jsonl"
TYPES = ["PERSON_NAME", "NATIONAL_ID", "IBAN", "PHONE", "EMAIL", "NATIONAL_ADDRESS"]
# Sensitivity weights for SWLR (justified in paper): identifiers that uniquely
# and durably single out a person and unlock financial/government access rank
# highest; an email or short address ranks lower.
W = {"NATIONAL_ID": 5, "IBAN": 5, "PHONE": 3, "PERSON_NAME": 2,
     "NATIONAL_ADDRESS": 2, "EMAIL": 1}
FAMILIES = ["translit", "codeswitch", "homoglyph", "format", "misplaced"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def match(gold, pred, mode="strict"):
    """Return (tp, fp, fn) matching pred spans to gold spans, one-to-one, by type."""
    used = set()
    tp = 0
    for p in pred:
        hit = None
        for gi, g in enumerate(gold):
            if gi in used or g["type"] != p["type"]:
                continue
            if mode == "strict":
                ok = (g["start"] == p["start"] and g["end"] == p["end"])
            else:
                ok = (p["start"] < g["end"] and g["start"] < p["end"])
            if ok:
                hit = gi
                break
        if hit is not None:
            used.add(hit)
            tp += 1
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def name_in_dictionary(value):
    """A clean person-name is 'in-dictionary' if at least one of its normalized
    tokens is present in the (train-only) locale gazetteer -- i.e. the dictionary
    has a vocabulary entry that lets it fire on the name. If no token is present,
    the whole name is held-out (out-of-vocabulary) and an exact-match dictionary
    cannot detect it. Uses the same tokenizer/normalizer as the detector so the
    classification matches detection mechanics exactly."""
    toks = [_norm(t.group()) for t in _TOK.finditer(value)]
    return any(tk in _GAZ_LOC for tk in toks)


def main():
    recs = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]

    # Precompute predictions per tool per record
    preds = {tool: [fn(r["text"]) for r in recs] for tool, fn in DETECTORS.items()}

    results = {"tools": list(DETECTORS), "types": TYPES, "weights": W,
               "n_records": len(recs)}

    for mode in ["strict", "lenient"]:
        mres = {}
        for tool in DETECTORS:
            # counters
            by_type = defaultdict(lambda: [0, 0, 0])          # type -> tp,fp,fn
            by_type_lang = defaultdict(lambda: [0, 0, 0])     # (type,lang)
            by_type_mod = defaultdict(lambda: [0, 0, 0])      # (type,modality)
            by_type_pert = defaultdict(lambda: [0, 0, 0])     # (type,pert)
            by_lang_mod = defaultdict(lambda: [0, 0])         # (lang,mod)->det,miss
            name_split = defaultdict(lambda: [0, 0])          # split -> det,miss
            name_split_lang = defaultdict(lambda: [0, 0])     # (split,lang)
            overall = [0, 0, 0]

            for ri, r in enumerate(recs):
                gold = r["entities"]
                pred = preds[tool][ri]
                # global match
                tp, fp, fn = match(gold, pred, mode)
                overall[0] += tp; overall[1] += fp; overall[2] += fn
                # per-type: restrict gold & pred to type
                for t in TYPES:
                    g_t = [g for g in gold if g["type"] == t]
                    p_t = [p for p in pred if p["type"] == t]
                    tp, fp, fn = match(g_t, p_t, mode)
                    by_type[t][0] += tp; by_type[t][1] += fp; by_type[t][2] += fn
                    # recall-only stratification (count tp & fn by gold attribute)
                    # rematch to attribute each gold as detected or not
                    used = set()
                    for p in p_t:
                        for gi, g in enumerate(g_t):
                            if gi in used:
                                continue
                            if mode == "strict":
                                ok = (g["start"] == p["start"] and g["end"] == p["end"])
                            else:
                                ok = (p["start"] < g["end"] and g["start"] < p["end"])
                            if ok:
                                used.add(gi); break
                    for gi, g in enumerate(g_t):
                        det = gi in used
                        lang = g["lang"]; mod = r["modality"]; pert = g["perturbation"]
                        kl = (t, lang); km = (t, mod); kp = (t, pert)
                        by_type_lang[kl][0 if det else 2] += 1
                        by_type_mod[km][0 if det else 2] += 1
                        by_type_pert[kp][0 if det else 2] += 1
                        by_lang_mod[(lang, mod)][0 if det else 1] += 1
                        # In-dictionary vs held-out split for CLEAN person-names
                        # only (the adversarial axis is handled by RG).
                        if t == "PERSON_NAME" and pert == "none":
                            split = ("in_dict" if name_in_dictionary(g["value"])
                                     else "held_out")
                            name_split[split][0 if det else 1] += 1
                            name_split_lang[(split, lang)][0 if det else 1] += 1

            def pack(c):
                tp, fp, fn = c
                P = tp / (tp + fp) if tp + fp else 0.0
                R = tp / (tp + fn) if tp + fn else 0.0
                F1 = 2 * P * R / (P + R) if P + R else 0.0
                lo, hi = wilson(tp, tp + fn)
                return {"tp": tp, "fp": fp, "fn": fn, "P": P, "R": R, "F1": F1,
                        "FNLR": 1 - R, "n": tp + fn, "R_ci": [lo, hi]}

            tool_block = {
                "overall": pack(overall),
                "by_type": {t: pack(by_type[t]) for t in TYPES},
                "by_type_lang": {f"{t}|{l}": pack(by_type_lang[(t, l)])
                                 for (t, l) in by_type_lang},
                "by_type_mod": {f"{t}|{m}": pack(by_type_mod[(t, m)])
                                for (t, m) in by_type_mod},
                "by_type_pert": {f"{t}|{p}": pack(by_type_pert[(t, p)])
                                 for (t, p) in by_type_pert},
            }

            tool_block["recall_lang_mod"] = {
                f"{lg}|{md}": {"R": v[0] / (v[0] + v[1]) if (v[0] + v[1]) else 0.0,
                               "n": v[0] + v[1]}
                for (lg, md), v in by_lang_mod.items()}

            # In-dictionary vs held-out person-name recall (clean names only)
            def packR(c):
                det, miss = c
                n = det + miss
                R = det / n if n else 0.0
                lo, hi = wilson(det, n)
                return {"R": R, "FNLR": 1 - R, "n": n, "R_ci": [lo, hi]}

            LANGS = ["arabic", "latin", "mixed"]
            tool_block["name_split"] = {
                "in_dict": packR(name_split["in_dict"]),
                "held_out": packR(name_split["held_out"]),
                "in_dict_by_lang": {lg: packR(name_split_lang[("in_dict", lg)])
                                    for lg in LANGS},
                "held_out_by_lang": {lg: packR(name_split_lang[("held_out", lg)])
                                     for lg in LANGS},
            }

            # SWLR over types
            num = sum(W[t] * by_type[t][2] for t in TYPES)
            den = sum(W[t] * (by_type[t][0] + by_type[t][2]) for t in TYPES)
            tool_block["SWLR"] = num / den if den else 0.0

            # FNLR by language (aggregate over all types) and by modality
            for axis, store in [("lang", by_type_lang), ("mod", by_type_mod)]:
                agg = defaultdict(lambda: [0, 0])  # key -> det, miss
                for (t, k), c in [((k.split("|")[0], k.split("|")[1]),
                                   tool_block[f"by_type_{'lang' if axis=='lang' else 'mod'}"][k])
                                  for k in tool_block[f"by_type_{'lang' if axis=='lang' else 'mod'}"]]:
                    agg[k][0] += c["tp"]; agg[k][1] += c["fn"]
                tool_block[f"FNLR_by_{axis}"] = {
                    k: {"FNLR": v[1] / (v[0] + v[1]) if (v[0] + v[1]) else 0.0,
                        "R": v[0] / (v[0] + v[1]) if (v[0] + v[1]) else 0.0,
                        "n": v[0] + v[1]} for k, v in agg.items()}

            # Robustness gap per family: recall on perturbed gold of affected
            # types vs recall on clean gold of the SAME types.
            rg = {}
            for fam in FAMILIES:
                aff_types = set(t for (t, p) in
                                [(k.split("|")[0], k.split("|")[1])
                                 for k in tool_block["by_type_pert"]]
                                if p == fam and
                                tool_block["by_type_pert"][f"{t}|{fam}"]["n"] > 0)
                if not aff_types:
                    continue
                det_a = miss_a = det_c = miss_c = 0
                for t in aff_types:
                    a = tool_block["by_type_pert"].get(f"{t}|{fam}")
                    c = tool_block["by_type_pert"].get(f"{t}|none")
                    if a:
                        det_a += a["tp"]; miss_a += a["fn"]
                    if c:
                        det_c += c["tp"]; miss_c += c["fn"]
                R_a = det_a / (det_a + miss_a) if (det_a + miss_a) else 0.0
                R_c = det_c / (det_c + miss_c) if (det_c + miss_c) else 0.0
                rg[fam] = {"R_clean": R_c, "R_adv": R_a, "RG": R_c - R_a,
                           "n_adv": det_a + miss_a, "types": sorted(aff_types)}
            tool_block["robustness_gap"] = rg

            mres[tool] = tool_block
        results[mode] = mres

    json.dump(results, open("results.json", "w"), indent=2)

    # ---- console summary (strict) ----
    print("=== STRICT  overall P/R/F1 ===")
    for tool in DETECTORS:
        o = results["strict"][tool]["overall"]
        print(f"  {tool:24s} P={o['P']:.3f} R={o['R']:.3f} F1={o['F1']:.3f} "
              f"SWLR={results['strict'][tool]['SWLR']:.3f}")
    print("\n=== STRICT  FNLR per type per tool ===")
    hdr = "type".ljust(18) + "".join(t[:14].ljust(15) for t in DETECTORS)
    print(hdr)
    for t in TYPES:
        row = t.ljust(18)
        for tool in DETECTORS:
            v = results["strict"][tool]["by_type"][t]
            row += f"{v['FNLR']*100:5.1f}% (n{v['n']})".ljust(15)
        print(row)
    print("\n=== STRICT PERSON_NAME FNLR by language ===")
    for tool in DETECTORS:
        bl = results["strict"][tool]["by_type_lang"]
        parts = []
        for lg in ["arabic", "latin", "mixed"]:
            k = f"PERSON_NAME|{lg}"
            if k in bl:
                parts.append(f"{lg}={bl[k]['FNLR']*100:.1f}%(n{bl[k]['n']})")
        print(f"  {tool:24s} " + "  ".join(parts))
    print("\n=== STRICT in-dictionary vs held-out person-name recall (clean) ===")
    for tool in DETECTORS:
        ns = results["strict"][tool]["name_split"]
        idc, ho = ns["in_dict"], ns["held_out"]
        print(f"  {tool:24s} in_dict R={idc['R']:.3f}(n{idc['n']})  "
              f"held_out R={ho['R']:.3f}(n{ho['n']})")
        if tool == "Gazetteer-NER (locale)":
            for lg in ["arabic", "latin", "mixed"]:
                a = ns["in_dict_by_lang"][lg]; b = ns["held_out_by_lang"][lg]
                print(f"        {lg:8s} in_dict R={a['R']:.3f}(n{a['n']})  "
                      f"held_out R={b['R']:.3f}(n{b['n']})")
    print("\n=== STRICT robustness gap (RG) per family ===")
    for tool in DETECTORS:
        rg = results["strict"][tool]["robustness_gap"]
        if rg:
            print(f"  {tool}:")
            for fam, v in rg.items():
                print(f"      {fam:11s} R_clean={v['R_clean']:.3f} "
                      f"R_adv={v['R_adv']:.3f} RG={v['RG']:+.3f} "
                      f"(n_adv={v['n_adv']}, {','.join(v['types'])})")


if __name__ == "__main__":
    main()
