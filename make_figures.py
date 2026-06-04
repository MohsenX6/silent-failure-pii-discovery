"""make_figures.py -- generate Figures 1-4 (IEEE single-column friendly)."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detectors import DETECTORS

plt.rcParams.update({"font.size": 9, "figure.dpi": 220,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})

R = json.load(open("results.json"))
TOOLS = R["tools"]
TYPES = R["types"]
SHORT = {"Regex baseline": "Regex", "Presidio (offline)": "Presidio",
         "Gazetteer-NER (EN)": "Gaz-EN", "Gazetteer-NER (locale)": "Gaz-loc"}
COLORS = ["#3b6ea5", "#c44e52", "#55a868", "#8172b3"]

# ---------------------------------------------------------------------------
# Figure 1 : pipeline + threat-model diagram
# ---------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis("off")
    boxes = [
        (0.02, "Production\nPII source", "#d9e3f0"),
        (0.215, "AI-assisted\nPII discovery", "#fde9d9"),
        (0.41, "Masking\nstep", "#d9e3f0"),
        (0.605, "Non-production\ntest copy", "#d9e3f0"),
        (0.80, "Developers /\nCI-CD use", "#d9e3f0"),
    ]
    w, h, y = 0.165, 0.34, 0.50
    centres = []
    for x, label, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c,
                                   edgecolor="#333", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
        centres.append(x + w / 2)
    for i in range(len(centres) - 1):
        ax.annotate("", xy=(boxes[i + 1][0], y + h / 2),
                    xytext=(boxes[i][0] + w, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
    # leakage annotation under the discovery box
    ax.annotate("Silent failure:\nundetected PII\n(names, Saudi IDs,\nperturbed values)",
                xy=(centres[1], y), xytext=(centres[1], 0.10),
                ha="center", va="center", fontsize=7.6, color="#b1141d",
                arrowprops=dict(arrowstyle="-|>", color="#b1141d", lw=1.2))
    # consequence caption flows along the lower pipeline (boxes 3-5)
    ax.text((centres[3] + centres[4]) / 2, 0.10,
            "not flagged → not masked →\npersists into test copy",
            ha="center", va="center", fontsize=7.6, color="#b1141d")
    # data-residency note on the AI box (hosted-LLM case), kept clear of the
    # top edge so it does not collide with other captions
    ax.text(centres[1], y + h + 0.04,
            "hosted-LLM variant: PII\nleaves data-residency boundary",
            ha="center", va="bottom", fontsize=7.0, color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig("fig1_pipeline.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 : grouped bar FNLR by entity type x tool  (strict)
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(TYPES))
    bw = 0.2
    for i, tool in enumerate(TOOLS):
        vals = [R["strict"][tool]["by_type"][t]["FNLR"] * 100 for t in TYPES]
        ax.bar(x + (i - 1.5) * bw, vals, bw, label=SHORT[tool], color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in TYPES], fontsize=7.6)
    ax.set_ylabel("False-Negative Leakage Rate (%)")
    ax.set_ylim(0, 108)
    ax.axhline(100, color="#999", lw=0.6, ls=":")
    ax.legend(ncol=4, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    fig.savefig("fig2_fnlr_bar.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 : heatmap PERSON_NAME recall by language x modality (locale gaz)
# ---------------------------------------------------------------------------
def fig3():
    import json as _j
    recs = [_j.loads(l) for l in open("corpus.jsonl", encoding="utf-8")]
    langs = ["arabic", "latin", "mixed"]; mods = ["structured", "freetext"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, tool in zip(axes, ["Gazetteer-NER (EN)", "Gazetteer-NER (locale)"]):
        det = DETECTORS[tool]
        cell = {(l, m): [0, 0] for l in langs for m in mods}
        for r in recs:
            preds = [p for p in det(r["text"]) if p["type"] == "PERSON_NAME"]
            for g in r["entities"]:
                if g["type"] != "PERSON_NAME":
                    continue
                hit = any(p["start"] == g["start"] and p["end"] == g["end"]
                          for p in preds)
                cell[(g["lang"], r["modality"])][0 if hit else 1] += 1
        M = np.array([[(cell[(l, m)][0] /
                        (cell[(l, m)][0] + cell[(l, m)][1])
                        if (cell[(l, m)][0] + cell[(l, m)][1]) else 0)
                       for m in mods] for l in langs])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, fontsize=8)
        ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs, fontsize=8)
        for i in range(len(langs)):
            for j in range(len(mods)):
                ax.text(j, i, f"{M[i, j]*100:.0f}%", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title(SHORT[tool] + "  (PERSON_NAME recall)", fontsize=8.5)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="recall")
    fig.savefig("fig3_heatmap.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 : clean vs adversarial recall per tool (strict)
# ---------------------------------------------------------------------------
def fig4():
    clean, adv = [], []
    for tool in TOOLS:
        bt = R["strict"][tool]["by_type_pert"]
        c = [0, 0]; a = [0, 0]
        for k, v in bt.items():
            t, p = k.split("|")
            if p == "none":
                c[0] += v["tp"]; c[1] += v["fn"]
            else:
                a[0] += v["tp"]; a[1] += v["fn"]
        clean.append(c[0] / (c[0] + c[1]) if (c[0] + c[1]) else 0)
        adv.append(a[0] / (a[0] + a[1]) if (a[0] + a[1]) else 0)
    x = np.arange(len(TOOLS)); bw = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - bw / 2, np.array(clean) * 100, bw, label="clean", color="#55a868")
    ax.bar(x + bw / 2, np.array(adv) * 100, bw, label="adversarial", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TOOLS], fontsize=8)
    ax.set_ylabel("Recall (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    for i, (cv, av) in enumerate(zip(clean, adv)):
        ax.text(i - bw / 2, cv * 100 + 1, f"{cv*100:.0f}", ha="center", fontsize=7)
        ax.text(i + bw / 2, av * 100 + 1, f"{av*100:.0f}", ha="center", fontsize=7)
    fig.savefig("fig4_clean_adv.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written: fig1_pipeline.png fig2_fnlr_bar.png "
          "fig3_heatmap.png fig4_clean_adv.png")
