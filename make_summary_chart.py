"""Summary bar chart: best foundation model vs best tree at the largest training size of each dataset,
from results/<name>_summary.csv (same numbers as the "FM vs tree" rows in results/tables.md).
Bars are coloured by whether the winning FM's weights allow commercial use (see README, Licences).
Usage: python make_summary_chart.py"""
import os, json, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); TREES = ["RF", "HGB", "XGB-tuned"]
NAMES = ["steel_faults", "MagicTelescope", "concrete", "energy_heating", "superconduct", "wine_quality"]
COMMERCIAL = {"Kumo": True, "LightPFN": True, "Causilo": False, "FTFM-1": False}
rows = []
for name in NAMES:
    r = json.load(open(f"{HERE}/results/{name}.json")); s = pd.read_csv(f"{HERE}/results/{name}_summary.csv")
    reg = r["task"] == "reg"; key = "RMSE_mean" if reg else "AUC_mean"; n = s.n.max(); sub = s[s.n == n]
    fm = sub[~sub.model.isin(TREES)]; tr = sub[sub.model.isin(TREES)]
    f = fm.loc[fm[key].idxmin() if reg else fm[key].idxmax()]; t = tr.loc[tr[key].idxmin() if reg else tr[key].idxmax()]
    gain = 100 * (1 - f[key] / t[key]) if reg else 100 * (1 - (1 - f[key]) / (1 - t[key]))
    rows.append((name, int(n), f.model, t.model, gain, "RMSE" if reg else "1 - AUC"))
fig, ax = plt.subplots(figsize=(9, 4.6))
y = range(len(rows))[::-1]
for yi, (name, n, fm, tm, g, met) in zip(y, rows):
    ax.barh(yi, g, color="#2b6cb0" if COMMERCIAL[fm] else "#bbbbbb", edgecolor="k", lw=0.6)
    ax.text(g + 0.8, yi, f"{g:.1f} %  ({fm} vs {tm})", va="center", fontsize=9)
ax.set_yticks(list(y)); ax.set_yticklabels([f"{nm}\n(n={n}, {met})" for nm, n, *_ , met in rows], fontsize=9)
ax.set_xlim(0, max(r[4] for r in rows) * 1.55); ax.set_xlabel("Error reduction of best FM vs best tree at the largest training size (%)")
ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#2b6cb0", ec="k", label="winning FM is commercially usable (Kumo)"),
                   Patch(fc="#bbbbbb", ec="k", label="winning FM is non-commercial (Causilo)")],
          loc="lower right", fontsize=8.5, frameon=True)
ax.set_title("Best foundation model vs best tree (oracle-picked), mean over repeats\n"
             "superconduct: Kumo was memory-killed after n=100, so Causilo is the only FM at n=2000", fontsize=10)
fig.tight_layout(); out = f"{HERE}/charts/summary_gain_largest_n.png"; fig.savefig(out, dpi=130); print(out)
for r in rows: print(r)
