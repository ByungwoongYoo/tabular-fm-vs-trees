"""Learning-curve PNGs from results/<name>_summary.csv. Usage: python make_charts.py name1 name2 ..."""
import sys, json, os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
TREES = ["RF", "HGB", "XGB-tuned"]
COL = {"Kumo": "#d62728", "Causilo": "#1f77b4", "FTFM-1": "#9467bd", "LightPFN": "#2ca02c", "TabPFN": "#8c564b",
       "RF": "#7f7f7f", "HGB": "#bcbd22", "XGB-tuned": "#ff7f0e"}
MARK = {"Kumo": "o", "Causilo": "s", "FTFM-1": "D", "LightPFN": "^", "RF": "v", "HGB": "P", "XGB-tuned": "X", "TabPFN": "h"}
TITLES = {"concrete": "Concrete compressive strength (UCI) - regression, RMSE in MPa",
          "energy_heating": "Building heating load (UCI energy efficiency) - regression, RMSE in kWh/m²",
          "steel_faults": "Steel plate fault type, 7 classes (UCI) - classification, 1 - macro AUC",
          "wine_quality": "Wine quality score 3-9 (UCI, red+white) - regression, RMSE in quality points",
          "MagicTelescope": "MAGIC gamma telescope, gamma vs hadron (UCI) - binary classification, 1 - AUC",
          "superconduct": "Superconductor critical temperature (UCI) - regression, RMSE in K"}
for name in sys.argv[1:]:
    res = json.load(open(f"{HERE}/results/{name}.json")); task = res["task"]
    s = pd.read_csv(f"{HERE}/results/{name}_summary.csv")
    if task == "reg":
        s["err"], s["err_sd"], ylab = s["RMSE_mean"], s["RMSE_std"], "Test RMSE (lower is better)"
    else:
        s["err"], s["err_sd"], ylab = 1 - s["AUC_mean"], s["AUC_std"], "Test error = 1 - AUC (lower is better)"
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    order = [m for m in ["Kumo", "Causilo", "FTFM-1", "LightPFN", "TabPFN", "XGB-tuned", "HGB", "RF"] if m in set(s.model)]
    for i, m in enumerate(order):
        d = s[s.model == m].sort_values("n"); tree = m in TREES
        dodge = 1 + 0.018 * (i - (len(order) - 1) / 2)  # small horizontal offset so error bars do not overlap
        lab = f"{m} (tree baseline)" if tree else f"{m} (foundation model)"
        if d.n.nunique() < s.n.nunique():  # incomplete learning curve: say why in the legend
            why = res["model_status"].get(m, {}).get("reason", "run incomplete")
            why = "killed at 3 GB RSS limit" if "RSS" in why else ("timed out" if "timeout" in why else why)
            lab += f"\n  partial: only n≤{int(d.n.max())} ({why})"
        ax.errorbar(d.n * dodge, d.err, yerr=d.err_sd.fillna(0), color=COL[m], marker=MARK.get(m, "o"), ms=6,
                    ls="--" if tree else "-", lw=1.4 if tree else 2.2, capsize=3, label=lab, alpha=0.95)
    ax.set_xscale("log"); ns = sorted(s.n.unique()); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Number of training rows (log scale)"); ax.set_ylabel(ylab)
    ax.set_title(TITLES.get(name, name) + f"\nmean ± std over {res['repeats']} random splits, fixed test set of {res['test_size']} rows per split\nerror bars = ±1 std; points slightly offset sideways so bars do not overlap", fontsize=9.5)
    if s.err.max() / s.err.min() > 3:  # wide range: log y so the leaders stay readable
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        ax.set_yscale("log"); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        sd = s.err_sd.fillna(0); lo = max((s.err - sd).min(), s.err.min() * 0.5) * 0.9; hi = (s.err + sd).max() * 1.1  # keep error bars inside the axes
        ticks = [t for t in [0.01,0.015,0.02,0.03,0.04,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1,1.5,2,3,4,5,6,8,10,15,20,30,40,50] if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_ylim(lo, hi); ax.set_ylabel(ylab + ", log scale")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    fig.tight_layout()
    out = f"{HERE}/charts/learning_curve_{name}.png"; fig.savefig(out, dpi=130); plt.close(fig); print(out)
