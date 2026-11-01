"""Headline 'data efficiency' numbers. For each dataset and each training size n_t, take the best tree model at n_t
(oracle choice among RF/HGB/XGB-tuned, chosen on test score) and find how many rows the single best foundation model
(the FM with the best average rank on that dataset) needs to reach the same score, by log-linear interpolation of its
mean learning curve. Ratio = n_t / n_fm. If the FM is already better at the smallest measured size, only a lower bound
(n_t / n_min) is known. Writes results/headline.json and prints a markdown table."""
import sys, json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TREES = ["RF", "HGB", "XGB-tuned"]
out = {}
print("| Dataset | Best FM | tree size n_t | best tree (score) | FM rows to match | data ratio |"); print("|---|---|---|---|---|---|")
for name in sys.argv[1:]:
    r = json.load(open(f"{HERE}/results/{name}.json")); s = pd.read_csv(f"{HERE}/results/{name}_summary.csv")
    key, low = ("RMSE_mean", True) if r["task"] == "reg" else ("AUC_mean", False)
    fms = s[~s.model.isin(TREES)]; tr = s[s.model.isin(TREES)]
    full = fms.groupby("model").n.nunique(); fms = fms[fms.model.isin(full[full == s.n.nunique()].index)]  # only FMs with a complete learning curve (e.g. superconduct Kumo was RSS-killed after n=100)
    rank = fms.groupby("n")[key].rank(ascending=low); fm = fms.assign(r=rank).groupby("model").r.mean().idxmin()
    f = fms[fms.model == fm].sort_values("n"); ns, vs = f.n.to_numpy(float), f[key].to_numpy(float)
    rows = []
    for nt in sorted(tr.n.unique()):
        t = tr[tr.n == nt]; b = t.loc[t[key].idxmin() if low else t[key].idxmax()]; target = b[key]
        ok = (vs <= target) if low else (vs >= target)
        if not ok.any(): nf, kind = None, "not reached"
        elif ok[0]: nf, kind = ns[0], "lower bound"
        else:
            i = int(np.argmax(ok)); x0, x1 = np.log(ns[i-1]), np.log(ns[i])
            nf, kind = float(np.exp(x0 + (target - vs[i-1]) * (x1 - x0) / (vs[i] - vs[i-1]))), "interpolated"
        ratio = None if nf is None else nt / nf
        rows.append({"n_tree": int(nt), "tree": b.model, "tree_score": float(target), "fm": fm, "fm_rows": nf, "ratio": ratio, "kind": kind})
        rs = "–" if ratio is None else (f"≥ {ratio:.2f}×" if kind == "lower bound" else f"{ratio:.2f}×")
        print(f"| {name} | {fm} | {nt} | {b.model} ({target:.4g}) | {'–' if nf is None else ('≤ ' if kind=='lower bound' else '') + f'{nf:.0f}'} | {rs} |")
    out[name] = rows
json.dump(out, open(f"{HERE}/results/headline.json", "w"), indent=1)
interp = [r["ratio"] for v in out.values() for r in v if r["kind"] == "interpolated"]
if interp:
    print(f"\nInterpolated ratios: n={len(interp)}, median {np.median(interp):.2f}×, min {min(interp):.2f}×, max {max(interp):.2f}×")
