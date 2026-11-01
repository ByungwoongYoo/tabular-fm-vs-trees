"""Markdown tables + headline numbers from results/*.json. Usage: python make_tables.py name1 name2 ... > results/tables.md"""
import sys, json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TREES = ["RF", "HGB", "XGB-tuned"]
ORDER = ["Kumo", "Causilo", "FTFM-1", "LightPFN", "XGB-tuned", "HGB", "RF"]
head = {}
for name in sys.argv[1:]:
    r = json.load(open(f"{HERE}/results/{name}.json")); s = pd.read_csv(f"{HERE}/results/{name}_summary.csv"); task = r["task"]
    key = "RMSE" if task == "reg" else "AUC"; low = task == "reg"
    ns = sorted(s.n.unique()); models = [m for m in ORDER if m in set(s.model)]
    print(f"\n### {name}  ({r['n_rows']} rows, {r['n_features']} features, {task}, test {r['test_size']} rows, {r['repeats']} repeats; metric {key} mean ± std)\n")
    print("| Model | " + " | ".join(f"n={n}" for n in ns) + " |"); print("|---|" + "---|" * len(ns))
    best_by_n = {}
    for n in ns:
        sub = s[s.n == n]; best_by_n[n] = sub.loc[sub[key + "_mean"].idxmin() if low else sub[key + "_mean"].idxmax()].model
    for m in models:
        cells = []
        for n in ns:
            d = s[(s.model == m) & (s.n == n)]
            if d.empty: cells.append("–"); continue
            v, sd = d[key + "_mean"].iloc[0], d[key + "_std"].iloc[0]
            c = f"{v:.3f} ± {sd:.3f}" if task == "reg" else f"{v:.4f} ± {sd:.4f}"
            cells.append(f"**{c}**" if best_by_n[n] == m else c)
        print(f"| {m}{' (tree)' if m in TREES else ''} | " + " | ".join(cells) + " |")
    # FM vs best tree per size
    print("\n| n | best FM | best tree | FM vs tree |"); print("|---|---|---|---|")
    for n in ns:
        sub = s[s.n == n]; fm = sub[~sub.model.isin(TREES)]; tr = sub[sub.model.isin(TREES)]
        if fm.empty or tr.empty: continue
        f = fm.loc[fm[key + "_mean"].idxmin() if low else fm[key + "_mean"].idxmax()]; t = tr.loc[tr[key + "_mean"].idxmin() if low else tr[key + "_mean"].idxmax()]
        if low: rel = f"{100 * (f[key + '_mean'] / t[key + '_mean'] - 1):+.1f} % RMSE"
        else: rel = f"{100 * ((1 - f[key + '_mean']) / (1 - t[key + '_mean']) - 1):+.1f} % (1−AUC)"
        print(f"| {n} | {f.model} {f[key + '_mean']:.4g} | {t.model} {t[key + '_mean']:.4g} | {rel} |")
    print("\nData efficiency (rows the FM needs to match the best tree at the largest size; log-interpolated):\n")
    for fm, e in r["data_efficiency"].items():
        print(f"- {fm}: tree {e['tree_reference']} at n={e['tree_n']} scores {e['tree_score']:.4g}; FM matches at n≈{e['fm_n_to_match']} → ratio {e['data_ratio']}  {e.get('note','')}")
    st = r["model_status"]
    print("\nModel status: " + "; ".join(f"{m}: {v['status']}" + (f" ({v.get('first_error','')[:90].splitlines()[0]})" if v.get('first_error') else "") + (f" [{v['reason']}]" if v.get('reason') else "") for m, v in st.items()))
