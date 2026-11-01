#!/usr/bin/env python3
"""bench_tabular.py — learning-curve benchmark: tabular foundation models (FMs) vs tree baselines.

Usage (orchestrator, any python with numpy+pandas+scikit-learn):
  python bench_tabular.py --csv data.csv --target y --task reg|cls --name mydata \
      [--sizes 250,500,1000,2000] [--repeats 3] [--test-size 1000] [--seed 0] \
      [--models RF,HGB,XGB-tuned,Causilo,LightPFN,FTFM-1,Kumo,TabPFN] \
      [--out results/] [--cores 2,3] [--max-rss-mb 3000] [--timeout-s 3600]

How it works
  * The CSV is loaded once; non-numeric feature columns are ordinal-encoded, missing feature
    values are filled with the training-split median (per split), rows with a missing target are dropped.
  * For every repeat r (seed = --seed + r) a fixed test set is drawn (stratified for cls); each
    training size n is drawn from the remaining pool (stratified for cls). Every model sees exactly
    the same train/test indices. Sizes larger than the pool are skipped.
  * Each model runs in its OWN python interpreter (its own venv, see MODEL_PYTHON) as a worker
    subprocess under nice/taskset with an RSS watchdog and a wall-clock timeout. A model whose venv
    is missing, whose import fails, which crashes, exceeds memory, or times out is recorded with a
    status/error instead of stopping the run; partial results are kept.
  * Output: <out>/<name>_raw.csv (one row per model x size x repeat), <out>/<name>_summary.csv
    (mean/std over repeats), <out>/<name>.json (config, model status, summary, data-efficiency).
Metrics: reg -> RMSE, MAE, R2 ; cls -> AUC (binary, or macro one-vs-rest), accuracy, log-loss.
FMs are used with package defaults (no tuning). XGB-tuned = randomized search (20 configs, 3-fold CV
on the training rows only). "Best tree" for each size is picked AFTER seeing test scores (oracle),
which favours the trees, i.e. the comparison is conservative for the FMs.
"""
import argparse, json, os, subprocess, sys, time, signal
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.environ.get("BENCH_VENVS", os.path.join(HERE, "venvs"))  # one venv per FM: venvs/{trees,causilo,lightpfn,ftfm,kumo,tabpfn}
# model name -> python interpreter (override with env BENCH_PY_<NAME>, e.g. BENCH_PY_KUMO=/path/python)
MODEL_PYTHON = {
    "RF": f"{SWEEP}/trees/bin/python",
    "HGB": f"{SWEEP}/trees/bin/python",
    "XGB-tuned": f"{SWEEP}/trees/bin/python",
    "Causilo": f"{SWEEP}/causilo/bin/python",
    "LightPFN": f"{SWEEP}/lightpfn/bin/python",
    "FTFM-1": f"{SWEEP}/ftfm/bin/python",
    "Kumo": f"{SWEEP}/kumo/bin/python",
    "TabPFN": f"{SWEEP}/tabpfn/bin/python",
}
TREES = ["RF", "HGB", "XGB-tuned"]
FMS = ["Causilo", "LightPFN", "FTFM-1", "Kumo", "TabPFN"]
CLS_ONLY = {"LightPFN"}
PRED_CHUNK = 500  # query rows per predict call (ICL models: rows are predicted independently)


def py_for(m):
    return os.environ.get("BENCH_PY_" + m.upper().replace("-", "_"), MODEL_PYTHON.get(m, sys.executable))


# ----------------------------------------------------------------------------- data + splits
def load_csv(path, target, task):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df[target].notna()].reset_index(drop=True)
    y_raw = df[target]
    Xdf = df.drop(columns=[target])
    cols = []
    for c in Xdf.columns:
        if not np.issubdtype(Xdf[c].dtype, np.number):
            Xdf[c] = Xdf[c].astype("category").cat.codes.replace(-1, np.nan)
        cols.append(c)
    X = Xdf.to_numpy(dtype=np.float32)
    if task == "cls":
        classes, y = np.unique(y_raw.astype(str).to_numpy(), return_inverse=True)
        classes = classes.tolist()
    else:
        y = y_raw.to_numpy(dtype=np.float64); classes = None
    return X, y, cols, classes


def make_splits(y, task, sizes, repeats, test_size, seed):
    from sklearn.model_selection import train_test_split
    N = len(y); idx = np.arange(N); splits = []
    for r in range(repeats):
        s = seed + r
        strat = y if task == "cls" else None
        pool, test = train_test_split(idx, test_size=test_size, random_state=s, stratify=strat)
        for n in sizes:
            if n > len(pool):
                continue
            if n == len(pool):
                tr = pool
            else:
                tr, _ = train_test_split(pool, train_size=n, random_state=s + 1000 * n,
                                         stratify=(y[pool] if task == "cls" else None))
            splits.append({"n": int(n), "rep": r, "seed": s, "train": np.sort(tr).tolist(), "test": np.sort(test).tolist()})
    # size-major order: if a slow/large model is killed at a big size, all smaller sizes are already complete
    return sorted(splits, key=lambda d: (d["n"], d["rep"]))


# ----------------------------------------------------------------------------- metrics
def metrics(task, yte, pred, K=None):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, accuracy_score, log_loss
    if task == "reg":
        pred = np.asarray(pred, dtype=float).reshape(-1)
        return {"RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "MAE": float(mean_absolute_error(yte, pred)),
                "R2": float(r2_score(yte, pred))}
    P = np.clip(np.asarray(pred, dtype=float), 1e-7, 1); P = P / P.sum(1, keepdims=True)
    if K == 2:
        auc = roc_auc_score(yte, P[:, 1])
    else:
        auc = roc_auc_score(yte, P, multi_class="ovr", average="macro", labels=list(range(K)))
    return {"AUC": float(auc), "ACC": float(accuracy_score(yte, P.argmax(1))), "LOGLOSS": float(log_loss(yte, P, labels=list(range(K))))}


# ----------------------------------------------------------------------------- model adapters (worker side)
def build(name, task, seed, threads):
    """Return an object with fit(X,y) and predict(X) (reg) / predict_proba(X) (cls, columns = sorted train labels)."""
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return (RandomForestClassifier if task == "cls" else RandomForestRegressor)(n_estimators=500, n_jobs=threads, random_state=seed)
    if name == "HGB":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        return (HistGradientBoostingClassifier if task == "cls" else HistGradientBoostingRegressor)(random_state=seed)
    if name == "XGB-tuned":
        import xgboost as xgb
        from scipy.stats import loguniform, randint, uniform
        from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
        dist = {"n_estimators": randint(100, 1000), "learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 9),
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": loguniform(1, 10),
                "reg_lambda": loguniform(0.1, 10)}
        if task == "cls":
            base = xgb.XGBClassifier(n_jobs=threads, random_state=seed, tree_method="hist")
            cv = StratifiedKFold(3, shuffle=True, random_state=seed); scoring = "neg_log_loss"
        else:
            base = xgb.XGBRegressor(n_jobs=threads, random_state=seed, tree_method="hist")
            cv = KFold(3, shuffle=True, random_state=seed); scoring = "neg_root_mean_squared_error"
        return RandomizedSearchCV(base, dist, n_iter=20, cv=cv, scoring=scoring, random_state=seed, n_jobs=1, refit=True)
    if name == "Causilo":
        from causilo import CausiloClassifier, CausiloRegressor
        return (CausiloClassifier if task == "cls" else CausiloRegressor)(n_estimators=4, random_state=seed, device="cpu")
    if name == "LightPFN":
        if task != "cls":
            raise NotImplementedError("LightPFN is classification-only")
        from lightpfn import LightPFNClassifier
        return LightPFNClassifier(device="cpu", seed=seed)
    if name == "FTFM-1":
        from ftfm import FTFMClassifier, FTFMRegressor
        return (FTFMClassifier if task == "cls" else FTFMRegressor)(device="cpu", random_state=seed)
    if name == "Kumo":
        return KumoAdapter(task, seed)
    if name == "TabPFN":
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        return (TabPFNClassifier if task == "cls" else TabPFNRegressor)(device="cpu", n_estimators=4, random_state=seed)
    raise ValueError("unknown model " + name)


class KumoAdapter:
    """NVIDIA Kumo-Tabular (package `sdm`): in-context model; regression output = 999 quantiles -> we use their mean."""
    def __init__(self, task, seed):
        import torch, sdm
        self.torch, self.sdm, self.task, self.seed = torch, sdm, task, seed
        self.m = sdm.models.KumoTabular(task="classification" if task == "cls" else "regression", device=torch.device("cpu"))

    def fit(self, X, y):
        self.X, self.y = X, y; self.classes_ = np.unique(y); return self

    def _run(self, Xq):
        import pandas as pd
        torch, sdm = self.torch, self.sdm
        n = len(self.X)
        df = pd.DataFrame(np.vstack([self.X, Xq]), columns=[f"f{i}" for i in range(self.X.shape[1])])
        df["target"] = np.concatenate([self.y, np.zeros(len(Xq), dtype=self.y.dtype) + (self.y[0] if self.task == "cls" else 0)])
        if self.task == "cls":
            df["target"] = df["target"].astype(int).astype(str)
        tab = sdm.TableTensor.from_pandas(df=df, stypes=sdm.infer_stypes(df, overrides={"target": "categorical" if self.task == "cls" else "numerical"}),
                                          device=torch.device("cpu"))
        with torch.no_grad():
            q = self.m(x_context=tab[:n].drop_columns("target"), y_context=tab[:n, "target"], x_query=tab[n:].drop_columns("target"),
                       num_estimators=4, generator=torch.Generator().manual_seed(self.seed))
        return q.to_pandas()

    def predict(self, Xq):
        return self._run(Xq).to_numpy(dtype=float).mean(1)

    def predict_proba(self, Xq):
        qd = self._run(Xq)
        return qd[[str(int(c)) for c in self.classes_]].to_numpy(dtype=float)  # select by NAME (column order is not guaranteed)


def chunked(fn, X):
    return np.concatenate([np.asarray(fn(X[a:a + PRED_CHUNK])) for a in range(0, len(X), PRED_CHUNK)], 0)



def run_worker(spec_path, model, out_jsonl):
    spec = json.load(open(spec_path)); d = np.load(spec["npz"])
    X, y, task, K = d["X"], d["y"], spec["task"], spec.get("K")
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    try:
        import torch; torch.set_num_threads(threads)
    except Exception:
        pass
    done = set()
    if os.path.exists(out_jsonl):
        for l in open(out_jsonl):
            r = json.loads(l); done.add((r["n"], r["rep"]))
    for sp in spec["splits"]:
        if (sp["n"], sp["rep"]) in done:
            continue
        tr, te = np.array(sp["train"]), np.array(sp["test"])
        Xtr, Xte, ytr, yte = X[tr].copy(), X[te].copy(), y[tr], y[te]
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
        for A in (Xtr, Xte):
            ii = np.where(np.isnan(A)); A[ii] = np.take(med, ii[1])
        row = {"model": model, "n": sp["n"], "rep": sp["rep"], "seed": sp["seed"], "n_test": len(te)}
        t0 = time.time()
        try:
            m = build(model, task, sp["seed"], threads)
            if task == "cls":
                labs = np.unique(ytr); ytr_enc = np.searchsorted(labs, ytr)
                m.fit(Xtr, ytr_enc); t1 = time.time()
                p = chunked(m.predict_proba, Xte)
                P = np.zeros((len(te), K)); P[:, labs] = p
                row.update(metrics(task, yte, P, K))
            else:
                m.fit(Xtr, ytr); t1 = time.time()
                row.update(metrics(task, yte, chunked(m.predict, Xte)))
            row["fit_s"] = round(t1 - t0, 2); row["predict_s"] = round(time.time() - t1, 2); row["status"] = "ok"
            if model == "XGB-tuned":
                row["best_params"] = {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in m.best_params_.items()}
        except Exception as e:
            row.update({"status": "error", "error": (type(e).__name__ + ": " + str(e))[:600], "fit_s": round(time.time() - t0, 2)})
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            # a model that cannot even be built/imported will fail for every split: stop early
            if isinstance(e, (ImportError, NotImplementedError)) or "License" in type(e).__name__:
                return
            continue
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "best_params"}), flush=True)


# ----------------------------------------------------------------------------- orchestrator
def rss_tree_mb(pid):
    try:
        out = subprocess.run(["ps", "-o", "pid=,ppid=,rss=", "-A"], capture_output=True, text=True).stdout
    except Exception:
        return 0
    kids, rss = {}, {}
    for l in out.split("\n"):
        p = l.split()
        if len(p) == 3:
            kids.setdefault(int(p[1]), []).append(int(p[0])); rss[int(p[0])] = int(p[2])
    tot, st = 0, [pid]
    while st:
        q = st.pop(); tot += rss.get(q, 0); st += kids.get(q, [])
    return tot // 1024


def launch(model, spec_path, out_jsonl, log_path, cores, max_rss, timeout, threads):
    py = py_for(model)
    if not os.path.exists(py):
        return {"status": "unavailable", "reason": f"interpreter not found: {py}"}
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED="1", TABPFN_DISABLE_TELEMETRY="1")
    cmd = ["nice", "-n", "10"] + (["taskset", "-c", cores] if cores else []) + [py, os.path.abspath(__file__), "--worker", model,
                                                                               "--spec", spec_path, "--wout", out_jsonl]
    t0, peak, why = time.time(), 0, None
    with open(log_path, "a") as lf:
        lf.write(f"\n### {time.strftime('%F %T %Z')} {' '.join(cmd)}\n"); lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while p.poll() is None:
            mb = rss_tree_mb(p.pid); peak = max(peak, mb)
            if mb > max_rss:
                why = f"killed: RSS {mb} MB > limit {max_rss} MB"
            elif time.time() - t0 > timeout:
                why = f"killed: timeout {timeout} s"
            if why:
                os.killpg(p.pid, signal.SIGKILL); p.wait(); break
            time.sleep(1)
    res = {"status": "finished" if (p.returncode == 0 and not why) else "failed", "returncode": p.returncode,
           "wall_s": round(time.time() - t0, 1), "peak_rss_mb": peak}
    if why:
        res["reason"] = why
    return res


def summarise(rows, task):
    import pandas as pd
    df = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    if df.empty:
        return df
    mets = ["RMSE", "MAE", "R2"] if task == "reg" else ["AUC", "ACC", "LOGLOSS"]
    agg = df.groupby(["model", "n"]).agg(**{f"{m}_mean": (m, "mean") for m in mets}, **{f"{m}_std": (m, "std") for m in mets},
                                         reps=("rep", "count"), fit_s_mean=("fit_s", "mean"), predict_s_mean=("predict_s", "mean")).reset_index()
    return agg


def data_efficiency(agg, task):
    """For each FM: smallest training size at which it matches the best tree's score at the largest common size
    (log-linear interpolation between measured sizes). Returns ratio tree_n / fm_n."""
    if agg is None or len(agg) == 0:
        return {}
    key, better_low = ("RMSE_mean", True) if task == "reg" else ("AUC_mean", False)
    trees = agg[agg.model.isin(TREES)]
    if trees.empty:
        return {}
    best = trees.loc[trees.groupby("n")[key].idxmin() if better_low else trees.groupby("n")[key].idxmax()]
    out = {}
    for fm in agg.model.unique():
        if fm in TREES:
            continue
        f = agg[agg.model == fm].sort_values("n")
        common = sorted(set(f.n) & set(best.n))
        if not common:
            continue
        nmax = common[-1]; target = float(best[best.n == nmax][key].iloc[0]); tname = best[best.n == nmax].model.iloc[0]
        ns, vs = f.n.to_numpy(float), f[key].to_numpy(float)
        ok = (vs <= target) if better_low else (vs >= target)
        rec = {"tree_reference": tname, "tree_n": int(nmax), "tree_score": target, "metric": key.replace("_mean", "")}
        if not ok.any():
            rec.update({"fm_n_to_match": None, "data_ratio": None, "note": "FM never reached the best-tree score at measured sizes"})
        else:
            i = int(np.argmax(ok))
            if i == 0:
                n_match = ns[0]; rec["note"] = f"FM already matches at the smallest size measured ({int(ns[0])}); ratio is a lower bound"
            else:
                x0, x1, y0, y1 = np.log(ns[i - 1]), np.log(ns[i]), vs[i - 1], vs[i]
                n_match = float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
            rec.update({"fm_n_to_match": round(n_match, 1), "data_ratio": round(nmax / n_match, 2)})
        out[fm] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv"); ap.add_argument("--target"); ap.add_argument("--task", choices=["reg", "cls"])
    ap.add_argument("--name"); ap.add_argument("--sizes", default="250,500,1000,2000"); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--test-size", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(TREES + FMS)); ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--cores", default=None, help="taskset CPU list, e.g. 2,3"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-rss-mb", type=int, default=3000); ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--model-sizes", default="", help="per-model size caps, e.g. Kumo=1000;FTFM-1=1000")
    ap.add_argument("--worker"); ap.add_argument("--spec"); ap.add_argument("--wout")
    a = ap.parse_args()
    if a.worker:
        return run_worker(a.spec, a.worker, a.wout)
    for req in ("csv", "target", "task"):
        if getattr(a, req) is None:
            ap.error(f"--{req} is required")
    name = a.name or os.path.splitext(os.path.basename(a.csv))[0]
    os.makedirs(a.out, exist_ok=True); wdir = os.path.join(a.out, name + "_work"); os.makedirs(wdir, exist_ok=True)
    X, y, cols, classes = load_csv(a.csv, a.target, a.task)
    K = len(classes) if classes else None
    sizes = [int(s) for s in a.sizes.split(",")]
    test_size = min(a.test_size, len(y) // 3)
    splits = make_splits(y, a.task, sizes, a.repeats, test_size, a.seed)
    npz = os.path.join(wdir, "data.npz"); np.savez(npz, X=X, y=y)
    caps = dict(kv.split("=") for kv in a.model_sizes.split(";") if kv)
    models = [m for m in a.models.split(",") if m]
    status = {}
    print(f"[{name}] N={len(y)} features={X.shape[1]} task={a.task} K={K} test={test_size} sizes={sorted(set(s['n'] for s in splits))} reps={a.repeats}", flush=True)
    for m in models:
        if a.task == "reg" and m in CLS_ONLY:
            status[m] = {"status": "skipped", "reason": "classification-only model"}; continue
        cap = int(caps[m]) if m in caps else None
        spec = {"npz": npz, "task": a.task, "K": K, "splits": [s for s in splits if cap is None or s["n"] <= cap]}
        sp = os.path.join(wdir, f"spec_{m}.json"); json.dump(spec, open(sp, "w"))
        st = launch(m, sp, os.path.join(wdir, f"{m}.jsonl"), os.path.join(wdir, f"{m}.log"), a.cores, a.max_rss_mb, a.timeout_s, a.threads)
        if cap:
            st["size_cap"] = cap
        status[m] = st; print(f"[{name}] {m}: {st}", flush=True)
    # collect
    rows = []
    for m in models:
        f = os.path.join(wdir, f"{m}.jsonl")
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
    for m in models:
        errs = [r for r in rows if r["model"] == m and r.get("status") == "error"]
        oks = [r for r in rows if r["model"] == m and r.get("status") == "ok"]
        if m in status:
            status[m]["n_ok"] = len(oks); status[m]["n_err"] = len(errs)
            if errs:
                status[m]["first_error"] = errs[0]["error"]
            if not oks and status[m].get("status") == "finished":
                status[m]["status"] = "unavailable"
    import pandas as pd
    pd.DataFrame(rows).drop(columns=["best_params"], errors="ignore").to_csv(os.path.join(a.out, name + "_raw.csv"), index=False)
    agg = summarise(rows, a.task)
    agg.to_csv(os.path.join(a.out, name + "_summary.csv"), index=False)
    res = {"dataset": name, "csv": os.path.abspath(a.csv), "target": a.target, "task": a.task, "n_rows": int(len(y)), "n_features": int(X.shape[1]),
           "classes": classes, "test_size": test_size, "sizes": sorted(set(s["n"] for s in splits)), "repeats": a.repeats, "seed": a.seed,
           "finished": time.strftime("%F %T %Z"), "model_status": status,
           "summary": json.loads(agg.to_json(orient="records")) if len(agg) else [], "data_efficiency": data_efficiency(agg, a.task),
           "xgb_best_params": [{"n": r["n"], "rep": r["rep"], **r["best_params"]} for r in rows if r.get("best_params")]}
    json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    print(json.dumps(res["data_efficiency"], indent=1))


if __name__ == "__main__":
    main()
