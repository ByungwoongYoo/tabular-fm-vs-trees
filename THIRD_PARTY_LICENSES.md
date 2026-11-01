# Third-party licences

Nothing in this repository redistributes model weights. The benchmark downloads and runs the models
below, each under its own licence. The MIT licence of this repository covers our code, results and charts only.
Before using model weights or model-generated outputs commercially, check the checkpoint-specific licences.
Licences were read on 2026-11-01/02 from installed package metadata, Hugging Face model cards and the licence
files in the model repositories. This is a summary, not legal advice; re-check upstream before you rely on it.

| Model | Checkpoint used (Hugging Face revision) | Code licence | Weights licence | Commercial use of weights / outputs |
|---|---|---|---|---|
| Kumo-Tabular (NVIDIA, `structured-data-models` 0.2.1) | `nvidia/Kumo-Tabular` @ `3c3e10bb`, `large/` classifier and regressor | Apache-2.0 | OpenMDW-1.1 | Permitted under the licence terms. If you redistribute the weights, keep a copy of the licence and the copyright/origin notices. The licence states it imposes no restrictions on outputs |
| LightPFN 1.1.0 | `ueuegio/LightPFN` @ `bd389ab5` | Apache-2.0 | Apache-2.0 | Permitted under Apache-2.0 (keep the licence and any NOTICE file when redistributing). Classification only |
| Causilo 1.0.3 (Nums AI) | `nums-ai/causilo` @ `94f2bd91` | Apache-2.0 (PyPI package) | **Causilo License v1.0** (non-commercial) | **No**, without a separate licence: commercial or production use of the model, derivatives *or outputs* (§2(b)), and offering it as a hosted, API or SaaS service, paid or free (§2(e)). §2(c) allows publishing scholarly papers and research results, including outputs; it does not authorise commercial or production use of outputs outside scholarly communication |
| FTFM-1 0.1.0 (`ftfm` from git commit `7e5e237d`) | `josifgrabocka/ftfm` @ `a8362b12` | Apache-2.0 | **CC BY-NC 4.0** (`LICENSE-WEIGHTS.md`) | **No.** The public weight licence does not permit commercial use; separate written permission is needed. The model card allows research, benchmarking and publishing results |
| TabPFN 9.2.0 / TabPFN-3.5 weights (Prior Labs) | not downloaded | Apache-2.0 | **TabPFN-3.5 License v1.0** | **No** for the downloadable TabPFN-3.5 weights (non-commercial, non-production, also covering outputs and hosted use) without a commercial licence. This applies to TabPFN-3.5 only, not to other TabPFN versions or Prior Labs' API. **Not run in this benchmark** |
| scikit-learn 1.9.1 (RF, HGB) / XGBoost 3.4.1 | – | BSD-3-Clause / Apache-2.0 | none | Permitted |

## Datasets

All six datasets come from the UCI Machine Learning Repository and are licensed CC BY 4.0
(checked on each dataset page on 2026-11-02). Citations are in `data/README.md`.
