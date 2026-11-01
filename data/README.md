# Data

Six public datasets from the UCI Machine Learning Repository, each licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (licence shown on each dataset page, checked 2026-11-02).
The files here are lightly reshaped copies of the originals; the changes are listed per file.

| File | Rows | Target column | Changes from the UCI original | Source / citation |
|---|---|---|---|---|
| `concrete.csv` | 1,030 | `strength_mpa` | columns renamed | Yeh, I.-C. (1998). Concrete Compressive Strength. https://doi.org/10.24432/C5PK67 |
| `energy_heating.csv` | 768 | `heating_load` | cooling load (second target) dropped, columns renamed | Tsanas, A. & Xifara, A. (2012). Energy Efficiency. https://doi.org/10.24432/C51307 |
| `steel_faults.csv` | 1,941 | `fault_type` | the 7 one-hot fault columns merged into one label column | Buscema, M., Terzi, S. & Tastle, W. (2010). Steel Plates Faults. https://doi.org/10.24432/C5J88N |
| `wine_quality.csv` | 6,497 | `quality` | red and white files concatenated, `is_red` flag added | Cortez, P. et al. (2009). Wine Quality. https://doi.org/10.24432/C56S3T |
| `MagicTelescope.csv` | 19,020 | `class:` | none (column names as distributed) | Bock, R. (2004). MAGIC Gamma Telescope. https://doi.org/10.24432/C52C8B |
| `superconduct.csv.gz` | 21,263 | `critical_temp` | `train.csv` only, gzip-compressed (pandas reads it directly) | Hamidieh, K. (2018). Superconductivity Data. https://doi.org/10.24432/C53P47 |
