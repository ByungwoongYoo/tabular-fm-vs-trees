#!/usr/bin/env bash
# Re-runs the full benchmark with the settings used for the published results.
# Needs: venvs/trees (numpy, pandas, scikit-learn, scipy, xgboost) and one venv per foundation model
# under venvs/<model>/ (see README "Reproduce"). Override locations with BENCH_VENVS / BENCH_PY_<MODEL>.
# CPU pinning: cores 2,3 were used on an 8-core box; change CORES for your machine.
set -euo pipefail
cd "$(dirname "$0")"
CORES=${CORES:-2,3}
B="nice -n 10 taskset -c $CORES python3 bench_tabular.py --cores $CORES --max-rss-mb 3000 --out results"
M=RF,HGB,XGB-tuned,Causilo,LightPFN,FTFM-1,Kumo,TabPFN
$B --csv data/concrete.csv       --target strength_mpa  --task reg --name concrete       --sizes 100,250,500,730          --repeats 5 --test-size 300  --models $M
$B --csv data/energy_heating.csv --target heating_load  --task reg --name energy_heating --sizes 100,250,500              --repeats 5 --test-size 268  --models $M  # capped to 256 (N/3)
$B --csv data/steel_faults.csv   --target fault_type    --task cls --name steel_faults   --sizes 100,250,500,1000,1500    --repeats 3 --test-size 441  --models $M
$B --csv data/wine_quality.csv   --target quality       --task reg --name wine_quality   --sizes 100,250,500,1000,2000    --repeats 3 --test-size 1000 --models $M --timeout-s 3000
$B --csv data/MagicTelescope.csv --target class:        --task cls --name MagicTelescope --sizes 100,250,500,1000,2000    --repeats 3 --test-size 1000 --models $M --timeout-s 3000
$B --csv data/superconduct.csv.gz --target critical_temp --task reg --name superconduct  --sizes 100,250,500,1000,2000    --repeats 3 --test-size 1000 --models $M --timeout-s 3000
NAMES="concrete energy_heating steel_faults wine_quality MagicTelescope superconduct"
python3 make_tables.py $NAMES > results/tables.md
python3 make_headline.py $NAMES > results/headline.md
python3 make_charts.py $NAMES
python3 make_summary_chart.py
