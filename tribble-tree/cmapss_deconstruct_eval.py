"""Stage B: NASA N-CMAPSS DS02 turbofan RUL, the real dataset the whiteboard
sketches were about.

`T_dev`/`T_test` in the N-CMAPSS file are exactly the whiteboard's
unobservable leaf-level health parameters (`fan_eff_mod, fan_flow_mod,
LPC_eff_mod, LPC_flow_mod, HPC_eff_mod, HPC_flow_mod, HPT_eff_mod,
HPT_flow_mod, LPT_eff_mod, LPT_flow_mod`); `X_s`/`X_v` are the physical/
virtual sensors; `Y` is RUL. The topology below (RUL -> HP/LP -> component
-> flow/eff leaf -> sensors) matches the second whiteboard photo exactly.

The sensor -> component grouping is a *domain-informed starting proposal*
from turbofan station numbers, NOT a verified fact -- several sensors sit at
component boundaries. Fold 1 below (each leaf vs. its own true health
parameter) is exactly the check on whether this grouping is any good; treat
its per-leaf R^2 as the thing to look at before trusting Fold 2's numbers.

Requires the optional `cmapss` dependency group (h5py):
    uv run --extra cmapss python tribble-tree/cmapss_deconstruct_eval.py

Run:
    uv run --extra cmapss python tribble-tree/cmapss_deconstruct_eval.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from fuzzytree import DeconstructedHierarchicalRegressor, HierarchicalFuzzyExpertsRegressor
from tribblefis.gaussian_regressor import TribbleRegressor

DATA_PATH = "/home/scott/PycharmProjects/grad-school/NASA-CMAPSS/N-CMAPSS_DS02-006.h5"

N_TRAIN = 40_000
N_TEST = 10_000
SEED = 42

FLIGHT_COLS = ["alt", "Mach", "TRA", "T2"]

# Domain-informed starting proposal (turbofan station numbers) -- see module
# docstring. Every X_s/X_v sensor is assigned to exactly one component.
COMPONENT_SENSORS = {
    "FAN": ["P2", "P15", "P21", "Nf", "SmFan", "W21"],
    "LPC": ["T24", "P24", "W22", "W25", "SmLPC"],
    "HPC": ["T30", "Ps30", "P30", "Nc", "W31", "SmHPC"],
    "HPT": ["T48", "T40", "Wf", "W32", "P40"],
    "LPT": ["P50", "T50", "P45", "W48", "W50", "phi"],
}

TOPOLOGY = {
    "RUL": ["HP", "LP"],
    "HP": ["HPT", "HPC"],
    "LP": ["FAN", "LPC", "LPT"],
    "HPT": ["HPT_flow", "HPT_eff"],
    "HPC": ["HPC_flow", "HPC_eff"],
    "FAN": ["FAN_flow", "FAN_eff"],
    "LPC": ["LPC_flow", "LPC_eff"],
    "LPT": ["LPT_flow", "LPT_eff"],
}
for _comp, _sensors in COMPONENT_SENSORS.items():
    TOPOLOGY[f"{_comp}_flow"] = _sensors + FLIGHT_COLS
    TOPOLOGY[f"{_comp}_eff"] = _sensors + FLIGHT_COLS

# leaf node name -> its T_var health-parameter column name
LEAF_TO_T_COLUMN = {
    "FAN_flow": "fan_flow_mod", "FAN_eff": "fan_eff_mod",
    "LPC_flow": "LPC_flow_mod", "LPC_eff": "LPC_eff_mod",
    "HPC_flow": "HPC_flow_mod", "HPC_eff": "HPC_eff_mod",
    "HPT_flow": "HPT_flow_mod", "HPT_eff": "HPT_eff_mod",
    "LPT_flow": "LPT_flow_mod", "LPT_eff": "LPT_eff_mod",
}


def _load_split(h5, split: str, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    n_total = h5[f"Y_{split}"].shape[0]
    idx = np.sort(rng.choice(n_total, size=min(n_rows, n_total), replace=False))

    def cols(group, names):
        arr = h5[f"{group}_{split}"][:]
        names_arr = [n.decode() for n in h5[f"{group}_var"][:]]
        return pd.DataFrame(arr[idx][:, [names_arr.index(n) for n in names]], columns=names)

    x_s = cols("X_s", [n.decode() for n in h5["X_s_var"][:]])
    x_v = cols("X_v", [n.decode() for n in h5["X_v_var"][:]])
    w = cols("W", [n.decode() for n in h5["W_var"][:]])
    t = cols("T", [n.decode() for n in h5["T_var"][:]])
    y = pd.Series(h5[f"Y_{split}"][:][idx].ravel().astype(float), name="RUL")
    df = pd.concat([x_s, x_v, w, t, y], axis=1)
    return df


def load_data():
    import h5py

    print(f"Loading {DATA_PATH} ...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    with h5py.File(DATA_PATH, "r") as h5:
        train = _load_split(h5, "dev", N_TRAIN, rng)
        test = _load_split(h5, "test", N_TEST, rng)
    print(f"  loaded {len(train)} train / {len(test)} test rows in {time.time() - t0:.1f}s\n")
    return train, test


def report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<56} R2={r2:6.3f}   RMSE={rmse:7.2f}")
    return r2, rmse


def main():
    train, test = load_data()
    sensor_cols = [c for comp in COMPONENT_SENSORS.values() for c in comp] + FLIGHT_COLS
    X_tr, X_te = train[sensor_cols], test[sensor_cols]
    y_tr, y_te = train["RUL"].to_numpy(), test["RUL"].to_numpy()

    print("=" * 82)
    print("FOLD 1: sensor -> leaf specificity (each leaf vs. its own true health parameter)")
    print("=" * 82)
    leaf_targets_tr = {leaf: train[col].to_numpy() for leaf, col in LEAF_TO_T_COLUMN.items()}
    fold1 = DeconstructedHierarchicalRegressor(
        flat_regressor_kwargs={"n_output_buckets": 4, "top_n": -1, "random_state": SEED},
    ).fit(X_tr, y_tr, TOPOLOGY, leaf_targets=leaf_targets_tr)
    for leaf in fold1.root_.iter_leaves():
        leaf_pred = fold1._predict_node(leaf, X_te)
        report(f"leaf {leaf.name:<10} vs. true {LEAF_TO_T_COLUMN[leaf.name]}", test[LEAF_TO_T_COLUMN[leaf.name]], leaf_pred)

    print("\n" + "=" * 82)
    print("FOLD 2: leaf -> root RUL")
    print("=" * 82)

    flat = TribbleRegressor(n_output_buckets=4, tsk_order="1st", top_n=-1, random_state=SEED).fit(X_tr, y_tr)
    report("Flat TRIBBLE on all sensors", y_te, flat.predict(X_te))

    hme = HierarchicalFuzzyExpertsRegressor(
        criterion="variance", max_depth=2, n_gate_terms=2, top_n=6,
        min_soft_count=100, min_expert_samples=150,
        expert_kwargs={"n_output_buckets": 4, "tsk_order": "1st"},
    ).fit(X_tr, y_tr)
    report("HME (auto topology)", y_te, hme.predict(X_te))

    fold2_no_oracle = DeconstructedHierarchicalRegressor(
        flat_regressor_kwargs={"n_output_buckets": 4, "top_n": -1, "random_state": SEED},
    ).fit(X_tr, y_tr, TOPOLOGY)
    report("Deconstructed tree (leaves supervised only on RUL)", y_te, fold2_no_oracle.predict(X_te))

    report("Deconstructed tree (leaves supervised on true health params)", y_te, fold1.predict(X_te))


if __name__ == "__main__":
    main()
