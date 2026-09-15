"""Run the five replicate M_aug surrogates on the US-Ton reference-solver run (02_run_reference.jl),
for the temporal surrogate-vs-reference comparison of Sec. 2.6 / 3.3 (Figure 4c,d). Model-to-model
only (no observations), so this just adds each replicate's predictions to the reference table.

The pre-solver FvCB features must use the SAME fixed physiology the reference run used, not
surrogate_features.py's field-validation defaults -- this script overrides those module constants
from the input table first.

In:  output/uston_2020_2022_reference.csv   (from 02_run_reference.jl)
Out: output/uston_2020_2022_surrogate.csv   (adds Anet_repNN / gs_repNN per replicate)

MODEL_ROOT/FVCB_REPS follow the same convention as ../05_field_validation/03_run_surrogate.py.
Only replicate 1 is on Zenodo; set FVCB_REPS=rep01 to run Zenodo-only.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_field_validation"))
import surrogate_features as vb  # noqa: E402


def _log(msg):
    print(msg, flush=True)


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
REF_CSV = OUT / "uston_2020_2022_reference.csv"
PRED_CSV = OUT / "uston_2020_2022_surrogate.csv"
MODEL_ROOT = Path(os.environ.get("FVCB_MODELS_ROOT",
                                 Path(__file__).resolve().parents[1] / "03_training" / "out" / "Maug"))
REPS = os.environ.get("FVCB_REPS", "rep01,rep02,rep03,rep04,rep05").split(",")
TARGETS = ["Anet", "gs"]


def model_path(rep, target):
    own = MODEL_ROOT / rep / "n10000000" / "models" / f"stage2_{target}_sim_model.cbm"
    if own.is_file():
        return own
    return MODEL_ROOT / rep / "10m" / f"stage2_{target}_sim_model.cbm"


def build_features(df):
    T = df["T_air"].to_numpy(float)
    RH = df["RH"].to_numpy(float)
    PFD = df["PFD"].to_numpy(float)
    CO2 = df["CO2"].to_numpy(float)
    # the surrogate's pre-solver features must use the SAME physiology the reference run used
    vb.VCM25 = float(df["Vcm25"].iloc[0])
    vb.JM25 = float(df["Jm25"].iloc[0])
    vb.RD25 = float(df["Rd25"].iloc[0])
    vb.TP25 = float(df["Tp25"].iloc[0])
    _log(f"[surrogate] features use Vcm25={vb.VCM25} Jm25={vb.JM25} Rd25={vb.RD25} Tp25={vb.TP25}")
    Ac, Aj, Ap = vb.compute_init(T, RH, PFD, CO2)
    VPD = vb.es_sat(T) * (1.0 - RH / 100.0)
    X = pd.DataFrame({"Aj_init": Aj, "Ac_init": Ac, "Ap_init": Ap, "T_air": T, "RH": RH,
                      "wind": df["wind"], "PFD": PFD, "CO2": CO2, "w": df["w"], "VPD": VPD,
                      "Vcm25": df["Vcm25"], "Jm25": df["Jm25"], "g0": df["g0"], "g1": df["g1"],
                      "Rd25": df["Rd25"]})[vb.FEATURES]
    return X.to_numpy(np.float32)


def main():
    if not REF_CSV.exists():
        raise SystemExit(f"reference run not found: {REF_CSV}\nrun 02_run_reference.jl first")
    df = pd.read_csv(REF_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n0 = len(df)
    df = df.dropna(subset=["Anet_ref", "gs_ref"]).reset_index(drop=True)
    _log(f"[surrogate] {len(df):,} rows with a converged reference solution "
         f"({n0 - len(df)} dropped of {n0:,})")
    _log(f"[surrogate] {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

    X = build_features(df)
    for target in TARGETS:
        for rep in REPS:
            m = CatBoostRegressor()
            m.load_model(str(model_path(rep, target)))
            df[f"{target}_{rep}"] = m.predict(X)
        _log(f"[surrogate] {target}: {len(REPS)} replicate(s) predicted")

    df.to_csv(PRED_CSV, index=False)
    _log(f"[save] {PRED_CSV}")


if __name__ == "__main__":
    main()
