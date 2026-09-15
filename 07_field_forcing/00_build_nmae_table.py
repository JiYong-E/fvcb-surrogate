"""Per-row, per-replicate normalized absolute error over the independent final test design,
with each row's input-side FvCB limitation regime -- the table 04_thresholds.py consumes, and
the same quantity that underlies Figure 3 and Figure 4a,b.

Two independent pieces, only one of which needs a model:

  regime, dcj   arithmetic on the pre-solver rates already stored in the test design
                (Ac_init, Aj_init, Ap_init). No inference, so this half reproduces exactly.
  nmae_*        |prediction - reference| / (reference range over the whole design) x 100,
                per target and per replicate. Needs the M_aug .cbm models.

The regime rule is Eq. 6 of the paper: TPU-limited when Ap is below both Ac and Aj; otherwise
the A_c/A_j boundary group when the boundary-proximity index D_cj falls below 0.15; otherwise
whichever of Ac, Aj is smaller. Thresholds other than 0.15 can be set with FVCB_DCJ.

Note on replicates: the Zenodo deposit carries replicate 1 only, so FVCB_REPS defaults to
rep01 and the resulting nmae_* columns are a single-replicate version of the manuscript's
five-replicate mean. The regime composition is unaffected. Replicates 2-5 are reproducible
from metadata/seeds.csv plus the training code.

In:  the independent final test design, as parquet parts (FVCB_TEST_DESIGN; the deposit's
     data/final_test/). Needs T_air, VPD, Ac_init, Aj_init, Ap_init, Anet_sim, gs_sim.
     M_aug models under FVCB_MODELS_ROOT, same layout as 03_run_surrogate.py.
Out: output/per_row_nmae.parquet  (T_air, VPD, regime, dcj, nmae_{Anet,gs}_rep*)

Run: python 00_build_nmae_table.py
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
TEST_DESIGN = Path(os.environ.get("FVCB_TEST_DESIGN", HERE / "final_test"))
MODEL_ROOT = Path(os.environ.get("FVCB_MODELS_ROOT",
                                 HERE.parent / "03_training" / "out" / "Maug"))
REPS = os.environ.get("FVCB_REPS", "rep01").split(",")
DCJ = float(os.environ.get("FVCB_DCJ", "0.15"))
EPS = 1e-9

# The models carry positional feature names ("0".."14"), so column ORDER is the contract.
# vb.FEATURES is that order, shared with 05_field_validation/03_run_surrogate.py.
FEATURES = vb.FEATURES
TARGETS = [("Anet", "Anet_sim"), ("gs", "gs_sim")]


def model_path(rep, target):
    own = MODEL_ROOT / rep / "n10000000" / "models" / f"stage2_{target}_sim_model.cbm"
    if own.is_file():
        return own
    return MODEL_ROOT / rep / "10m" / f"stage2_{target}_sim_model.cbm"


def classify(ac, aj, ap):
    """Eq. 6. Returns (regime labels, D_cj). Input-side rates only, never the solved state."""
    dcj = np.abs(ac - aj) / (np.abs(ac) + np.abs(aj) + EPS)
    regime = np.where(ac < aj, "Ac-limited", "Aj-limited")
    regime = np.where(dcj < DCJ, "Ac/Aj boundary", regime)
    regime = np.where((ap < ac) & (ap < aj), "Ap-limited", regime)   # TPU wins outright
    return regime, dcj


def main():
    # part-*.parquet only: the same directory also holds run_metadata / run_summary
    parts = sorted(TEST_DESIGN.glob("part-*.parquet"))
    if not parts:
        raise SystemExit(
            f"final test design not found: {TEST_DESIGN}\n"
            "Point FVCB_TEST_DESIGN at the deposit's data/final_test/ directory.")
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    _log(f"[nmae] test design: {len(df):,} rows from {len(parts)} parts")

    regime, dcj = classify(df["Ac_init"].to_numpy(float),
                           df["Aj_init"].to_numpy(float),
                           df["Ap_init"].to_numpy(float))
    out = pd.DataFrame({"T_air": df["T_air"], "VPD": df["VPD"],
                        "regime": regime, "dcj": dcj})
    share = pd.Series(regime).value_counts(normalize=True) * 100
    _log("[nmae] regime composition (%): "
         + "  ".join(f"{k}={v:.2f}" for k, v in share.items()))

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"test design is missing predictor columns: {missing}")
    X = df[FEATURES].to_numpy(np.float32)     # order matters: names are positional
    for rep in REPS:
        for target, refcol in TARGETS:
            p = model_path(rep, target)
            if not p.is_file():
                raise SystemExit(f"model not found: {p}\nSet FVCB_MODELS_ROOT / FVCB_REPS.")
            m = CatBoostRegressor()
            m.load_model(str(p))
            y = df[refcol].to_numpy(float)
            rng = y.max() - y.min()          # range over the whole design, as in Eq. 5
            out[f"nmae_{target}_{rep}"] = np.abs(m.predict(X) - y) / rng * 100.0
            _log(f"[nmae] {rep} {target:4s}: mean {out[f'nmae_{target}_{rep}'].mean():.4f}%"
                 f"  (reference range {rng:.4g})")

    OUT.mkdir(parents=True, exist_ok=True)
    q = OUT / "per_row_nmae.parquet"
    out.to_parquet(q, index=False)
    _log(f"[save] {q}  ({len(out):,} rows x {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
