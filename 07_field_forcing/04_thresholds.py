"""US-Ton hot-drought thresholds, and the surrogate error / limitation-regime composition of the
independent test design under them -- the data behind Sec. 2.6 / 3.3, Figure 4c.

Three conditions: whole domain / T_air >= p90 / T_air + VPD >= p90 (both thresholds jointly).
Thresholds are the 90th percentiles of the real AmeriFlux US-Ton record, 2020-2022 June-October
daytime -- the conventional cutoff for hot extremes (Zhang et al. 2011; Perkins & Alexander 2013).

Scope: these are i.i.d. design rows selected to match drought-typical conditions, not a time
series -- 02_run_reference.jl / 03_run_surrogate.py cover the temporal-process question instead.

In:  the per-row, per-replicate NMAE table for the independent final test design (T_air, VPD,
     regime, nmae_{Anet,gs}_rep*; see NMAE_TABLE below), from 00_build_nmae_table.py.
     the raw AmeriFlux US-Ton record (as in 01_prepare_forcing.py), for the p90 thresholds only.
Out: output/uston_thresholds_conditions.csv

Run: python 04_thresholds.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _log(msg):
    print(msg, flush=True)


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
AMERIFLUX_DIR = Path(os.environ.get("AMERIFLUX_DIR", HERE / "ameriflux"))
FLUX = AMERIFLUX_DIR / "AMF_US-Ton_FLUXNET_FLUXMET_HH_2001-2025_v1.3_r1.csv"
NMAE_TABLE = Path(os.environ.get("FVCB_NMAE_TABLE", OUT / "per_row_nmae.parquet"))

REPS = None   # discovered from the table's own columns (see main)
YEARS = (2020, 2022)
MONTHS = (6, 10)
QUANT = 0.90
WHIS = (5, 95)

REGIMES = ["Ac-limited", "Aj-limited", "Ap-limited", "Ac/Aj boundary"]
TARGETS = ["Anet", "gs"]


def site_thresholds():
    d = pd.read_csv(FLUX, usecols=["TIMESTAMP_START", "TA_F", "VPD_F", "PPFD_IN"],
                    low_memory=False).replace(-9999, np.nan)
    ts = pd.to_datetime(d["TIMESTAMP_START"], format="%Y%m%d%H%M")
    m = (ts.dt.year.between(*YEARS) & ts.dt.month.between(*MONTHS) & (d["PPFD_IN"] > 0))
    d = d[m].dropna(subset=["TA_F", "VPD_F"])
    t90 = float(d["TA_F"].quantile(QUANT))
    v90 = float(d["VPD_F"].quantile(QUANT) / 10.0)  # VPD_F is hPa
    _log(f"[thresholds] US-Ton {YEARS[0]}-{YEARS[1]} Jun-Oct daytime, n={len(d):,}: "
         f"p{QUANT*100:.0f} T_air={t90:.2f} C, VPD={v90:.2f} kPa")
    return t90, v90


def main():
    if not NMAE_TABLE.exists():
        raise SystemExit(
            f"per-row NMAE table not found: {NMAE_TABLE}\n"
            "Build it first with 00_build_nmae_table.py, or point FVCB_NMAE_TABLE at an "
            "existing copy.")
    t90, v90 = site_thresholds()
    df = pd.read_parquet(NMAE_TABLE)
    # the deposit carries replicate 1 only, so take whatever replicates the table actually has
    reps = sorted(c.split("_")[-1] for c in df.columns if c.startswith("nmae_Anet_"))
    if not reps:
        raise SystemExit(f"no nmae_Anet_rep* columns in {NMAE_TABLE}")
    globals()["REPS"] = reps
    _log(f"[thresholds] test design: {len(df):,} rows, replicates: {', '.join(reps)}")
    # one value per row: the mean over replicates, so the mean of THIS is the manuscript's NMAE
    for target in TARGETS:
        df[f"{target}_row"] = df[[f"nmae_{target}_{r}" for r in REPS]].mean(axis=1)

    conds = [
        ("All conditions", np.ones(len(df), bool)),
        (f"T_air >= {t90:.1f} C", (df["T_air"] >= t90).to_numpy()),
        (f"T_air + VPD >= {v90:.1f} kPa", ((df["T_air"] >= t90) & (df["VPD"] >= v90)).to_numpy()),
    ]

    rows = []
    for label, mask in conds:
        sub = df[mask]
        rec = dict(condition=label, n=len(sub), frac_pct=len(sub) / len(df) * 100)
        for target in TARGETS:
            x = sub[f"{target}_row"].to_numpy()
            q = np.percentile(x, [WHIS[0], 25, 50, 75, WHIS[1]])
            rec.update({f"{target}_mean": float(x.mean()), f"{target}_median": q[2],
                        f"{target}_p25": q[1], f"{target}_p75": q[3],
                        f"{target}_p5": q[0], f"{target}_p95": q[4]})
        comp = sub["regime"].value_counts(normalize=True).reindex(REGIMES).fillna(0) * 100
        for r in REGIMES:
            rec[r] = float(comp[r])
        rows.append(rec)
        _log(f"[thresholds] {rec['condition']:24s} n={len(sub):7,} ({rec['frac_pct']:5.2f}%)  "
             + "  ".join(f"{t} mean={rec[f'{t}_mean']:.4f}" for t in TARGETS))
    tab = pd.DataFrame(rows)

    # is the error rise compositional, or within-regime? (log only, matches the Discussion text)
    whole, drought = df, df[conds[2][1]]
    for target in TARGETS:
        w_share = whole["regime"].value_counts(normalize=True).reindex(REGIMES)
        d_share = drought["regime"].value_counts(normalize=True).reindex(REGIMES)
        w_n = {r: whole.loc[whole.regime == r, f"{target}_row"].mean() for r in REGIMES}
        d_n = {r: drought.loc[drought.regime == r, f"{target}_row"].mean() for r in REGIMES}
        base, actual = whole[f"{target}_row"].mean(), drought[f"{target}_row"].mean()
        comp_only = sum(d_share[r] * w_n[r] for r in REGIMES)
        within_only = sum(w_share[r] * d_n[r] for r in REGIMES)
        _log(f"[thresholds] {target}: whole {base:.4f}% -> drought {actual:.4f}% "
             f"({actual-base:+.4f}); composition alone {comp_only:.4f}% ({comp_only-base:+.4f}), "
             f"within-regime alone {within_only:.4f}% ({within_only-base:+.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "uston_thresholds_conditions.csv"
    tab.to_csv(p, index=False)
    _log(f"[save] {p}")


if __name__ == "__main__":
    main()
