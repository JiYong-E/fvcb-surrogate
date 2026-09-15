"""Run the M_aug surrogate on a field record already calibrated by 02_fit_reference.jl, and
summarize all five records into the field-validation metrics table.

usage: python 03_run_surrogate.py <stem>     # stem = pine2011, pine2013, ..., pine (=2019)
       python 03_run_surrogate.py --summarize

per-record: reads   output/<stem>_input_litdefault.csv    (forcing + fixed physiology columns)
                     output/<stem>_c3bb_calibrated.csv      (Date, Anet_obs, period, Anet_c3bb_cal, Vcm25/Jm25/Rd25)
            writes  output/<stem>_external_validation.csv  (adds Anet_pred_mean, Anet_pred_rep0x)
            prints  reference-vs-observation and surrogate-vs-reference metrics for the record

summarize: reads each record's output/<stem>_external_validation.csv, writes
           output/pine_field_metrics.csv (one row per record, mean +- 1 SD across replicates)

MODEL_ROOT must hold <rep>/n10000000/models/stage2_Anet_sim_model.cbm for each replicate
(the ../03_training/ output layout) or the Zenodo deposit's <rep>/10m/stage2_Anet_sim_model.cbm.
Only replicate 1 is on Zenodo; the paper's Figure 5 uses the mean of all five, so reproduce
the rest with ../03_training/ or set FVCB_REPS=rep01 to run Zenodo-only.
"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
import surrogate_features as vb

OUT = Path(__file__).resolve().parent / "output"
MODEL_ROOT = Path(os.environ.get("FVCB_MODELS_ROOT",
                                 Path(__file__).resolve().parents[1] / "03_training" / "out" / "Maug"))
REPS = os.environ.get("FVCB_REPS", "rep01,rep02,rep03,rep04,rep05").split(",")
RECORDS = [("pine2011", 2011), ("pine2013", 2013), ("pine2015", 2015),
           ("pine2017", 2017), ("pine", 2019)]
GAP_CAP_S = 3600.0     # a single instantaneous flux may stand for at most one hour


def model_path(rep):
    """<rep>/n10000000/models/stage2_Anet_sim_model.cbm for a self-trained run, or the
    Zenodo deposit's <rep>/10m/stage2_Anet_sim_model.cbm (no "models" subfolder)."""
    own = MODEL_ROOT / rep / "n10000000" / "models" / "stage2_Anet_sim_model.cbm"
    if own.is_file():
        return own
    return MODEL_ROOT / rep / "10m" / "stage2_Anet_sim_model.cbm"


def nse(p, y):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def gapdt(dates):
    # Time step per reading for the cumulative integral. Gaps > GAP_CAP_S (2011 has an 86h
    # outage) get the nominal spacing instead of standing for the whole gap. The identical
    # interval vector is applied to obs/reference/surrogate.
    dt = dates.diff().dt.total_seconds().to_numpy().copy()
    nom = np.nanmedian(dt[1:][(dt[1:] > 0) & (dt[1:] < 3600)])
    dt[0] = nom
    return np.where(np.isnan(dt) | (dt <= 0) | (dt > GAP_CAP_S), nom, dt)


def cumulative_total(dates, values):
    return float(np.sum(values * gapdt(pd.Series(dates))) / 1e6)


def run_one(stem):
    cal = pd.read_csv(OUT / f"{stem}_c3bb_calibrated.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    inp = pd.read_csv(OUT / f"{stem}_input_litdefault.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    assert np.allclose(cal["Anet_obs"], inp["Anet_obs"], atol=1e-6)

    fit = (float(cal["Vcm25"][0]), float(cal["Jm25"][0]), float(cal["Rd25"][0]))
    print(f"[{stem}] fitted: Vcmax25={fit[0]:.2f} Jmax25={fit[1]:.2f} Rd25={fit[2]:.3f}")

    T = inp["T_air"].to_numpy(float); RH = inp["RH"].to_numpy(float)
    PFD = inp["PFD"].to_numpy(float); CO2 = inp["CO2"].to_numpy(float)
    VPD = vb.es_sat(T) * (1.0 - RH / 100.0)
    obs = cal["Anet_obs"].to_numpy(float); ref = cal["Anet_c3bb_cal"].to_numpy(float)
    period = cal["period"].to_numpy()
    val = period != "cal"          # "val" for a split record, "all" for a whole-record fit
    dt = gapdt(cal["Date"])
    print(f"[{stem}] n={len(cal)}  (calibration {np.sum(~val)}, evaluated {np.sum(val)})\n")

    vb.VCM25, vb.JM25, vb.RD25 = fit
    Ac, Aj, Ap = vb.compute_init(T, RH, PFD, CO2)
    X = pd.DataFrame({"Aj_init": Aj, "Ac_init": Ac, "Ap_init": Ap, "T_air": T, "RH": RH,
                      "wind": inp["wind"], "PFD": PFD, "CO2": CO2, "w": inp["w"], "VPD": VPD,
                      "Vcm25": fit[0], "Jm25": fit[1], "g0": inp["g0"], "g1": inp["g1"],
                      "Rd25": fit[2]})[vb.FEATURES].to_numpy(np.float32)

    surr = []
    for rep in REPS:
        m = CatBoostRegressor()
        m.load_model(str(model_path(rep)))
        surr.append(m.predict(X))
    surr = np.array(surr)

    sv = pd.DataFrame({"Date": cal["Date"], "Anet_obs": obs, "Anet_c3bb_cal": ref, "period": period,
                       "Anet_pred_mean": surr.mean(0)})
    for i, rep in enumerate(REPS):
        sv[f"Anet_pred_{rep}"] = surr[i]
    sv.to_csv(OUT / f"{stem}_external_validation.csv", index=False)
    print(f"[save] {OUT / (stem + '_external_validation.csv')}\n")

    if (period == "all").all():
        blocks = [("evaluation (whole record)", val)]
    else:
        blocks = [("validation (held-out)", val), ("calibration", ~val)]
    for lbl, mask in blocks:
        y = obs[mask]; rng = y.max() - y.min()
        rf = ref[mask]; sm = surr[:, mask]; rrng = rf.max() - rf.min()
        s_nse = [nse(sm[i], y) for i in range(len(REPS))]
        sr_r2 = [nse(sm[i], rf) for i in range(len(REPS))]
        sr_nr = [100 * np.sqrt(np.mean((sm[i] - rf) ** 2)) / rrng for i in range(len(REPS))]
        cobs = np.sum(y * dt[mask]) / 1e6; cref = np.sum(rf * dt[mask]) / 1e6
        csur = [np.sum(sm[i] * dt[mask]) / 1e6 for i in range(len(REPS))]
        print(f"[{lbl}]  n={mask.sum()}")
        print(f"  reference vs obs : NSE={nse(rf, y):+.3f}  NRMSE={100*np.sqrt(np.mean((rf-y)**2))/rng:.1f}%  bias={(rf-y).mean():+.3f}")
        print(f"  surrogate vs obs : NSE={np.mean(s_nse):+.3f}+-{np.std(s_nse,ddof=1):.3f}")
        print(f"  surrogate vs ref : R2={np.mean(sr_r2):.4f}+-{np.std(sr_r2,ddof=1):.4f}  NRMSE={np.mean(sr_nr):.2f}+-{np.std(sr_nr,ddof=1):.2f}%")
        d = 100 * (np.mean(csur) - cref) / cref if cref else float("nan")
        print(f"  cumulative  obs={cobs:.2f}  ref={cref:.2f} ({100*(cref-cobs)/abs(cobs):+.1f}%)  surr vs ref {d:+.2f}%\n")


def summarize():
    reps_col = [f"Anet_pred_{rep}" for rep in REPS]
    rows = []
    for stem, yr in RECORDS:
        d = pd.read_csv(OUT / f"{stem}_external_validation.csv", parse_dates=["Date"]) \
              .sort_values("Date").reset_index(drop=True)
        obs, ref = d.Anet_obs.to_numpy(), d.Anet_c3bb_cal.to_numpy()
        c_obs, c_ref = cumulative_total(d.Date, obs), cumulative_total(d.Date, ref)
        sst = np.sum((ref - ref.mean()) ** 2)
        rng = ref.max() - ref.min()
        r2 = [1 - np.sum((d[c].to_numpy() - ref) ** 2) / sst for c in reps_col]
        nr = [100 * np.sqrt(np.mean((d[c].to_numpy() - ref) ** 2)) / rng for c in reps_col]
        cs = [100 * (cumulative_total(d.Date, d[c].to_numpy()) - c_ref) / abs(c_ref) for c in reps_col]
        rows.append(dict(
            yr=yr, n=len(d),
            nse=1 - np.mean((ref - obs) ** 2) / obs.var(),
            cum_ro=100 * (c_ref - c_obs) / abs(c_obs),
            r2=np.mean(r2), r2sd=np.std(r2, ddof=1),
            nr=np.mean(nr), nrsd=np.std(nr, ddof=1),
            cs=np.mean(cs), cssd=np.std(cs, ddof=1)))
    T = pd.DataFrame(rows)
    T.rename(columns=dict(yr="record_year", nse="nse_ref_vs_obs", cum_ro="cum_pct_ref_vs_obs",
                          r2="r2_surr_vs_ref_mean", r2sd="r2_surr_vs_ref_sd",
                          nr="nrmse_pct_surr_vs_ref_mean", nrsd="nrmse_pct_surr_vs_ref_sd",
                          cs="cum_pct_surr_vs_ref_mean", cssd="cum_pct_surr_vs_ref_sd")).round(4) \
     .to_csv(OUT / "pine_field_metrics.csv", index=False)
    print(T.round(4).to_string(index=False))
    print(f"\nNSE {T.nse.min():.2f}-{T.nse.max():.2f} | cum ref-obs {T.cum_ro.min():+.1f} to "
          f"{T.cum_ro.max():+.1f}% | R2 {T.r2.min():.4f}-{T.r2.max():.4f} | "
          f"NRMSE {T.nr.min():.2f}-{T.nr.max():.2f}% | cum surr-ref {T.cs.min():+.2f} to {T.cs.max():+.2f}%")
    print(f"\n[save] {OUT / 'pine_field_metrics.csv'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize()
    else:
        run_one(sys.argv[1] if len(sys.argv) > 1 else "pine")
