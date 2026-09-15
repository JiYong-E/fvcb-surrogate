"""Build a sequential model-forcing table from the AmeriFlux US-Ton half-hourly record, for the
temporal surrogate-vs-reference comparison of Sec. 2.6 / 3.3 (Figure 4c,d). Model-to-model only
(no leaf-level observations needed): reference and surrogate are both driven by this table alone.

Period: 2020-2022, full 24-h cycle. Physiology fixed at the training-design median (Table 2) --
there is nothing to calibrate against here, so the parameter set is just a fixed operating point.

Drivers are clamped into the sampled ranges (RH, PFD, wind); PFD gaps are filled from SW_IN_F via
the site's own regression. Clamp/gap counts are printed at run time as the provenance record.

Source: AmeriFlux FLUXNET-1F US-Ton (Tonzi Ranch) half-hourly release (CC-BY-4.0; see the
manuscript's Data Availability Statement for the required attribution). Not redistributed here --
point AMERIFLUX_DIR at your own copy of the FLUXNET_HH csv.

Output: output/uston_2020_2022_input.csv (+ .meta.json with fixed physiology and clamp counts)
Run: python 01_prepare_forcing.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _log(msg):
    print(msg, flush=True)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

# directory holding the AmeriFlux US-Ton FLUXNET-1F download (doi:10.17190/AMF/2204880)
AMERIFLUX_DIR = Path(os.environ.get("AMERIFLUX_DIR", HERE / "ameriflux"))
SRC = AMERIFLUX_DIR / "AMF_US-Ton_FLUXNET_FLUXMET_HH_2001-2025_v1.3_r1.csv"

YEARS = (2020, 2022)

# medians of the sampled design (Table 2), matching 05_field_validation's fixed-parameter convention
FIXED = dict(Vcm25=132.5, Jm25=240.0, Rd25=2.6, g0=0.03, g1=10.0, w=10.15)
# non-sampled FvCB temperature-response constants, as used by the field-validation inputs
CONST = dict(EaVc=67.5, Eaj=47.5, Hj=225.0, Sj=630.0, Ear=45.0,
             Gamma25=45.0, Tp25=16.0, EaTp=45.0)

# sampled design ranges (Table 2) that the forcing must stay inside
LIMITS = dict(T_air=(-15.0, 45.0), RH=(5.0, 99.0), PFD=(0.0, 2499.999),
              CO2=(250.001, 1499.999), wind=(0.1, 8.0))


def main():
    cols = ["TIMESTAMP_START", "TA_F", "RH", "PPFD_IN", "SW_IN_F", "CO2_F_MDS", "WS_F"]
    _log(f"[input] reading {SRC.name}")
    d = pd.read_csv(SRC, usecols=cols, low_memory=False).replace(-9999, np.nan)
    d["Date"] = pd.to_datetime(d["TIMESTAMP_START"], format="%Y%m%d%H%M")
    d = d[d["Date"].dt.year.between(*YEARS)].copy()
    _log(f"[input] {YEARS[0]}-{YEARS[1]}, full 24 h: {len(d):,} half-hourly rows")

    # PFD: measured where available, else from SW_IN_F via the site's own regression
    fit_rows = d.dropna(subset=["PPFD_IN", "SW_IN_F"])
    slope, icpt = np.polyfit(fit_rows["SW_IN_F"], fit_rows["PPFD_IN"], 1)
    r2 = np.corrcoef(fit_rows["SW_IN_F"], fit_rows["PPFD_IN"])[0, 1] ** 2
    n_gap = int(d["PPFD_IN"].isna().sum())
    d["PFD"] = d["PPFD_IN"].where(d["PPFD_IN"].notna(), slope * d["SW_IN_F"] + icpt)
    _log(f"[input] PFD: {n_gap:,} gaps ({n_gap/len(d)*100:.2f}%) filled from SW_IN_F "
         f"(slope {slope:.3f}, intercept {icpt:.2f}, R2 {r2:.4f})")

    d = d.rename(columns={"TA_F": "T_air", "CO2_F_MDS": "CO2", "WS_F": "wind"})
    d = d.dropna(subset=["T_air", "RH", "PFD", "CO2", "wind"])
    _log(f"[input] {len(d):,} rows with all five drivers present")

    clamps = {}
    for col, (lo, hi) in LIMITS.items():
        below = int((d[col] < lo).sum())
        above = int((d[col] > hi).sum())
        d[col] = d[col].clip(lo, hi)
        clamps[col] = dict(below=below, above=above,
                           pct=round((below + above) / len(d) * 100, 4))
        flag = "" if below + above == 0 else "  <-- clamped"
        _log(f"[input] {col:6s} to [{lo}, {hi}]: {below:,} below, {above:,} above "
             f"({clamps[col]['pct']}%){flag}")

    for k, v in {**FIXED, **CONST}.items():
        d[k] = v

    keep = (["Date", "T_air", "RH", "PFD", "CO2", "wind"]
            + list(FIXED.keys()) + list(CONST.keys()))
    out = d[keep].sort_values("Date").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "uston_2020_2022_input.csv"
    out.to_csv(p, index=False)
    _log(f"[save] {p}  ({len(out):,} rows)")

    meta = dict(source=SRC.name, years=list(YEARS), n_rows=int(len(out)),
                fixed_physiology=FIXED, constants=CONST,
                pfd_gapfill=dict(n_gaps=n_gap, slope=float(slope), intercept=float(icpt),
                                 r2=float(r2), source="SW_IN_F"),
                clamps=clamps, limits={k: list(v) for k, v in LIMITS.items()})
    pm = OUT_DIR / "uston_2020_2022_input.meta.json"
    pm.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"[save] {pm}")

    _log("\n[input] driver summary of the file actually written:")
    _log(out[["T_air", "RH", "PFD", "CO2", "wind"]].describe()
         .loc[["min", "50%", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
