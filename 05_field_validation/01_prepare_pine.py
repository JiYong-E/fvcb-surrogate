"""Build a Scots pine (SMEAR-II Hyytiälä automated shoot chamber) forcing table for the
LeafGasExchange.jl C3 Ball--Berry reference solver and the surrogate.

Source: Juho Aalto (2023) FFlux shoot-chamber archive (Zenodo 10360968), e.g. FFlux2019.429
(chamber 429, a 3-digit = Scots pine shoot chamber; 2-digit = soil chamber). Columns are
tab-separated: yyyy mm dd HH MM SS Tcuv Tamb PAR PARtower CO2 H2O RHcuv Pamb F_CO2 F_H2O.
-999 = bad/missing.

This archive already provides:
  * RHcuv  -- chamber RH computed from the measured H2O concentration (no VPD back-calc)
  * CO2    -- measured chamber CO2 in ppm (2019 file; older files are gCO2 m-3), per timestamp
so Ca is not assumed at 400 and RH is not reconstructed.

Two conversions applied to the flux:
  * F_CO2 is reported per ALL-SIDED (total) needle surface area in ug CO2 s-1 m-2, efflux
    negative. LeafGasExchange.jl and the surrogate use a projected-area FvCB parameterisation,
    and for Scots pine total = projected * pi (Lin et al. 2002; ~Kolari et al. 2014 "/3").
    Anet_projected [umol m-2 s-1] = F_CO2 / 44.01 * PI_NEEDLE.
  * sign: F_CO2 positive = net uptake already matches the A_net convention.

Non-Dreyer physiology columns follow a "midpoint / generic" convention; Vcm25 and Jm25 are
starting guesses; 04_run_all.py calibrates them per record.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FF = sys.argv[1] if len(sys.argv) > 1 else "FFlux2019.429"
TAG = sys.argv[2] if len(sys.argv) > 2 else "pine"
MONTHS = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (6, 7)
# directory holding the Aalto (2023) SMEAR II FFlux files (Zenodo doi:10.5281/zenodo.10360968)
SMEARII_DIR = Path(os.environ.get("SMEARII_DIR", HERE / "smearii"))
SRC = SMEARII_DIR / FF
DST = OUT / f"{TAG}_input_litdefault.csv"

PI_NEEDLE = np.pi          # all-sided -> projected needle-area factor for Scots pine
MM_CO2 = 44.01             # ug CO2 per umol
BAD = -999

# starting / fixed physiology (projected-area basis). Vcm25 and Jm25 starting values are
# refit downstream (02_fit_reference.jl); for pine2013 only, Jm25 is tied to 2x the fitted
# Vcm25 instead of being fitted independently (see FIX_JV in 04_run_all.py).
COLS_FIXED = dict(
    wind=1.5,     # chamber airflow, not ambient wind
    w=0.10,       # Scots pine needle characteristic width ~1 mm
    Vcm25=50.0, EaVc=67.5,
    Jm25=90.0, Eaj=47.5, Hj=225.0, Sj=630.0,
    Rd25=1.0, Ear=45.0,
    Gamma25=45.0,
    Tp25=16.0, EaTp=45.0,
    g0=0.03, g1=10.0,
)


def main():
    raw = pd.read_csv(SRC, sep="\t")
    raw.columns = [c.strip() for c in raw.columns]
    # some year files carry Excel error strings ("#ARVO!", "#VALUE!") and extra instrument
    # columns; force every column we use to numeric and drop the rest
    for c in ("yyyy", "mm", "dd", "HH", "MM", "SS", "Tcuv", "Tamb", "PAR",
              "CO2", "H2O", "RHcuv", "Pamb", "F_CO2", "F_H2O"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.replace(BAD, np.nan)
    raw["Date"] = pd.to_datetime(dict(year=raw.yyyy, month=raw.mm, day=raw.dd,
                                      hour=raw.HH, minute=raw.MM, second=raw.SS))
    m = raw["Date"].dt.month.isin(MONTHS)
    sub = raw[m].dropna(subset=["Date", "Tcuv", "PAR", "CO2", "RHcuv", "F_CO2", "Pamb"]).sort_values("Date").reset_index(drop=True)
    print(f"[pine] {len(sub)} rows in months {MONTHS} with Tcuv,PAR,CO2,RHcuv,F_CO2,Pamb valid "
          f"({sub['Date'].min()} .. {sub['Date'].max()})")

    # CO2 is gCO2/m3 in older FFlux files, ppm in recent ones. Convert mass density -> mixing
    # ratio with the ideal gas law at the chamber T and P when the column is clearly not ppm.
    co2 = sub["CO2"].to_numpy(float)
    if np.nanmedian(co2) < 50.0:
        R = 8.314462618
        Tk = sub["Tcuv"].to_numpy(float) + 273.15
        Pa = sub["Pamb"].to_numpy(float) * 100.0        # hPa -> Pa
        co2 = co2 / MM_CO2 * R * Tk / Pa * 1e6
        print(f"[pine] CO2 converted gCO2 m-3 -> ppm  (median now {np.nanmedian(co2):.0f})")

    anet = sub["F_CO2"].to_numpy(float) / MM_CO2 * PI_NEEDLE
    out = pd.DataFrame({
        "Date": sub["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Anet_obs": anet,
        "T_air": sub["Tcuv"].to_numpy(float),
        "RH": sub["RHcuv"].to_numpy(float).clip(0, 100),
        "PFD": sub["PAR"].to_numpy(float),
        "CO2": co2,
    })
    for k, v in COLS_FIXED.items():
        out[k] = v
    order = ["Date", "Anet_obs", "T_air", "RH", "PFD", "CO2", "wind", "w",
             "Vcm25", "EaVc", "Jm25", "Eaj", "Hj", "Sj", "Rd25", "Ear",
             "Gamma25", "Tp25", "EaTp", "g0", "g1"]
    out = out[order]

    print(f"[pine] Anet_obs (projected): min {anet.min():.2f}  med {np.median(anet):.2f}  max {anet.max():.2f}  umol m-2 s-1")
    print(f"[pine] T_air {out.T_air.min():.1f}..{out.T_air.max():.1f}   RH {out.RH.min():.1f}..{out.RH.max():.1f}   "
          f"CO2 {out.CO2.min():.0f}..{out.CO2.max():.0f}  PFD max {out.PFD.max():.0f}")
    out.to_csv(DST, index=False)
    print(f"[save] {DST}   ({len(out)} rows)")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
