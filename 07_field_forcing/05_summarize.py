"""Summarize the US-Ton field-forcing run (02/03 output) into the cumulative and seasonal
numbers reported in Results Sec. 3.3 / Figure 4d.

usage: python 05_summarize.py
reads   output/uston_2020_2022_{reference,surrogate}.csv
writes  output/uston_cumulative.csv   per-target totals + running-ratio checkpoints (q1/mid/q3/end)
        output/uston_seasonal.csv     per-target in-window Jan-Jun/Jul-Dec ratio + per-replicate
                                       final difference

Note: "in-window" (seasonal.csv) and "running" (cumulative.csv) ratios use different denominators
(that window's own total vs. the elapsed total) and are not interchangeable. Replicates are
averaged before the difference is taken, not after -- averaging per-replicate ratios instead
gives a different, smaller number.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"
REPS = None   # discovered from the prediction table's own columns (see main)
GAP_CAP_S = 3600.0  # a single half-hourly reading stands for at most one hour of a gap


def gapdt(dates):
    dt = pd.Series(dates).diff().dt.total_seconds().to_numpy().copy()
    nominal = np.nanmedian(dt[1:]) if len(dt) > 1 else 1800.0
    dt[0] = nominal
    return np.where(np.isnan(dt) | (dt <= 0) | (dt > GAP_CAP_S), nominal, dt)


def main():
    d = pd.read_csv(OUT / "uston_2020_2022_surrogate.csv", parse_dates=["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    dt = gapdt(d["Date"])
    month = d["Date"].dt.month.to_numpy()
    # the deposit carries replicate 1 only, so take whatever replicates 03 actually predicted
    global REPS
    REPS = sorted(c.split("_")[-1] for c in d.columns if c.startswith("Anet_rep"))
    if not REPS:
        raise SystemExit("no Anet_rep* columns; run 03_run_surrogate.py first")
    print(f"[summarize] {len(d)} timestamps, {d['Date'].min()} .. {d['Date'].max()}, "
          f"replicates: {', '.join(REPS)}", flush=True)

    # target -> (reference column, unit-conversion factor to mol/kmol m-2)
    targets = {"Anet": ("Anet_ref", 1e-6), "gs": ("gs_ref", 1e-3)}

    cumulative_rows = []
    seasonal_rows = []
    for target, (ref_col, scale) in targets.items():
        ref_step = d[ref_col].to_numpy(float) * dt * scale
        rep_steps = np.array([d[f"{target}_{r}"].to_numpy(float) * dt * scale for r in REPS])
        mean_step = rep_steps.mean(axis=0)
        diff_step = mean_step - ref_step

        cum_ref = np.cumsum(ref_step)
        cum_diff = np.cumsum(diff_step)
        ref_total, diff_total = cum_ref[-1], cum_diff[-1]

        checkpoints = {}
        for label, frac in (("q1", 0.25), ("mid", 0.50), ("q3", 0.75), ("end", 1.00)):
            i = min(int(len(d) * frac), len(d) - 1)
            checkpoints[f"ratio_{label}_pct"] = 100 * cum_diff[i] / cum_ref[i]

        cumulative_rows.append(dict(
            target=target, ref_total=ref_total, surrogate_total=ref_total + diff_total,
            final_diff=diff_total, final_pct=100 * diff_total / ref_total, **checkpoints,
        ))
        print(f"[summarize] {target}: total {ref_total:.2f} -> {ref_total + diff_total:.2f} "
              f"({100 * diff_total / ref_total:+.3f}%)", flush=True)

        for label, sel in (("Jan-Jun", (month >= 1) & (month <= 6)),
                            ("Jul-Dec", (month >= 7) & (month <= 12))):
            in_window_pct = 100 * diff_step[sel].sum() / ref_step[sel].sum()
            seasonal_rows.append(dict(target=target, window=label, in_window_pct=in_window_pct))
            print(f"[summarize]   {label} (pooled 2020-2022): {in_window_pct:+.2f}%", flush=True)

        for r in REPS:
            rep_step = d[f"{target}_{r}"].to_numpy(float) * dt * scale
            rep_diff_total = np.cumsum(rep_step - ref_step)[-1]
            seasonal_rows.append(dict(
                target=target, window=f"replicate_{r}_final",
                in_window_pct=100 * rep_diff_total / ref_total,
            ))

    pd.DataFrame(cumulative_rows).round(6).to_csv(OUT / "uston_cumulative.csv", index=False)
    pd.DataFrame(seasonal_rows).round(4).to_csv(OUT / "uston_seasonal.csv", index=False)
    print(f"[save] {OUT / 'uston_cumulative.csv'}", flush=True)
    print(f"[save] {OUT / 'uston_seasonal.csv'}", flush=True)


if __name__ == "__main__":
    main()
