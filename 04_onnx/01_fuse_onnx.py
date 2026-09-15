"""Build the fused four-output ONNX graph for the M_aug surrogate at all five training sizes.

Each fused graph merges the four per-target CatBoost models (Anet, Ci, gs, Tleaf) trained by
../03_training/ into one graph with a single 15-feature input. The deployment-speed benchmark
in ../06_benchmark/ runs against these.

Input:  a training-output tree  <MODELS_ROOT>/rep01/n<N>/{models/*.cbm, stage2_metrics.csv}
        (MODELS_ROOT defaults to ../03_training/out/Maug; set FVCB_MODELS_ROOT to override).
        fuse_core._resolve_run() also auto-detects the Zenodo CBM deposit layout directly
        (models/rep01/<size-label>/*.cbm, best_iter read from the deposit's
        results/per_model_metrics.csv) -- no copying or retraining needed for that layout.
Output: ../onnx/Maug_rep01_n<N>_fused.onnx  (+ .features.txt sidecar). No external .onnx.data
        file is produced: onnx.save_model's externalization only applies to tensors above a
        size threshold, and CatBoost's fused TreeEnsembleRegressor graph has none large enough
        to qualify -- all five real graphs confirm this (checked against the deployed ONNX
        files, no .data alongside any of them).

    python 01_fuse_onnx.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuse_core

HERE = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.environ.get("FVCB_MODELS_ROOT", HERE.parent / "03_training" / "out" / "Maug"))
OUT_DIR = Path(os.environ.get("FVCB_ONNX_DIR", HERE.parent / "onnx"))
REP = "rep01"
SIZES = [1000, 10000, 100000, 1000000, 10000000]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n in SIZES:
        out_path = OUT_DIR / f"Maug_{REP}_n{n}_fused.onnx"
        print(f"\n=== Maug n={n} -> {out_path} ===", flush=True)
        t1 = time.time()
        try:
            fuse_core.build_fused(REP, n, MODELS_ROOT, out_path)
            mb = out_path.stat().st_size / 1e6
            print(f"    OK  {mb:.1f} MB  {time.time() - t1:.0f}s", flush=True)
        except FileNotFoundError as e:
            print(f"    SKIP ({e})", flush=True)
    print(f"\nall sizes done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
