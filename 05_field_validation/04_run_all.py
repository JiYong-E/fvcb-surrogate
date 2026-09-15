"""Scots pine temporal field evaluation, all five records (manuscript Figure 5, Table 5).

usage: python 04_run_all.py [SMEARII_DIR]

SMEARII_DIR (also read from $SMEARII_DIR; default ./smearii) must hold the Aalto (2023)
FFlux files: FFlux2011.180  FFlux2013.187  FFlux2015.409  FFlux2017.421  FFlux2019.429
(Zenodo doi:10.5281/zenodo.10360968, not redistributed with this repo).

For each record: prepare the forcing table, fit V_cmax25 and J_max25 with the reference
solver (R_d25 fixed at the pooled night value), run the 5-replicate surrogate; then summarize
all five records into output/pine_field_metrics.csv.

The reference-solver fit needs a Julia environment with LeafGasExchange.jl, Cropbox and Optim
added -- point JULIA_PROJECT at it (default: this directory).
"""
import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMEARII_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMEARII_DIR", str(HERE / "smearii"))
JULIA = os.environ.get("JULIA", "julia")
JULIA_PROJECT = os.environ.get("JULIA_PROJECT", ".")
RD25 = "1.78"   # pooled nighttime least-squares R_d25 (n = 3,125 night points), fixed for every record

FFLUX = {"pine2011": "FFlux2011.180", "pine2013": "FFlux2013.187", "pine2015": "FFlux2015.409",
         "pine2017": "FFlux2017.421", "pine": "FFlux2019.429"}

# pine2013 has half the rows of the other records (n=1510) and 81% of its daytime points below
# PAR 400, so the whole-record objective barely constrains J_max25: an unconstrained fit runs to
# J_max25/V_cmax25 = 4.42, well outside the 1.72-2.30 range every other record identifies on its
# own. For this record only, J_max25 is tied to 2.0x V_cmax25 (only V_cmax25 is fitted) so the
# reported capacities stay physiologically consistent with the other four records.
FIX_JV = {"pine2013": "2.0"}


def main():
    for tag, ff in FFLUX.items():
        print(f"=== {tag} ===", flush=True)
        subprocess.run([sys.executable, "01_prepare_pine.py", ff, tag], cwd=HERE, check=True,
                        env={**os.environ, "SMEARII_DIR": SMEARII_DIR})
        fit_env = {**os.environ, "FIELD_INPUT": f"output/{tag}_input_litdefault.csv", "FIX_RD25": RD25}
        fit_env.pop("FIX_JV", None)  # never inherit a pre-set FIX_JV from the calling shell
        if tag in FIX_JV:
            fit_env["FIX_JV"] = FIX_JV[tag]
        subprocess.run([JULIA, f"--project={JULIA_PROJECT}", "02_fit_reference.jl", "fitall"],
                        cwd=HERE, check=True, env=fit_env)
        subprocess.run([sys.executable, "03_run_surrogate.py", tag], cwd=HERE, check=True)

    subprocess.run([sys.executable, "03_run_surrogate.py", "--summarize"], cwd=HERE, check=True)
    print("done -> output/pine_field_metrics.csv")


if __name__ == "__main__":
    main()
