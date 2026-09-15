"""FvCB pre-solver features and shared kinetics for the field-validation surrogate step.

compute_init() returns the three pre-solver rates (Ac_init, Aj_init, Ap_init -- the
manuscript's A_c,Input / A_j,Input / A_p,Input) from T_air, RH, PFD and CO2, with the same
Arrhenius / peaked kinetic constants as 02_features/02_fvcb_features.jl. 03_run_surrogate.py sets
VCM25 / JM25 / RD25 to the record's fitted values before calling compute_init(); the other
temperature-response parameters use the fixed values below.
"""
import numpy as np

FEATURES = ["Aj_init", "Ac_init", "Ap_init", "T_air", "RH", "wind", "PFD", "CO2", "w", "VPD",
            "Vcm25", "Jm25", "g0", "g1", "Rd25"]

# VCM25/JM25/RD25 are overridden per record by 03_run_surrogate.py (set to the fitted values);
# the remaining temperature-response parameters use midpoint-of-range defaults.
VCM25, JM25, RD25 = 70.5, 125.3, 2.16
TP25, GAMMA25 = 16.0, 45.0  # matches 01_prepare_pine.py's COLS_FIXED Tp25 for the reference solver
EAVC, EAJ, EAR, EATP = 67.5, 47.5, 45.0, 45.0
HJ, SJ = 225.0, 630.0

# init-feature kinetic constants (Bernacchi 2001), identical to
# 02_features/02_fvcb_features.jl
R_ = 8.31446261815324
TB_ = 298.15
KC25, EAC = 404.9, 79.43
KO25, EAO = 278.4, 36.38
OM = 210.0
EAG = 37.83
DELTA, F_, THETA = 0.15, 0.15, 0.7
VP_A, VP_B, VP_C = 0.611, 17.502, 240.97


def arrh(T, Ea):
    return np.exp(Ea * 1e3 * (T - 25.0) / (R_ * (T + 273.15) * TB_))


def peaked(T, Ea, H, S):
    Tk = T + 273.15
    k = arrh(T, Ea)
    return k * (1 + np.exp((S * TB_ - H * 1e3) / (R_ * TB_))) / (1 + np.exp((S * Tk - H * 1e3) / (R_ * Tk)))


def compute_init(T_air, RH, PFD, CO2):
    Ci0 = CO2
    Vcmax = VCM25 * arrh(T_air, EAVC)
    Jmax = JM25 * peaked(T_air, EAJ, HJ, SJ)
    Rd = RD25 * arrh(T_air, EAR)
    Tp = TP25 * arrh(T_air, EATP)
    Gam = GAMMA25 * arrh(T_air, EAG)
    Kc = KC25 * arrh(T_air, EAC)
    Ko = KO25 * arrh(T_air, EAO)
    Km = Kc * (1 + OM / Ko)

    I2 = PFD * (1 - DELTA) * (1 - F_) / 2
    b = I2 + Jmax
    disc = np.maximum(0.0, b**2 - 4 * THETA * I2 * Jmax)
    J = (b - np.sqrt(disc)) / (2 * THETA)

    Ac_init = Vcmax * (Ci0 - Gam) / (Ci0 + Km) - Rd
    Aj_init = J * (Ci0 - Gam) / (4 * (Ci0 + 2 * Gam)) - Rd
    Ap_init = 3 * Tp - Rd
    return Ac_init, Aj_init, Ap_init


def es_sat(T_air):
    return VP_A * np.exp(VP_B * T_air / (VP_C + T_air))

