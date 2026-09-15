#!/usr/bin/env julia
# Adds the three FvCB pre-solver features (Ac_init/Aj_init/Ap_init = manuscript's
# A_c,Input/A_j,Input/A_p,Input), pre-solver estimates (Anet_init, Ci_init, Tleaf_init), and
# VPD to a reference-simulation Parquet dataset. Pure function of existing columns (Ci0=Ca,
# Tleaf0=T_air), no re-simulation; constants/equations match LeafGasExchange.jl.
#
# Usage: julia --project=../01_design 02_fvcb_features.jl <dir-of-part-*.parquet>
using DataFrames, Parquet2

const R_GAS = 8.31446261815324
const TB_C = 25.0
const TB_K = TB_C + 273.15

const Kc25 = 404.9; const Eac = 79.43
const Ko25 = 278.4; const Eao = 36.38
const Om = 210.0
const Eag = 37.83   # activation energy for CO2 compensation point
const THETA = 0.7
const DELTA = 0.15
const F = 0.15

arrhenius_kT(Tc::Float64, Ea_kJ_mol::Float64) = begin
    Tk = Tc + 273.15
    exp((Ea_kJ_mol * 1000.0) * (Tc - TB_C) / (R_GAS * Tk * TB_K))
end

peaked_kT(Tc::Float64, Ea_kJ_mol::Float64, H_kJ_mol::Float64, S_J_mol_K::Float64) = begin
    Tk = Tc + 273.15
    kT = arrhenius_kT(Tc, Ea_kJ_mol)
    num = 1 + exp((S_J_mol_K * TB_K - H_kJ_mol * 1000.0) / (R_GAS * TB_K))
    den = 1 + exp((S_J_mol_K * Tk - H_kJ_mol * 1000.0) / (R_GAS * Tk))
    kT * num / den
end

# VaporPressure system constants (a, b, c) from LeafGasExchange.jl/src/vaporpressure.jl
const VP_A = 0.611; const VP_B = 17.502; const VP_C = 240.97
es_kpa(Tc::Float64) = VP_A * exp(VP_B * Tc / (VP_C + Tc))
vpd_kpa(Tc::Float64, RH_pct::Float64) = es_kpa(Tc) * (1.0 - RH_pct / 100.0)

function enrich!(df::DataFrame)
    n = nrow(df)
    Ac0 = fill(NaN, n); Aj0 = fill(NaN, n); Ap0 = fill(NaN, n); Anet0 = fill(NaN, n)
    Ci0 = fill(NaN, n); Tleaf0 = fill(NaN, n); VPD = fill(NaN, n)

    Threads.@threads for i in 1:n
        Tc = Float64(df.T_air[i])
        Ca = Float64(df.CO2[i])          # numerically == partial pressure in μbar, since P_air = 1 bar
        Vcmax = Float64(df.Vcm25[i]) * arrhenius_kT(Tc, Float64(df.EaVc[i]))
        Jmax  = Float64(df.Jm25[i]) * peaked_kT(Tc, Float64(df.Eaj[i]), Float64(df.Hj[i]), Float64(df.Sj[i]))
        Rd    = Float64(df.Rd25[i]) * arrhenius_kT(Tc, Float64(df.Ear[i]))
        Tp    = Float64(df.Tp25[i]) * arrhenius_kT(Tc, Float64(df.EaTp[i]))
        Γ     = Float64(df.Gamma25[i]) * arrhenius_kT(Tc, Eag)
        Kc    = Kc25 * arrhenius_kT(Tc, Eac)
        Ko    = Ko25 * arrhenius_kT(Tc, Eao)
        Km    = Kc * (1 + Om / Ko)

        I2 = Float64(df.PFD[i]) * (1 - DELTA) * (1 - F) / 2
        b = I2 + Jmax
        disc = max(0.0, b^2 - 4THETA * I2 * Jmax)
        J = (b - sqrt(disc)) / (2THETA)

        ac0 = Vcmax * (Ca - Γ) / (Ca + Km) - Rd
        aj0 = J * (Ca - Γ) / (4 * (Ca + 2Γ)) - Rd
        ap0 = 3Tp - Rd

        Ac0[i] = ac0; Aj0[i] = aj0; Ap0[i] = ap0
        Anet0[i] = min(ac0, aj0, ap0)
        Ci0[i] = Ca; Tleaf0[i] = Tc
        VPD[i] = vpd_kpa(Tc, Float64(df.RH[i]))
    end

    df[!, :Ac_init]   = Ac0
    df[!, :Aj_init]   = Aj0
    df[!, :Ap_init]   = Ap0
    df[!, :Anet_init] = Anet0
    df[!, :Ci_init]   = Ci0
    df[!, :Tleaf_init] = Tleaf0
    df[!, :VPD]       = VPD
    return df
end

function main()
    length(ARGS) >= 1 || error("Usage: 02_fvcb_features.jl <dir-of-part-*.parquet>")
    dir = ARGS[1]
    parts = sort(filter(f -> startswith(f, "part-") && endswith(f, ".parquet"), readdir(dir)))
    isempty(parts) && error("No part-*.parquet found in $dir")
    println("enriching $(length(parts)) parts in $dir (threads=$(Threads.nthreads()))")
    for (i, p) in enumerate(parts)
        path = joinpath(dir, p)
        df = DataFrame(Parquet2.Dataset(path); copycols=false)
        if "Ac_init" in names(df)
            println("  $i/$(length(parts)) already enriched, skip")
            continue
        end
        enrich!(df)
        tmp = path * ".tmp"
        Parquet2.writefile(tmp, df; compression_codec=:zstd)
        mv(tmp, path; force=true)
        (i % 40 == 0 || i == length(parts)) && println("  $i/$(length(parts)) done")
    end
    println("enrichment complete: $dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
