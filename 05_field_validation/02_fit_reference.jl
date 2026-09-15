#!/usr/bin/env julia
#
# Reference-solver calibration for a field gas-exchange record (manuscript Sections 2.7 / 3.4).
#
# Given one prepared forcing table (FIELD_INPUT, from 01_prepare_pine.py), fit the FvCB
# capacities against the observed A_net with LeafGasExchange.jl ModelC3BB, then re-run the
# solver at the fitted values. R_d25 is held fixed (FIX_RD25); V_cmax25 and J_max25 are fitted
# per record. See 04_run_all.py for how the five Scots pine records are driven through this.
#
# Usage: FIELD_INPUT=output/<tag>_input_litdefault.csv FIX_RD25=1.78 \
#        julia --project=<env-with-LeafGasExchange.jl> -t auto 02_fit_reference.jl fitall

using CSV, DataFrames, Statistics, Printf, Unitful, Dates
using Cropbox, LeafGasExchange
using Optim
using Logging

const OUT    = joinpath(@__DIR__, "output")
# FIELD_INPUT points at one prepared forcing table, e.g. output/pine_input_litdefault.csv
# from 01_prepare_pine.py.
const IN_CSV = get(ENV, "FIELD_INPUT", joinpath(OUT, "pine_input_litdefault.csv"))

getnum(x) = x isa Unitful.Quantity ? Unitful.ustrip(x) : x

struct QuietLogger <: AbstractLogger end
Logging.min_enabled_level(::QuietLogger) = Logging.Error
Logging.shouldlog(::QuietLogger, level, _module, group, id) = false
Logging.catch_exceptions(::QuietLogger) = false
Logging.handle_message(::QuietLogger, args...; kwargs...) = nothing

make_cfg(r, vcm25, jm25, rd25; g0=r.g0, g1=r.g1, w=r.w, wind=r.wind, co2=r.CO2) = (
    :Weather => (T_air=r.T_air, RH=r.RH, PFD=r.PFD, CO2=co2, wind=max(0.1, wind)),
    :BoundaryLayer => (w=w,),
    :C3c => (Vcm25=vcm25, EaVc=r.EaVc),
    :C3j => (Jm25=jm25, Eaj=r.Eaj, Hj=r.Hj, Sj=r.Sj),
    :C3r => (Rd25=rd25, Ear=r.Ear, Γ25=r.Gamma25),
    :C3p => (Tp25=r.Tp25, EaTp=r.EaTp),
    :StomataBallBerry => (g0=g0, g1=g1),
    :Controller => (),
)

function run_c3bb(df::DataFrame, vcm25, jm25, rd25; g0=nothing, g1=nothing, w=nothing)
    n = nrow(df)
    anet = fill(NaN, n)
    Threads.@threads for i in 1:n
        r = df[i, :]
        try
            cfg = make_cfg(r, vcm25, jm25, rd25;
                           g0 = g0 === nothing ? r.g0 : g0,
                           g1 = g1 === nothing ? r.g1 : g1,
                           w  = w  === nothing ? r.w  : w)
            sim = with_logger(QuietLogger()) do
                simulate(LeafGasExchange.ModelC3BB; config=cfg)
            end
            anet[i] = getnum(only(sim[!, :A_net]))
        catch
        end
    end
    anet
end

rmse(a, b) = sqrt(mean((a .- b) .^ 2))

function objective(df, target; label="", tie_jv=nothing)
    nev = Ref(0)
    function obj(p)
        tie_jv === nothing || (p = [p[1], tie_jv * p[1], p[3]])
        (p[1] <= 5 || p[2] <= 5 || p[3] <= 0) && return 1e6
        a = run_c3bb(df, p[1], p[2], p[3])
        ok = .!isnan.(a)
        v = rmse(a[ok], target[ok])
        nev[] += 1
        @printf("    %s eval %3d: Vcm25=%9.5f Jm25=%10.5f Rd25=%8.5f  RMSE=%.8e\n",
                label, nev[], p[1], p[2], p[3], v)
        flush(stdout)
        v
    end
end

# Nelder-Mead fit of (Vcmax25, Jmax25, Rd25); `free` selects which indices are optimized
# (rest fixed at x0), restarted until stable. Fitting all three is weakly identified
# (mostly Rubisco-limited field data) -- prefer free=[1] or free=[1,3].
function optim_fit(df, target, x0::Vector{Float64}; free=[1, 2, 3], label="fit", restarts=4,
                   tie_jv=nothing)
    full = copy(x0)
    embed(q) = (z = copy(full); z[free] .= q;
                tie_jv === nothing || (z[2] = tie_jv * z[1]); z)
    p = copy(x0[free]); fmin = Inf
    for k in 1:restarts
        obj3 = objective(df, target; label="$label$k", tie_jv=tie_jv)
        res = optimize(q -> obj3(embed(q)), p, NelderMead(),
                       Optim.Options(x_abstol=1e-5, f_abstol=1e-10, iterations=1500))
        pnew, fnew = Optim.minimizer(res), Optim.minimum(res)
        full = embed(pnew)
        @printf("  [%s restart %d] Vcm25=%.5f Jm25=%.5f Rd25=%.5f  RMSE=%.8e  (converged=%s)\n",
                label, k, full..., fnew, Optim.converged(res))
        moved = maximum(abs.(pnew .- p) ./ max.(abs.(p), 1e-6))
        p, fmin = pnew, fnew
        moved < 1e-4 && (println("  [$label] restart optimum stable, stopping"); break)
    end
    embed(p), fmin
end

"Profile RMSE along one parameter with the others held at `p0`, to show identifiability."
function profile_param(df, target, p0, idx, grid)
    println("  profile of parameter $idx around the optimum:")
    for g in grid
        q = copy(p0); q[idx] = g
        a = run_c3bb(df, q...); ok = .!isnan.(a)
        @printf("    p[%d]=%9.3f  RMSE=%.6f\n", idx, g, rmse(a[ok], target[ok]))
    end
end

function main()
    mode = isempty(ARGS) ? "fitall" : ARGS[1]
    mode == "fitall" || error("unknown mode: $mode (only fitall is supported)")
    df = CSV.read(IN_CSV, DataFrame)
    @info "loaded $(nrow(df)) field-record rows | threads=$(Threads.nthreads())"
    do_fitall(df)
end

# Whole-record calibration (the version the manuscript uses): fits (Vcmax25, Jmax25) once
# against the entire record with Rd25 fixed from the night-derived FIX_RD25 value (see below;
# Rd25 stays free only if FIX_RD25 is unset), no cal/val split -- the surrogate-vs-reference
# comparison doesn't depend on where the parameters came from, since both models get the same
# set. Writes <stem>_c3bb_calibrated.csv, period="all".
function do_fitall(df)
    dts = df.Date isa AbstractVector{<:AbstractString} ?
          DateTime.(first.(split.(string.(df.Date), '.')), dateformat"yyyy-mm-dd HH:MM:SS") :
          DateTime.(df.Date)
    df = df[sortperm(dts), :]
    @printf("\n=== FITALL  n=%d  %s .. %s ===\n", nrow(df),
            string(minimum(dts)), string(maximum(dts)))

    # R_d25 is identifiable on its own from the night observations (PFD = 0 gives A_net = -R_d),
    # whereas letting the whole-record fit choose it puts R_d25 on a ridge with V_cmax25: the
    # objective is dominated by the daytime points, and daytime A_net = A_gross - R_d, so the
    # two trade off. Supplying the night-derived value via FIX_RD25 breaks that ridge and leaves
    # only (V_cmax25, J_max25) free.
    rd_fixed = haskey(ENV, "FIX_RD25") ? parse(Float64, ENV["FIX_RD25"]) : nothing
    # for a record whose A_net trajectory never saturates electron transport, the whole-record
    # J_max25 profile is flat and the free fit runs to a non-physiological J_max25/V_cmax25. If
    # FIX_JV is set, J_max25 is tied to that ratio times V_cmax25 and only V_cmax25 is fitted.
    jv = haskey(ENV, "FIX_JV") ? parse(Float64, ENV["FIX_JV"]) : nothing
    x0 = [df.Vcm25[1], jv === nothing ? df.Jm25[1] : jv * df.Vcm25[1],
          rd_fixed === nothing ? df.Rd25[1] : rd_fixed]
    free = jv !== nothing ? [1] : (rd_fixed === nothing ? [1, 2, 3] : [1, 2])
    rd_fixed === nothing || @printf("  R_d25 fixed at %.4f\n", rd_fixed)
    jv === nothing || @printf("  J_max25 tied to %.2f x V_cmax25; fitting V_cmax25 only\n", jv)
    p, fmin = optim_fit(df, df.Anet_obs, x0; free=free, label="all", tie_jv=jv)
    @printf("\n  FITTED on the whole record: Vcmax25=%.4f  Jmax25=%.4f  Rd25=%.4f  (RMSE %.4f)\n",
            p..., fmin)
    @printf("  ratios: Jmax25/Vcmax25=%.2f   Rd25/Vcmax25=%.4f\n", p[2] / p[1], p[3] / p[1])

    # identifiability of J_max25 at the optimum: a flat profile means the record does not
    # constrain it and the fitted value should not be read as a physiological estimate
    profile_param(df, df.Anet_obs, p, 2, [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6, 2.2] .* p[2])

    r2(a, b) = 1 - sum((a .- b) .^ 2) / sum((b .- mean(b)) .^ 2)
    full = run_c3bb(df, p...)
    ok = .!isnan.(full)
    @printf("  reference(calibrated) vs observed: NSE=%.3f  bias=%+.3f  n=%d\n",
            r2(full[ok], df.Anet_obs[ok]), mean(full[ok] .- df.Anet_obs[ok]), count(ok))

    out = DataFrame(Date=df.Date, Anet_obs=df.Anet_obs,
                    period=fill("all", nrow(df)), Anet_c3bb_cal=full)
    out.Vcm25 .= p[1]; out.Jm25 .= p[2]; out.Rd25 .= p[3]; out.n_free .= length(free)
    stem = replace(basename(IN_CSV), "_input_litdefault.csv" => "")
    path = joinpath(OUT, "$(stem)_c3bb_calibrated.csv")
    CSV.write(path, out)
    println("  [save] $path")
end

main()
