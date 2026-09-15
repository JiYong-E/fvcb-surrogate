#!/usr/bin/env julia
#
# Sequential reference-solver run over the US-Ton half-hourly forcing (01_prepare_forcing.py),
# for the temporal surrogate-vs-reference comparison of Sec. 2.6 / 3.3 (Figure 4c,d).
#
# Reuses ../05_field_validation/02_fit_reference.jl's config layout without modifying that file.
# Differs from it in two ways: nothing is fitted (physiology is fixed by the input table, since
# there are no observations to calibrate against here), and g_s/C_i/T_leaf are returned alongside
# A_net (that script only keeps A_net).
#
# Usage: julia --project=. -t auto 02_run_reference.jl [N_ROWS]
#   N_ROWS, if given, truncates the run -- use it to time a slice before running all rows.
#
# In:  output/uston_2020_2022_input.csv   (from 01_prepare_forcing.py)
# Out: output/uston_2020_2022_reference.csv

using CSV, DataFrames, Statistics, Printf, Unitful, Dates
using Cropbox, LeafGasExchange
using Logging

const OUT    = joinpath(@__DIR__, "output")
const IN_CSV = get(ENV, "FORCING_INPUT", joinpath(OUT, "uston_2020_2022_input.csv"))

getnum(x) = x isa Unitful.Quantity ? Unitful.ustrip(x) : x

struct QuietLogger <: AbstractLogger end
Logging.min_enabled_level(::QuietLogger) = Logging.Error
Logging.shouldlog(::QuietLogger, level, _module, group, id) = false
Logging.catch_exceptions(::QuietLogger) = false
Logging.handle_message(::QuietLogger, args...; kwargs...) = nothing

# identical grouping to 05_field_validation/02_fit_reference.jl's make_cfg, with every value
# taken from the row (nothing fitted here)
make_cfg(r) = (
    :Weather => (T_air=r.T_air, RH=r.RH, PFD=r.PFD, CO2=r.CO2, wind=max(0.1, r.wind)),
    :BoundaryLayer => (w=r.w,),
    :C3c => (Vcm25=r.Vcm25, EaVc=r.EaVc),
    :C3j => (Jm25=r.Jm25, Eaj=r.Eaj, Hj=r.Hj, Sj=r.Sj),
    :C3r => (Rd25=r.Rd25, Ear=r.Ear, Γ25=r.Gamma25),
    :C3p => (Tp25=r.Tp25, EaTp=r.EaTp),
    :StomataBallBerry => (g0=r.g0, g1=r.g1),
    :Controller => (),
)

function main()
    df = CSV.read(IN_CSV, DataFrame)
    truncated = length(ARGS) >= 1
    outcsv = joinpath(OUT, truncated ? "uston_2020_2022_reference_test$(ARGS[1]).csv"
                                      : "uston_2020_2022_reference.csv")
    if truncated
        n = min(parse(Int, ARGS[1]), nrow(df))
        df = df[1:n, :]
        @printf("[ref] TRUNCATED to %d rows for timing -- writing to %s, not the full-run file\n",
                n, outcsv)
    end
    n = nrow(df)
    @printf("[ref] %d rows, %d threads\n", n, Threads.nthreads())
    flush(stdout)

    anet  = fill(NaN, n)
    gs    = fill(NaN, n)
    ci    = fill(NaN, n)
    tleaf = fill(NaN, n)
    done  = Threads.Atomic{Int}(0)
    t0    = time()

    Threads.@threads for i in 1:n
        r = df[i, :]
        try
            sim = with_logger(QuietLogger()) do
                simulate(LeafGasExchange.ModelC3BB; config=make_cfg(r))
            end
            anet[i]  = getnum(only(sim[!, :A_net]))
            gs[i]    = getnum(only(sim[!, :gs]))
            ci[i]    = getnum(only(sim[!, :Ci]))
            tleaf[i] = getnum(only(sim[!, :T]))
        catch
        end
        k = Threads.atomic_add!(done, 1) + 1
        if k % 2000 == 0
            el = time() - t0
            @printf("[ref] %6d / %6d  (%.1f%%)  %.1f s elapsed, ~%.1f s remaining\n",
                    k, n, 100k/n, el, el * (n - k) / k)
            flush(stdout)
        end
    end

    df.Anet_ref  = anet
    df.gs_ref    = gs
    df.Ci_ref    = ci
    df.Tleaf_ref = tleaf

    nfail = count(isnan, anet)
    @printf("[ref] done in %.1f s; %d rows failed to converge (%.3f%%)\n",
            time() - t0, nfail, 100nfail/n)
    CSV.write(outcsv, df)
    @printf("[save] %s\n", outcsv)
    flush(stdout)
end

main()
