#=
Reference-solver side of the deployment-speed benchmark (Table 6, Figure 6): times
LeafGasExchange.jl ModelC3BB on the SAME 100 rows 02_benchmark.jl uses for the fused-ONNX
side (EVAL_PATH, rows 1:100), so both sides are compared on identical inputs. Writes
reference_solver_timing.csv.

Run: julia --project=. -t auto 01_reference_timing.jl
(this directory's Project.toml/Manifest.toml pin LeafGasExchange.jl and Cropbox at the same
commits as ../02_features/, plus BenchmarkTools/ONNXRunTime for 02_benchmark.jl)
=#
using BenchmarkTools
using Parquet2
using DataFrames
using CSV
using Cropbox
using LeafGasExchange
using Unitful
using Logging

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const EVAL_PATH = get(ENV, "EVAL_PATH", joinpath(REPO_ROOT, "data", "final_test", "part-000001.parquet"))
const OUT_CSV   = joinpath(@__DIR__, "reference_solver_timing.csv")

struct QuietLogger <: AbstractLogger end
Logging.min_enabled_level(::QuietLogger) = Logging.Error
Logging.shouldlog(::QuietLogger, level, _module, group, id) = false
Logging.catch_exceptions(::QuietLogger) = false
Logging.handle_message(::QuietLogger, args...; kwargs...) = nothing

getnum(x) = x isa Unitful.Quantity ? Unitful.ustrip(x) : Float64(x)

# Same field mapping as 02_features/01_simulate_reference.jl's config_from_row.
config_from_row(row) = (
    :Weather => (T_air=row.T_air, RH=row.RH, PFD=row.PFD, CO2=row.CO2, wind=max(0.1, Float64(row.wind))),
    :BoundaryLayer => (w=row.w,),
    :C3c => (Vcm25=row.Vcm25, EaVc=row.EaVc),
    :C3j => (Jm25=row.Jm25, Eaj=row.Eaj, Hj=row.Hj, Sj=row.Sj),
    :C3r => (Rd25=row.Rd25, Ear=row.Ear, Γ25=row.Gamma25),
    :C3p => (Tp25=row.Tp25, EaTp=row.EaTp),
    :StomataBallBerry => (g0=row.g0, g1=row.g1),
    :Controller => (),
)

function solve_batch(df::DataFrame)
    for i in 1:nrow(df)
        with_logger(QuietLogger()) do
            simulate(LeafGasExchange.ModelC3BB; config=config_from_row(df[i, :]))
        end
    end
end

function main()
    df = DataFrame(Parquet2.Dataset(EVAL_PATH))[1:100, :]
    println("[reference-timing] loaded ", nrow(df), " rows from ", EVAL_PATH)

    t1_s   = @belapsed solve_batch($(df[1:1, :]))
    t100_s = @belapsed solve_batch($df)
    t1_ms, t100_ms = t1_s * 1000, t100_s * 1000
    println("[reference-timing] batch1=$(round(t1_ms, digits=3))ms  " *
            "batch100=$(round(t100_ms, digits=1))ms  per_sample=$(round(t100_ms/100, digits=3))ms")

    CSV.write(OUT_CSV, DataFrame(
        condition=["batch1", "batch100"],
        min_ms=[t1_ms, t100_ms],
        ms_per_sample=[t1_ms, t100_ms / 100],
    ))
    println("[save] $OUT_CSV")
end

main()
