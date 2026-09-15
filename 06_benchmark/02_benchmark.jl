#=
Deployment-speed benchmark for M_aug: fused ONNX vs the reference solver (Table 6, Figure 6).
batch=100 uses 100 distinct rows (part-000001.parquet rows 1:100); reference_solver_timing.csv
was measured on the SAME rows, so both sides compare on identical inputs.

Protocol: batch_n=100 distinct rows (single-sample = row 1 alone), NamedTuple session call,
BenchmarkTools @belapsed, session released + GC.gc() after each model.

Run: julia --project=. 02_benchmark.jl (needs fused ONNX graphs from ../04_onnx/ under ONNX_DIR
and reference_solver_timing.csv, produced by 01_reference_timing.jl)
=#
using BenchmarkTools
using ONNXRunTime
using Parquet2
using DataFrames
using CSV
using Dates
using Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const ONNX_DIR  = get(ENV, "ONNX_DIR",  joinpath(REPO_ROOT, "onnx"))
const EVAL_PATH = get(ENV, "EVAL_PATH", joinpath(REPO_ROOT, "data", "final_test", "part-000001.parquet"))
const REF_CSV   = get(ENV, "REF_CSV",   joinpath(@__DIR__, "reference_solver_timing.csv"))
const REPS  = ["rep01", "rep02", "rep03", "rep04", "rep05"]   # missing reps are skipped
const SIZES = [1000, 10000, 100000, 1000000, 10000000]
const OUT_CSV = get(ENV, "OUT_CSV", joinpath(@__DIR__, "benchmark_results.csv"))

function load_features(onnx_path::String)
    feat_path = replace(onnx_path, ".onnx" => ".features.txt")
    strip.(split(strip(read(feat_path, String)), '\n'))
end

run1(s, x) = s((features = x,))

function main()
    println("[canonical] loading eval rows ..."); flush(stdout)
    ds = Parquet2.Dataset(EVAL_PATH)
    df_all = DataFrame(ds)
    println("[canonical] eval rows loaded: ", nrow(df_all)); flush(stdout)

    ref = CSV.read(REF_CSV, DataFrame)
    ref_batch1_ms   = ref[ref.condition .== "batch1",   :min_ms][1]
    ref_batch100_ms = ref[ref.condition .== "batch100", :min_ms][1]
    println("[canonical] C3BB reference (diverse 100-row, from 01_reference_timing.jl): ",
            "batch1=$(ref_batch1_ms)ms  batch100_total=$(ref_batch100_ms)ms  ",
            "per_sample=$(ref_batch100_ms/100)ms")
    println(); flush(stdout)

    rows = NamedTuple[]
    total = length(REPS) * length(SIZES)
    done = 0
    for rep in REPS, n in SIZES
        onnx_path = joinpath(ONNX_DIR, "Maug_$(rep)_n$(n)_fused.onnx")
        if !isfile(onnx_path)
            println("[canonical] SKIP $rep n=$n (missing $onnx_path)"); flush(stdout)
            done += 1
            continue
        end
        feats = load_features(onnx_path)
        Xall = Matrix{Float32}(Matrix(df_all[1:100, Symbol.(feats)]))
        X100 = Xall
        X1   = Xall[1:1, :]

        session = ONNXRunTime.load_inference(onnx_path)
        size_mb = filesize(onnx_path) / 1024^2

        run1(session, X1); run1(session, X100)  # warm-up

        t_single_ms = @belapsed(run1($session, $X1),   seconds=0.5) * 1000
        t_batch_ms  = @belapsed(run1($session, $X100), seconds=0.5) * 1000
        per_sample_ms = t_batch_ms / 100

        speedup_single = ref_batch1_ms / t_single_ms
        speedup_batch  = ref_batch100_ms / t_batch_ms

        @printf("[%d/%d] %s n=%-8d  single=%.4fms  batch100=%.4fms (%.5fms/sample)  size=%.1fMB  speedup: single=%.2fx batch=%.2fx\n",
                done+1, total, rep, n, t_single_ms, t_batch_ms, per_sample_ms, size_mb,
                speedup_single, speedup_batch)
        flush(stdout)

        push!(rows, (model="Maug", rep=rep, n_training_samples=n, model_size_mb=round(size_mb, digits=3),
                      single_ms=round(t_single_ms, digits=6), batch100_total_ms=round(t_batch_ms, digits=6),
                      batch100_ms_per_sample=round(per_sample_ms, digits=8),
                      c3bb_single_ms=ref_batch1_ms, c3bb_batch100_total_ms=ref_batch100_ms,
                      batch1_speedup=round(speedup_single, digits=4), batch100_speedup=round(speedup_batch, digits=4)))
        CSV.write(OUT_CSV, DataFrame(rows))

        try; ONNXRunTime.release(session); catch; end
        GC.gc()
        done += 1
    end

    println(); println("[save] $OUT_CSV")
    show(DataFrame(rows), allrows=true)
    println()
end

main()
