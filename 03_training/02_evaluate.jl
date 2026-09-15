# Re-scores every trained (replicate, N) model against the fixed final test design (seed
# 1225, n=1,000,000, Ball-Berry). Pure predict(), no retraining/GPU; skips missing models.
#
# ENV overrides (for scoring straight from the Zenodo deposit -- see 04_onnx/fuse_core.py's
# _resolve_run for the matching model-layout autodetection):
#   FVCB_TEST_PATH    = final_test parquet dir (default: ../data/final_test)
#   FVCB_MODELS_ROOT  = models root, e.g. the Zenodo deposit's models/ (default: ../out/Maug)
#   FVCB_REPS         = comma-separated replicate list, e.g. "rep01" for Zenodo-only
#   FVCB_OUT_DIR      = where to write the report CSVs (default: ../out/reports); deliberately
#                       NOT inside FVCB_MODELS_ROOT, so scoring straight from an unzipped Zenodo
#                       deposit never writes into that deposit's own directory tree

include(joinpath(@__DIR__, "catboost_core.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const FINAL_TEST_PATH = get(ENV, "FVCB_TEST_PATH", joinpath(REPO_ROOT, "data", "final_test"))
const PRODUCTION_ROOT = get(ENV, "FVCB_MODELS_ROOT", joinpath(REPO_ROOT, "out", "Maug"))
const OUT_DIR = get(ENV, "FVCB_OUT_DIR", joinpath(REPO_ROOT, "out", "reports"))
const REPS = haskey(ENV, "FVCB_REPS") ? String.(split(ENV["FVCB_REPS"], ",")) :
             ["rep01", "rep02", "rep03", "rep04", "rep05"]
const SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
# Zenodo's deposited models/rep01/{1k,10k,100k,1m,10m}/ (no "models" subfolder, no
# stage2_feature_map.csv alongside -- it's always the M_aug feature set) uses these shorthand
# size labels instead of "n<N>".
const SIZE_LABELS = Dict(1_000=>"1k", 10_000=>"10k", 100_000=>"100k",
                          1_000_000=>"1m", 10_000_000=>"10m")

# Returns (models_dir, x_cols) or (nothing, nothing) if no run is found for (rep, n).
function _resolve_run(production_root::String, rep::String, n::Int)
    own_dir = joinpath(production_root, rep, "n$n")
    fm_path = joinpath(own_dir, "stage2_feature_map.csv")
    if isfile(fm_path)
        return joinpath(own_dir, "models"), String.(CSV.read(fm_path, DataFrame).feature)
    end
    label = get(SIZE_LABELS, n, nothing)
    if label !== nothing
        zdir = joinpath(production_root, rep, label)
        isdir(zdir) && return zdir, STAGE1_FEATURES
    end
    return nothing, nothing
end

function run_final_report(; test_path::String=FINAL_TEST_PATH,
        production_root::String=PRODUCTION_ROOT,
        sizes::Vector{Int}=SIZES,
        reps::Vector{String}=REPS,
        out_csv::String=joinpath(OUT_DIR, "final_report_seed1225.csv"))
    mkpath(dirname(out_csv))
    logmsg("="^70)
    logmsg("Final report re-scoring against fixed test (seed=1225, n=1,000,000)")
    logmsg("test = $(abspath(test_path))")
    logmsg("root = $(abspath(production_root))")
    logmsg("reps = $(join(reps, ", "))")
    logmsg("="^70)

    df_test = load_dataframe(test_path; max_rows=0)
    logmsg("[load] test n=$(nrow(df_test))")

    rows = NamedTuple[]
    skipped = String[]
    for rep in reps, n in sizes
        models_dir, x_cols = _resolve_run(production_root, rep, n)
        if models_dir === nothing
            # own layout: feature_map is written only after all targets finish, so its absence
            # means in-progress; Zenodo layout: this (rep, n) just isn't in the deposit.
            for target in JOINT_TARGETS
                push!(skipped, "$rep/n$n/$target")
            end
            continue
        end
        X_test, _ = fill_matrix(df_test, x_cols)
        for target in JOINT_TARGETS
            model_path = joinpath(models_dir, "stage2_$(target)_model.cbm")
            if !isfile(model_path)
                push!(skipped, "$rep/n$n/$target")
                continue
            end
            y_test = Float64.(df_test[!, target])

            model = cb.CatBoostRegressor()
            model.load_model(model_path)
            pred = Vector{Float64}(pyconvert(Array, model.predict(np.array(X_test))))

            push!(rows, (rep=rep, n=n, target=target, n_test=nrow(df_test),
                R2=r2(y_test, pred), RMSE=rmse(y_test, pred),
                NRMSE=nrmse(y_test, pred), NRMSE_pct=nrmse(y_test, pred) * 100.0,
                MAE=mae(y_test, pred)))
        end
        logmsg("[final-report] $rep/n$n scored ($(count(r -> r.rep == rep && r.n == n, rows))/$(length(JOINT_TARGETS)) targets)")
    end

    if !isempty(skipped)
        logmsg("\n[skip] $(length(skipped)) combo/target(s) had no trained model yet (not an error -- rerun this script after they finish):")
        for s in skipped
            logmsg("  - $s")
        end
    end

    isempty(rows) && error("No trained models found under $production_root -- nothing to score")
    out_df = DataFrame(rows)
    sort!(out_df, [:target, :n, :rep])
    CSV.write(out_csv, out_df)
    logmsg("\n[save] $out_csv  ($(nrow(out_df)) rows)")

    summary = combine(groupby(out_df, [:target, :n]),
        :R2 => mean => :R2_mean, :R2 => std => :R2_std,
        :NRMSE_pct => mean => :NRMSE_pct_mean, :NRMSE_pct => std => :NRMSE_pct_std,
        nrow => :n_reps)
    sort!(summary, [:target, :n])
    summary_csv = joinpath(dirname(out_csv), replace(basename(out_csv), r"\.csv$" => "") * "_summary.csv")
    CSV.write(summary_csv, summary)
    logmsg("[save] $summary_csv")
    return out_df, summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_final_report()
end
