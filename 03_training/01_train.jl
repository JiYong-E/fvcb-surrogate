# Production trainer: fits on 100% of the given training N (no internal split). Early
# stopping and final metrics come from two fixed, independent LHS designs shared across
# every (replicate, N) combination:
#   - validation (100k, seed 424242): CatBoost eval_set / best_iteration only, never scored.
#   - test       (1M,  seed 1225):   final R2/RMSE/NRMSE/MAE reporting only, never seen by fit().
#
# Usage: julia --project=. 01_train.jl <Maug|Mbase|Mfull> [DATA_DIR] [OUT_DIR] [REPS...]
#   trains all five sizes (1e3..1e7) for each replicate in REPS (default just "01"). DATA_DIR
#   (default ./data) must hold the Zenodo layout for rep01: rep01_training_10m/  validation/
#   final_test/; replicates 2-5 need their own designs regenerated first (see ../01_design/),
#   laid out as rep<NN>/n10000000_ballberry under DATA_DIR.
#   CATBOOST_ITERATIONS, CATBOOST_EARLY_STOPPING, CATBOOST_TASK_TYPE, CATBOOST_VERBOSE (env vars)

include(joinpath(@__DIR__, "catboost_core.jl"))
# Reuses load_dataframe/fill_matrix/build_model/logmsg/rmse/mae/r2/nrmse and the
# CATBOOST_* / STAGE1_FEATURES / JOINT_TARGETS constants defined there.

const REPO_ROOT          = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_VALIDATION = joinpath(REPO_ROOT, "data", "validation")
const DEFAULT_TEST       = joinpath(REPO_ROOT, "data", "final_test")

# Fits on 100% of df_train; df_val is eval_set only (never scored). Final scoring is
# against df_test in run_production.
function fit_full_with_external_validation(df_train::DataFrame, df_val::DataFrame,
        target::String, cols::Vector{String})
    y_train = Float64.(df_train[!, target])
    y_val   = Float64.(df_val[!, target])

    X_tr, fill_vals = fill_matrix(df_train, cols)
    X_val, _         = fill_matrix(df_val, cols, fill_vals)

    iters = CATBOOST_ITERATIONS
    X_fit  = np.array(X_tr)
    X_eval = np.array(X_val)
    y_fit  = np.array(y_train)
    y_eval = np.array(y_val)

    model = build_model(iters)
    try
        model.fit(X_fit, y_fit; eval_set=(X_eval, y_eval))
    catch err
        if CATBOOST_TASK_TYPE == "GPU"
            @warn "GPU fit failed, retry CPU" exception=(err, catch_backtrace())
            model = cb.CatBoostRegressor(
                iterations=iters, depth=8, learning_rate=0.05, loss_function="RMSE",
                od_type="Iter", od_wait=CATBOOST_EARLY_STOPPING, random_seed=RANDOM_STATE,
                verbose=CATBOOST_VERBOSE, allow_writing_files=false,
                boosting_type="Plain", border_count=254, thread_count=-1,
            )
            model.fit(X_fit, y_fit; eval_set=(X_eval, y_eval))
        else
            rethrow()
        end
    end

    best_iter = something(tryparse(Int, string(model.best_iteration_)), -1)
    pct  = round(100 * best_iter / iters; digits=1)
    flag = best_iter >= Int(round(iters * 0.95)) ? " ⚠️  HIT LIMIT (raise CATBOOST_ITERATIONS)" : ""

    # CatBoost records the full per-iteration validation-metric history internally.
    evals = model.get_evals_result()
    val_key = "validation" in pyconvert(Vector{String}, evals.keys()) ? "validation" : "validation_0"
    metric_name = first(pyconvert(Vector{String}, evals[val_key].keys()))
    val_trajectory = Vector{Float64}(pyconvert(Array, evals[val_key][metric_name]))
    stopped_iteration = length(val_trajectory)  # actual trees fit before early stop, <= iters

    logmsg("    n_train=$(nrow(df_train)) (100% used)  n_validation=$(nrow(df_val))  " *
        "best_iteration=$best_iter / $iters ($pct%)$flag  stopped_iteration=$stopped_iteration  " *
        "best_val_$metric_name=$(round(val_trajectory[best_iter+1], digits=6))")

    return model, fill_vals, best_iter, iters, stopped_iteration, metric_name, val_trajectory
end

# Downsamples a trajectory to every `step` iterations (always keeps the final point).
function downsample_trajectory(traj::Vector{Float64}, step::Int=1000)
    n = length(traj)
    idx = collect(1:step:n)
    (isempty(idx) || idx[end] != n) && push!(idx, n)
    return DataFrame(iteration = idx .- 1, value = traj[idx])
end

function run_production(; train_path::String=get(ENV, "STAGE2_TRAIN", ""),
        val_path::String=get(ENV, "STAGE2_VALIDATION", DEFAULT_VALIDATION),
        test_path::String=get(ENV, "STAGE2_TEST", DEFAULT_TEST),
        out_root::String=get(ENV, "STAGE2_OUT", ""),
        train_max_rows::Int=parse(Int, get(ENV, "STAGE2_TRAIN_MAX_ROWS", "0")),
        features::Vector{String}=STAGE1_FEATURES)
    isempty(train_path) && error("STAGE2_TRAIN is required")
    isempty(out_root) && error("STAGE2_OUT is required")
    t0 = time()

    logmsg("="^70)
    logmsg("CatBoost Revision Production (TRAIN != VALIDATION != TEST)")
    logmsg("train      = $(abspath(train_path))")
    logmsg("validation = $(abspath(val_path))  (early stopping only, never scored)")
    logmsg("test       = $(abspath(test_path))  (final metrics only, never fit)")
    logmsg("output     = $(abspath(out_root))")
    logmsg("train_max_rows = $(train_max_rows == 0 ? "unlimited (whole train_path)" : train_max_rows)")
    logmsg("iterations cap = $(CATBOOST_ITERATIONS), early_stopping wait = $(CATBOOST_EARLY_STOPPING)")
    logmsg("features ($(length(features))): $(join(features, ", "))")
    logmsg("targets: $(join(JOINT_TARGETS, ", "))")
    logmsg("start = $(Dates.now())")
    logmsg("="^70)

    df_train = load_dataframe(train_path; max_rows=train_max_rows)
    df_val   = load_dataframe(val_path; max_rows=0)
    df_test  = load_dataframe(test_path; max_rows=0)
    if train_max_rows > 0
        nrow(df_train) == train_max_rows ||
            error("train_max_rows=$train_max_rows requested but got $(nrow(df_train)) rows -- train_path has too few rows")
    end
    logmsg("[load] train n=$(nrow(df_train))  validation n=$(nrow(df_val))  test n=$(nrow(df_test))")

    for t in JOINT_TARGETS
        for (name, df) in (("train", df_train), ("validation", df_val), ("test", df_test))
            t in names(df) || error("Missing target column '$t' in $name set")
        end
    end
    x_cols = [f for f in features if f in names(df_train)]
    missing_feats = [f for f in features if !(f in x_cols)]
    isempty(missing_feats) || @warn "Features not in training data (skipped): $(join(missing_feats, ", "))"
    isempty(x_cols) && error("No usable feature columns")

    mkpath(out_root)
    metrics_rows = Dict[]
    model_dir = joinpath(out_root, "models")
    mkpath(model_dir)

    for target in JOINT_TARGETS
        logmsg("\n[stage2] target=$target")
        model, fill_vals, best_iter, used_iters, stopped_iteration, metric_name, val_trajectory =
            fit_full_with_external_validation(df_train, df_val, target, x_cols)

        X_test, _ = fill_matrix(df_test, x_cols, fill_vals)
        pred = Vector{Float64}(pyconvert(Array, model.predict(np.array(X_test))))
        y_test = Float64.(df_test[!, target])

        push!(metrics_rows, Dict(
            "target"     => target,
            "n_train"    => nrow(df_train),
            "n_validation" => nrow(df_val),
            "n_test"     => nrow(df_test),
            "n_features" => length(x_cols),
            "best_iter"  => best_iter,
            "max_iter"   => used_iters,
            "stopped_iteration" => stopped_iteration,
            "hit_limit"  => best_iter >= Int(round(used_iters * 0.95)),
            "best_val_metric_name" => metric_name,
            "best_val_metric"      => val_trajectory[best_iter+1],
            "R2"         => r2(y_test, pred),
            "RMSE"       => rmse(y_test, pred),
            "NRMSE"      => nrmse(y_test, pred),
            "NRMSE_pct"  => nrmse(y_test, pred) * 100.0,
            "MAE"        => mae(y_test, pred),
        ))
        logmsg("[stage2] target=$target  R2=$(round(r2(y_test, pred), digits=4))  (scored on external fixed test set, n=$(nrow(df_test)))")

        traj_df = downsample_trajectory(val_trajectory, 1000)
        traj_df.target .= target
        CSV.write(joinpath(out_root, "trajectory_$(target).csv"), traj_df)

        safe = replace(target, r"[^A-Za-z0-9_\-]" => "_")
        model.save_model(joinpath(model_dir, "stage2_$(safe)_model.cbm"))
    end

    metrics_df = DataFrame(metrics_rows)
    sort!(metrics_df, :target)
    CSV.write(joinpath(out_root, "stage2_metrics.csv"), metrics_df)
    CSV.write(joinpath(out_root, "stage2_feature_map.csv"),
        DataFrame(feature_index=1:length(x_cols), feature=x_cols))

    logmsg("\n[save] $(out_root)")
    logmsg("[done] elapsed=$(round(time() - t0, digits=1))s")
    return metrics_df
end

const SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
const CONFIG_FEATURES = Dict(
    "Maug"  => STAGE1_FEATURES,
    "Mbase" => ["T_air","RH","wind","PFD","CO2","w","VPD","Vcm25","Jm25","g0","g1","Rd25"],
    "Mfull" => ["T_air","RH","wind","PFD","CO2","w","VPD","Vcm25","Jm25","Rd25","Tp25","Gamma25",
                "EaVc","Eaj","Ear","EaTp","Hj","Sj","g0","g1"],
)

# Trains one config (Maug|Mbase|Mfull) at all five sizes, for each replicate in `reps`; skips
# a size whose stage2_metrics.csv already exists. DATA_DIR must hold the Zenodo layout for
# rep01 (rep01_training_10m/, validation/, final_test/); replicates 2-5 need their designs
# regenerated with ../01_design/ first, laid out as rep<NN>/n10000000_ballberry under DATA_DIR.
function run_all(config::String; data_dir::String="./data", out_dir::String=joinpath("out", config),
        reps::Vector{String}=["01"])
    haskey(CONFIG_FEATURES, config) || error("unknown config: $config (expected Maug | Mbase | Mfull)")
    features = CONFIG_FEATURES[config]
    for rep in reps
        train_dir = rep == "01" ? joinpath(data_dir, "rep01_training_10m") :
                                   joinpath(data_dir, "rep$rep", "n10000000_ballberry")
        for n in SIZES
            out = joinpath(out_dir, "rep$rep", "n$n")
            if isfile(joinpath(out, "stage2_metrics.csv"))
                logmsg("skip $config rep$rep n=$n (already done)")
                continue
            end
            logmsg("=== $config rep$rep n=$n  $(Dates.now()) ===")
            run_production(train_path=train_dir, val_path=joinpath(data_dir, "validation"),
                test_path=joinpath(data_dir, "final_test"), out_root=out, train_max_rows=n,
                features=features)
        end
    end
    logmsg("=== $config done  $(Dates.now()) ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: julia --project=. 01_train.jl <Maug|Mbase|Mfull> [DATA_DIR] [OUT_DIR] [REPS...]")
    data_dir = length(ARGS) >= 2 ? ARGS[2] : "./data"
    out_dir  = length(ARGS) >= 3 ? ARGS[3] : joinpath("out", ARGS[1])
    reps     = length(ARGS) >= 4 ? ARGS[4:end] : ["01"]
    run_all(ARGS[1]; data_dir=data_dir, out_dir=out_dir, reps=reps)
end
