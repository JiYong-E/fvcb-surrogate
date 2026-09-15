using CSV, DataFrames
using Parquet2
using Statistics
using PythonCall
using Dates

const RANDOM_STATE = 42
const CATBOOST_ITERATIONS    = parse(Int, get(ENV, "CATBOOST_ITERATIONS",    "20000"))
const CATBOOST_EARLY_STOPPING = parse(Int, get(ENV, "CATBOOST_EARLY_STOPPING", "200"))
const CATBOOST_TASK_TYPE = uppercase(get(ENV, "CATBOOST_TASK_TYPE", Sys.iswindows() ? "GPU" : "CPU"))
const CATBOOST_DEVICES   = get(ENV, "CATBOOST_DEVICES", "0")
# CatBoost's own periodic progress line (train/eval loss every N iterations); 0 disables it.
const CATBOOST_VERBOSE   = parse(Int, get(ENV, "CATBOOST_VERBOSE", "50"))
# Caps the number of rows loaded from a `part-*.parquet` directory (reads only as many
# parts as needed instead of the whole 10M-row set) for a fast subset run; unset/0 = all rows.
const STAGE1_MAX_ROWS = parse(Int, get(ENV, "STAGE1_MAX_ROWS", "0"))

# Flushed println -- otherwise buffered progress doesn't show until exit when redirected.
function logmsg(args...)
    println(args...)
    flush(stdout)
end

const JOINT_TARGETS = ["Tleaf_sim", "Anet_sim", "gs_sim", "Ci_sim"]

const STAGE1_FEATURES = [
    # init
    "Aj_init", "Ac_init", "Ap_init",
    # environmental
    "T_air", "RH", "wind", "PFD", "CO2", "w", "VPD",
    # physiological
    "Vcm25", "Jm25", "g0", "g1", "Rd25",
]

const CSV_RENAME = Dict(
    "Tair_C"         => "T_air",
    "RH_pct"         => "RH",
    "U_ms"           => "wind",
    "PPFD_umol_m2_s" => "PFD",
    "Ca_umol_mol"    => "CO2",
    "Vcmax25"        => "Vcm25",
    "Jmax25"         => "Jm25",
    "Aj0_init"       => "Aj_init",
    "Ac0_init"       => "Ac_init",
    "Ap0_init"       => "Ap_init",
    "Anet_full"      => "Anet_sim",
    "Ci_full"        => "Ci_sim",
    "gs_full"        => "gs_sim",
    "gvc_full"       => "gvc_sim",
    "VPD_sim"        => "VPD",
)

const np = pyimport("numpy")
const cb = pyimport("catboost")

rmse(y, ŷ) = sqrt(mean((y .- ŷ).^2))
mae(y, ŷ)  = mean(abs.(y .- ŷ))
nrmse(y, ŷ) = rmse(y, ŷ) / max(maximum(y) - minimum(y), 1e-12)
function r2(y, ŷ)
    ss_tot = sum((y .- mean(y)).^2)
    ss_tot == 0.0 ? 0.0 : 1.0 - sum((y .- ŷ).^2) / ss_tot
end

# Reads a Parquet file, a dir of part-*.parquet files, or a CSV, dispatched on `path`.
function load_dataframe(path::String; max_rows::Int=STAGE1_MAX_ROWS)
    df = if isdir(path)
        parts = sort(filter(f -> startswith(f, "part-") && endswith(f, ".parquet"), readdir(path)))
        isempty(parts) && error("No part-*.parquet files found in directory: $path")
        frames = DataFrame[]
        rows_so_far = 0
        for p in parts
            d = DataFrame(Parquet2.Dataset(joinpath(path, p)); copycols=false)
            push!(frames, d)
            rows_so_far += nrow(d)
            max_rows > 0 && rows_so_far >= max_rows && break
        end
        out = vcat(frames...)
        max_rows > 0 ? out[1:min(max_rows, nrow(out)), :] : out
    elseif endswith(lowercase(path), ".parquet")
        DataFrame(Parquet2.Dataset(path); copycols=false)
    else
        CSV.read(path, DataFrame)
    end
    for (old, new) in CSV_RENAME
        old in names(df) && rename!(df, old => new)
    end
    df
end

function to_float(v)
    (!ismissing(v) && v isa Number && isfinite(Float64(v))) ? Float64(v) : NaN
end

function fill_matrix(df::DataFrame, cols::Vector{String}, fill_vals::Union{Vector{Float64},Nothing}=nothing)
    X = Matrix{Float64}(undef, nrow(df), length(cols))
    computed = Float64[]
    for (j, c) in enumerate(cols)
        raw = to_float.(df[!, c])
        fv = isnothing(fill_vals) ? begin
            valid = filter(isfinite, raw)
            isempty(valid) ? 0.0 : mean(valid)
        end : fill_vals[j]
        push!(computed, fv)
        X[:, j] = [isnan(v) ? fv : v for v in raw]
    end
    X, computed
end


function build_model(iters::Int)
    try
        CATBOOST_TASK_TYPE == "GPU" && return cb.CatBoostRegressor(
            iterations=iters, depth=8, learning_rate=0.05, loss_function="RMSE",
            od_type="Iter", od_wait=CATBOOST_EARLY_STOPPING, random_seed=RANDOM_STATE,
            verbose=CATBOOST_VERBOSE, allow_writing_files=false,
            task_type="GPU", devices=CATBOOST_DEVICES,
            boosting_type="Plain",   # Ordered->Plain avoids GPU sync overhead, ~2-3x faster
            border_count=255,        # max split search on GPU (default is 128); this is the
                                      # value actually used for training and reported in the
                                      # manuscript (task type defaults to GPU on Windows)
        )
    catch
    end
    cb.CatBoostRegressor(
        iterations=iters, depth=8, learning_rate=0.05, loss_function="RMSE",
        od_type="Iter", od_wait=CATBOOST_EARLY_STOPPING, random_seed=RANDOM_STATE,
        verbose=CATBOOST_VERBOSE, allow_writing_files=false,
        boosting_type="Plain",
        border_count=254,            # CatBoost recommends <255 on CPU; unused fallback path,
                                      # not the value used for the reported results
        thread_count=-1,             # use all CPU cores
    )
end
