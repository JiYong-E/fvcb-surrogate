module IndependentEvalLHS

using DataFrames
using Parquet2
using StableRNGs
using SHA

include(joinpath(@__DIR__, "01_nested_lhs.jl"))
using .NestedC3LHS: PARAMETER_NAMES, RANGES, PHYSICAL_DIMENSIONS, BALLBERRY_G,
    _initial_lhs, _table, _lhs_ok, _range_ok, _no_duplicate_rows, write_parquet, read_parquet

export VALIDATION_SEED, VALIDATION_SIZE, TEST_SEED, TEST_SIZE_N,
       generate_independent_lhs, qc_independent, generate_eval_designs

# Standalone eval seeds, distinct from all training seeds (301, 717, 815, 1003, 1009).
const VALIDATION_SEED = 424242
const VALIDATION_SIZE = 100_000
const TEST_SEED = 1225
const TEST_SIZE_N = 1_000_000

# Single-layer LHS (not nested), n rows, seed-reproducible.
function generate_independent_lhs(seed::Int, n::Int; d::Int=length(PARAMETER_NAMES))
    rng = StableRNG(seed)
    return _initial_lhs(rng, n, d)
end

# QC: stratification, range, finiteness, no duplicate rows, reproducibility.
function qc_independent(x::Matrix{Float64}, seed::Int, n::Int)
    row_count_ok = size(x, 1) == n
    lhs_ok = _lhs_ok(x)
    finite_ok = all(isfinite, x)
    range_ok = _range_ok(x, :ballberry)
    duplicate_ok = _no_duplicate_rows(x)
    reproducible_ok = x == generate_independent_lhs(seed, n)
    ok = row_count_ok && lhs_ok && finite_ok && range_ok && duplicate_ok && reproducible_ok
    return (ok=ok, row_count=row_count_ok, lhs=lhs_ok, finite=finite_ok,
        range=range_ok, no_duplicate_rows=duplicate_ok, reproducible=reproducible_ok)
end

# Builds validation + test Ball-Berry designs, QC's them, writes Parquet + manifest.
# Never reads a training design file -- independence is structural, not seed-numeric only.
function generate_eval_designs(; outdir::String=joinpath(@__DIR__, "..", "designs", "eval"))
    mkpath(outdir)
    manifest = NamedTuple[]
    for (kind, seed, n) in (("validation", VALIDATION_SEED, VALIDATION_SIZE),
                             ("test", TEST_SEED, TEST_SIZE_N))
        x = generate_independent_lhs(seed, n)
        qc = qc_independent(x, seed, n)
        qc.ok || error("QC failed for $kind design (seed=$seed, n=$n): $qc")
        path = joinpath(outdir, "$(kind)_n$(n)_seed$(seed)_ballberry.parquet")
        write_parquet(path, x, :ballberry)
        push!(manifest, (kind=kind, seed=seed, rows=n, columns=length(PARAMETER_NAMES),
            output_path=relpath(path, @__DIR__), qc_ok=qc.ok, qc_lhs=qc.lhs, qc_range=qc.range,
            qc_finite=qc.finite, qc_no_duplicate_rows=qc.no_duplicate_rows,
            qc_reproducible=qc.reproducible))
        println("[eval-lhs] $kind: n=$n seed=$seed qc_ok=$(qc.ok) -> $path")
    end
    # Verify validation/test share zero rows.
    xv = generate_independent_lhs(VALIDATION_SEED, VALIDATION_SIZE)
    xt = generate_independent_lhs(TEST_SEED, TEST_SIZE_N)
    shared = length(intersect(Set(eachrow(xv)), Set(eachrow(xt))))
    shared == 0 || error("validation and test designs share $shared row(s) -- seed collision?")
    println("[eval-lhs] validation/test independence verified: 0 shared rows")

    manifest_path = joinpath(outdir, "manifest.parquet")
    Parquet2.writefile(manifest_path, DataFrame(manifest); compression_codec=:zstd)
    println("[eval-lhs] manifest: $manifest_path")
    return DataFrame(manifest)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .IndependentEvalLHS
    m = generate_eval_designs()
    show(m, allrows=true)
end
