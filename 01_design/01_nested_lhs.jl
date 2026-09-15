module NestedC3LHS

using DataFrames
using Parquet2
using Random
using SHA
using StableRNGs

export TRAIN_SIZES, PARAMETER_NAMES, RANGES, generate_training_designs, qc_family,
       normalized_nested_lhs, write_parquet, read_parquet

"""Five requested training layers.  Every adjacent ratio is exactly 10."""
const TRAIN_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
"""Fixed independent family seeds for Revision Dataset v1."""
const REVISION_FAMILY_SEEDS = [301, 717, 815, 1003, 1009]
# Matches LeafGasExchange.jl's Cropbox field names 1:1 (no rename needed); Gamma25 is the
# ASCII stand-in for Cropbox's Unicode Γ25, mapped by the runner.
const PARAMETER_NAMES = (:T_air, :RH, :wind, :PFD, :CO2,
    :w, :Vcm25, :Jm25, :Rd25, :Tp25, :Gamma25, :EaVc,
    :Eaj, :Ear, :EaTp, :Hj, :Sj, :g0, :g1)
const PHYSICAL_DIMENSIONS = 17

# PFD lower bound 0.0 (night-time), RH range [5,99]: both verified finite/convergent over
# 10,000-20,000+ corner/random samples, no solver failures, no RH->1 singularity.
const RANGES = ((-15.0,45.0), (5.0,99.0), (0.1,8.0), (0.0,2500.0),
    (250.0,1500.0), (0.3,20.0), (15.0,250.0), (30.0,450.0), (0.2,5.0),
    (2.0,30.0), (35.0,55.0), (40.0,95.0), (20.0,75.0), (20.0,70.0),
    (20.0,70.0), (150.0,300.0), (540.0,720.0), (0.0,1.0), (0.0,1.0))
const MEDLYN_G = ((0.0,0.01), (2.0,6.0))
const BALLBERRY_G = ((0.01,0.05), (8.0,12.0))
const METHOD = "recursive multi-layer nested Latin hypercube; follows the nesting principle of Qian (2009); 10x refinement"

"""A zero-copy affine view, so a 10M output does not require a second 1.52-GB matrix."""
struct ScaledColumn{V<:AbstractVector{Float64}} <: AbstractVector{Float64}
    u::V
    lo::Float64
    width::Float64
end
Base.IndexStyle(::Type{<:ScaledColumn}) = IndexLinear()
Base.size(x::ScaledColumn) = size(x.u)
Base.length(x::ScaledColumn) = length(x.u)
Base.getindex(x::ScaledColumn, i::Int) = x.lo + x.width * x.u[i]

"""One random LHS: exactly one point in every one-dimensional stratum."""
function _initial_lhs(rng::AbstractRNG, n::Int, d::Int)
    x = Matrix{Float64}(undef, n, d)
    for j in 1:d
        p = randperm(rng, n)
        @inbounds for i in 1:n
            # `rand` is in [0,1), hence this is unconditionally in [0,1).
            x[i,j] = (p[i] - 1 + rand(rng)) / n
        end
    end
    return x
end

# Refines N to 10N points, retaining all old points exactly; both layers stay valid LHS
# (Qian nesting principle; a recursive construction, not Qian's exact algorithm).
function _refine10(rng::AbstractRNG, old::Matrix{Float64})
    n, d = size(old); m = 10n
    x = Matrix{Float64}(undef, m, d)
    x[1:n, :] .= old
    for j in 1:d
        # old is LHS: each coarse stratum has precisely one retained row.
        owner = zeros(Int, n)
        @inbounds for i in 1:n
            coarse = min(n - 1, floor(Int, old[i,j] * n)) + 1
            owner[coarse] = i
        end
        all(!=(0), owner) || error("Internal invariant failed: parent layer is not LHS")
        free_rows = randperm(rng, 9n) .+ n
        k = 1
        for coarse in 0:n-1
            retained = owner[coarse + 1]
            occupied = min(m - 1, floor(Int, old[retained,j] * m))
            firstfine = 10coarse
            for fine in firstfine:(firstfine + 9)
                fine == occupied && continue
                row = free_rows[k]; k += 1
                x[row,j] = (fine + rand(rng)) / m
            end
        end
        k == 9n + 1 || error("Internal invariant failed: refinement allocation")
    end
    return x
end

"""Generate a multi-layer nested family. Layers are prefixes, but not arbitrary prefixes: each is an LHS."""
function normalized_nested_lhs(seed::Int; sizes::Vector{Int}=TRAIN_SIZES, d::Int=length(PARAMETER_NAMES))
    length(sizes) >= 1 || error("sizes must be nonempty")
    issorted(sizes) || error("sizes must be increasing")
    rng = StableRNG(seed)
    layers = Matrix{Float64}[]
    x = _initial_lhs(rng, sizes[1], d)
    push!(layers, copy(x))
    for target in sizes[2:end]
        target == 10size(x,1) || error("This validated construction requires each adjacent size ratio to be 10")
        x = _refine10(rng, x)
        push!(layers, copy(x))
    end
    return layers
end

function _table(x::Matrix{Float64}, stomatal::Symbol)
    granges = stomatal === :medlyn ? MEDLYN_G : stomatal === :ballberry ? BALLBERRY_G : error("unknown model")
    cols = ntuple(length(PARAMETER_NAMES)) do j
        r = j <= PHYSICAL_DIMENSIONS ? RANGES[j] : granges[j - PHYSICAL_DIMENSIONS]
        ScaledColumn(view(x,:,j), r[1], r[2] - r[1])
    end
    return NamedTuple{PARAMETER_NAMES}(cols)
end

write_parquet(path::AbstractString, x::Matrix{Float64}, model::Symbol) =
    Parquet2.writefile(path, _table(x, model); compression_codec=:zstd)
read_parquet(path::AbstractString) = DataFrame(Parquet2.Dataset(path); copycols=false)

"""Exact O(N×d) LHS check; uses a one-bit occupancy vector, not O(N log N) sorting."""
function _lhs_ok(x::Matrix{Float64})
    n, d = size(x)
    for j in 1:d
        occupied = falses(n)
        @inbounds for i in 1:n
            u = x[i,j]
            (isfinite(u) && 0.0 <= u < 1.0) || return false
            stratum = min(n, floor(Int, u*n) + 1)
            occupied[stratum] && return false
            occupied[stratum] = true
        end
        all(occupied) || return false
    end
    return true
end

function _range_ok(x::Matrix{Float64}, model::Symbol)
    # Checks the normalized source, not the scaled columns (avoids 19 extra 10M-allocs).
    all(isfinite, x) && all(u -> 0.0 <= u < 1.0, x) || return false
    granges = model === :medlyn ? MEDLYN_G : BALLBERRY_G
    return all(r -> r[1] <= r[2], RANGES[1:PHYSICAL_DIMENSIONS]) &&
           all(r -> r[1] <= r[2], granges)
end

function _no_duplicate_rows(x::Matrix{Float64})
    # Valid LHS => distinct values per column => no two rows can be equal.
    return _lhs_ok(x)
end

function _physical_shared_ok(x::Matrix{Float64})
    medlyn, ballberry = _table(x, :medlyn), _table(x, :ballberry)
    for j in 1:PHYSICAL_DIMENSIONS
        a, b = medlyn[j], ballberry[j]
        # Same backing normalized column and identical physical affine map.
        (parent(a.u) === parent(b.u) && a.lo == b.lo && a.width == b.width) || return false
    end
    return true
end

# QC for one family: sizes, nesting, LHS occupancy, range, finiteness, no dupes,
# reproducibility (regenerated up to reproducibility_max_n).
function qc_family(layers::Vector{Matrix{Float64}}, seed::Int;
        expected_sizes::Vector{Int}=TRAIN_SIZES, check_reproducibility::Bool=true,
        reproducibility_max_n::Int=100_000)
    sizes = size.(layers, 1)
    actual_sizes = collect(sizes)
    count_ok = actual_sizes == expected_sizes
    nested_ok = all(layers[k][1:size(layers[k-1],1), :] == layers[k-1] for k in 2:length(layers))
    lhs_ok = all(_lhs_ok, layers)
    finite_ok = all(x -> all(isfinite, x), layers)
    range_ok = all(x -> _range_ok(x, :medlyn) && _range_ok(x, :ballberry), layers)
    duplicate_ok = all(_no_duplicate_rows, layers)
    # Medlyn/Ball-Berry tables share the same backing x, differ only in g0/g1 scaling.
    model_shared_ok = all(_physical_shared_ok, layers)
    reproducibility_sizes = [n for n in expected_sizes if n <= reproducibility_max_n]
    reproducible_ok = !check_reproducibility || (!isempty(reproducibility_sizes) &&
        all(a == b for (a,b) in zip(layers[1:length(reproducibility_sizes)],
            normalized_nested_lhs(seed; sizes=reproducibility_sizes))))
    ok = count_ok && nested_ok && lhs_ok && finite_ok && range_ok && duplicate_ok && model_shared_ok && reproducible_ok
    return (ok=ok, row_count=count_ok, range=range_ok, nested=nested_ok, lhs=lhs_ok,
        independent_replicates=true, reproducible=reproducible_ok,
        reproducibility_max_n=reproducibility_max_n, physical_shared=model_shared_ok,
        finite=finite_ok, no_duplicate_rows=duplicate_ok)
end

"""Create only training designs. Test-design generation is deliberately a separate future API."""
function generate_training_designs(; outdir::String=joinpath(@__DIR__, "..", "designs", "train"),
        family_seeds::Vector{Int}=REVISION_FAMILY_SEEDS, sizes::Vector{Int}=TRAIN_SIZES,
        reproducibility_max_n::Int=100_000,
        replicate_ids::Vector{Int}=collect(1:length(family_seeds)))
    sizes == TRAIN_SIZES || error("Production training sizes are fixed to $(TRAIN_SIZES)")
    !isempty(family_seeds) || error("family_seeds must be nonempty")
    all(>(0), family_seeds) || error("family_seeds must be positive")
    length(unique(family_seeds)) == length(family_seeds) || error("family_seeds must be unique")
    length(replicate_ids) == length(family_seeds) || error("replicate_ids must match family_seeds")
    all(>(0), replicate_ids) && length(unique(replicate_ids)) == length(replicate_ids) || error("replicate_ids must be unique positive integers")
    manifest = NamedTuple[]
    signatures = Set{String}()
    for (rep, seed) in zip(replicate_ids, family_seeds)
        layers = normalized_nested_lhs(seed; sizes=sizes)
        qc = qc_family(layers, seed; expected_sizes=sizes,
            reproducibility_max_n=reproducibility_max_n)
        qc.ok || error("QC failed for replicate $(rep): $(qc)")
        push!(signatures, bytes2hex(sha256(reinterpret(UInt8, vec(layers[1])))))
        repdir = joinpath(outdir, "rep" * lpad(string(rep), 2, '0'))
        mkpath(repdir)
        for (n, x) in zip(sizes, layers)
            for model in (:medlyn, :ballberry)
                path = joinpath(repdir, "n$(n)_$(model).parquet")
                write_parquet(path, x, model)
                push!(manifest, (replicate=rep, family_seed=seed, sample_size=n,
                    stomatal_model=String(model), rows=size(x,1), columns=size(x,2),
                    output_path=relpath(path, @__DIR__), generation_method=METHOD, qc_result=qc.ok,
                    qc_row_count=qc.row_count, qc_range=qc.range, qc_nested=qc.nested,
                    qc_lhs=qc.lhs, qc_reproducible=qc.reproducible,
                    qc_physical_shared=qc.physical_shared, qc_finite=qc.finite,
                    qc_no_duplicate_rows=qc.no_duplicate_rows,
                    qc_independent_replicates=true))
            end
        end
    end
    length(signatures) == length(family_seeds) || error("QC failed: replicate 1k designs are not independent")
    manifest_path = joinpath(outdir, "manifest.parquet")
    mkpath(dirname(manifest_path))
    new_manifest = DataFrame(manifest)
    if isfile(manifest_path)
        existing = DataFrame(Parquet2.Dataset(manifest_path); copycols=false)
        # new_manifest first: unique! keeps fresh rows over stale ones on re-run.
        new_manifest = unique!(vcat(new_manifest, existing; cols=:union), :output_path)
    end
    Parquet2.writefile(manifest_path, new_manifest; compression_codec=:zstd)
    return new_manifest
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .NestedC3LHS
    # julia 01_nested_lhs.jl            -> all 5 replicate families (301, 717, 815, 1003, 1009)
    # julia 01_nested_lhs.jl 301        -> just replicate 1 (rep01)
    seeds = isempty(ARGS) ? NestedC3LHS.REVISION_FAMILY_SEEDS : parse.(Int, ARGS)
    m = generate_training_designs(family_seeds=seeds)
    show(m, allrows=true)
end
