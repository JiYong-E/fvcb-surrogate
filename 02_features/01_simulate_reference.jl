#!/usr/bin/env julia

# C3 reference-simulation runner for nested-LHS Parquet designs.
# Usage: julia --project=<env-with-LeafGasExchange.jl> 01_simulate_reference.jl INPUT.parquet OUTPUT_DIR [chunk_size] [max_rows]
# OUTPUT_DIR is a resumable Parquet dataset (one part-*.parquet per chunk); don't reuse it for a different input.

using DataFrames
using Parquet2
using Tables
using Cropbox
using LeafGasExchange
using Unitful
using Base.Threads
using SHA

const INPUT_COLUMNS = (:T_air, :RH, :wind, :PFD, :CO2, :w,
    :Vcm25, :Jm25, :Rd25, :Tp25, :Gamma25, :EaVc, :Eaj,
    :Ear, :EaTp, :Hj, :Sj, :g0, :g1)
const OUTPUT_COLUMNS = (:Anet_sim, :Agross_sim, :Ac_sim, :Aj_sim, :Ap_sim, :Rd_sim,
    :gs_sim, :gvc_sim, :Ci_sim, :Tleaf_sim, :sim_ok, :sim_status, :sim_error)
const RUNNER_VERSION = "c3_lhs_parquet_runner_v2.0.0"

getnum(x) = x isa Unitful.Quantity ? Unitful.ustrip(x) : Float64(x)
@inline getsim(sim::DataFrame, sym::Symbol) = hasproperty(sim, sym) ? getnum(only(sim[!, sym])) : NaN

function model_from_path(path::AbstractString)
    name = lowercase(basename(path))
    if occursin("ballberry", name) || occursin("_bb", name)
        return LeafGasExchange.ModelC3BB, :StomataBallBerry
    elseif occursin("medlyn", name) || occursin("_md", name)
        return LeafGasExchange.ModelC3MD, :StomataMedlyn
    end
    error("Cannot infer stomatal model from input file name: $(basename(path))")
end

# Groups the 19 design columns into the LeafGasExchange config tuple (direct Cropbox
# passthrough, except Gamma25 -> Γ25).
function config_from_row(row, stomata_system::Symbol)
    return (
        :Weather => (T_air=row.T_air, RH=row.RH, PFD=row.PFD,
                     CO2=row.CO2, wind=max(0.1, Float64(row.wind))),
        :BoundaryLayer => (w=row.w,),
        :C3c => (Vcm25=row.Vcm25, EaVc=row.EaVc),
        :C3j => (Jm25=row.Jm25, Eaj=row.Eaj, Hj=row.Hj, Sj=row.Sj),
        :C3r => (Rd25=row.Rd25, Ear=row.Ear, Γ25=row.Gamma25),
        :C3p => (Tp25=row.Tp25, EaTp=row.EaTp),
        stomata_system => (g0=row.g0, g1=row.g1),
        :Controller => (),
    )
end

function completed_part(path::AbstractString, expected_rows::Int)
    isfile(path) || return false
    try
        ds = Parquet2.Dataset(path)
        return Tuple(Tables.schema(ds).names) == (INPUT_COLUMNS..., OUTPUT_COLUMNS...) &&
               length(Parquet2.load(ds, String(first(INPUT_COLUMNS)))) == expected_rows
    catch
        return false
    end
end

function sha256_stream(path::AbstractString)
    ctx = SHA.SHA2_256_CTX(); buffer = Vector{UInt8}(undef, 8 * 1024 * 1024)
    open(path, "r") do io
        while !eof(io)
            n = readbytes!(io, buffer)
            n > 0 && SHA.update!(ctx, view(buffer, 1:n))
        end
    end
    return bytes2hex(SHA.digest!(ctx))
end

function initialize_or_validate_metadata(outdir, input_path, input_sha, model_name, rows, chunk_size)
    path = joinpath(outdir, "run_metadata.parquet")
    if isfile(path)
        old = DataFrame(Parquet2.Dataset(path); copycols=false)
        nrow(old) == 1 || error("invalid run metadata: $(path)")
        rec = only(eachrow(old))
        rec.input_sha256 == input_sha || error("refusing resume: input checksum differs")
        rec.input_rows == rows || error("refusing resume: input row count differs")
        rec.stomatal_model == model_name || error("refusing resume: stomatal model differs")
        rec.chunk_size == chunk_size || error("refusing resume: chunk size differs")
        rec.runner_version == RUNNER_VERSION || error("refusing resume: runner version differs")
    else
        metadata = DataFrame(input_path=[input_path], input_sha256=[input_sha], input_rows=[rows],
            stomatal_model=[model_name], chunk_size=[chunk_size], runner_version=[RUNNER_VERSION])
        Parquet2.writefile(path, metadata; compression_codec=:zstd)
    end
end

function simulate_chunk(chunk::DataFrame, model, stomata_system::Symbol)
    n = nrow(chunk)
    anet = fill(NaN, n); agross = fill(NaN, n); ac = fill(NaN, n); aj = fill(NaN, n)
    ap = fill(NaN, n); rd = fill(NaN, n); gs = fill(NaN, n); gvc = fill(NaN, n)
    # Do not use `falses(n)`: BitVector packs bits and concurrent writes race.
    ci = fill(NaN, n); tleaf = fill(NaN, n); ok = fill(false, n); status = fill("error", n)
    errors = fill("", n)
    done = Atomic{Int}(0)

    Threads.@threads for i in 1:n
        try
            sim = simulate(model; config=config_from_row(chunk[i, :], stomata_system))
            anet[i] = getsim(sim, :A_net); ac[i] = getsim(sim, :Ac); aj[i] = getsim(sim, :Aj)
            ap[i] = getsim(sim, :Ap); rd[i] = getsim(sim, :Rd); gs[i] = getsim(sim, :gs)
            gvc[i] = getsim(sim, :gvc); ci[i] = getsim(sim, :Ci); tleaf[i] = getsim(sim, :T)
            agross[i] = hasproperty(sim, :A_gross) ? getsim(sim, :A_gross) : anet[i] + rd[i]
            ok[i] = all(isfinite, (anet[i], ac[i], aj[i], ap[i], rd[i], gs[i], gvc[i], ci[i], tleaf[i]))
            status[i] = ok[i] ? "ok" : "invalid_output"
        catch err
            errors[i] = sprint(showerror, err)
        end
        c = atomic_add!(done, 1) + 1
        c % 50_000 == 0 && @info "chunk progress" done=c total=n
    end

    out = copy(chunk)
    out.Anet_sim = anet; out.Agross_sim = agross; out.Ac_sim = ac; out.Aj_sim = aj
    out.Ap_sim = ap; out.Rd_sim = rd; out.gs_sim = gs; out.gvc_sim = gvc
    out.Ci_sim = ci; out.Tleaf_sim = tleaf; out.sim_ok = ok; out.sim_status = status
    out.sim_error = errors
    return out
end

function run_file(inpath::AbstractString, outdir::AbstractString; chunk_size::Int=50_000,
                  max_rows::Union{Nothing,Int}=nothing, failure_warn_rate::Float64=0.01)
    isfile(inpath) || error("Input Parquet does not exist: $inpath")
    ds = Parquet2.Dataset(inpath)
    Tuple(Tables.schema(ds).names) == INPUT_COLUMNS || error("Input schema must be the exact 19-column revision LHS schema")
    df = DataFrame(ds; copycols=false)  # 10M×19 Float64 is ~1.5 GiB; deliberately bounded to one input file.
    !isnothing(max_rows) && (df = df[1:min(max_rows, nrow(df)), :])
    model, stomata_system = model_from_path(inpath)
    mkpath(outdir)
    input_sha = sha256_stream(inpath)
    initialize_or_validate_metadata(outdir, basename(inpath), input_sha, String(stomata_system), nrow(df), chunk_size)
    @info "reference simulation" input=inpath rows=nrow(df) model=string(stomata_system) threads=Threads.nthreads()

    summary = NamedTuple[]
    for first in 1:chunk_size:nrow(df)
        chunk_last = min(first + chunk_size - 1, nrow(df))
        part = joinpath(outdir, "part-" * lpad(string(div(first - 1, chunk_size) + 1), 6, '0') * ".parquet")
        if completed_part(part, chunk_last - first + 1)
            @info "skipping validated completed part" part=basename(part)
            continue
        end
        part_started = time()
        out = simulate_chunk(df[first:chunk_last, :], model, stomata_system)
        Parquet2.writefile(part, out; compression_codec=:zstd)
        push!(summary, (part=basename(part), first_row=first, last_row=chunk_last, rows=nrow(out),
            sim_ok=count(out.sim_ok), sim_failed=count(!, out.sim_ok)))
        failed_rate = summary[end].sim_failed / summary[end].rows
        failed_rate > failure_warn_rate && @warn "high simulation failure rate" part=basename(part) failed_rate threshold=failure_warn_rate
        elapsed_s = time() - part_started
        rate = summary[end].rows / elapsed_s
        eta_s = (nrow(df) - chunk_last) / rate
        @info "part saved" part=basename(part) sim_ok=summary[end].sim_ok sim_failed=summary[end].sim_failed elapsed_s=round(elapsed_s, digits=1) rows_per_s=round(rate, digits=2) eta_hours=round(eta_s / 3600, digits=2)
        GC.gc()
    end
    isempty(summary) || Parquet2.writefile(joinpath(outdir, "run_summary.parquet"), DataFrame(summary); compression_codec=:zstd)
    return nothing
end

function main()
    length(ARGS) >= 2 || error("Usage: INPUT.parquet OUTPUT_DIR [chunk_size] [max_rows]")
    chunk_size = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50_000
    max_rows = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
    run_file(abspath(ARGS[1]), abspath(ARGS[2]); chunk_size=chunk_size, max_rows=max_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
