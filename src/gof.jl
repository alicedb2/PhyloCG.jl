const GOF = @NamedTuple{cgtree_p::Float64, cgtree_Gstat::Float64, slice_ps::Dict{@NamedTuple{t::Float64, s::Float64}, @NamedTuple{p::Float64, G::Float64}}, gof_null::@NamedTuple{G_samples::Vector{Float64}, slice_Gs_samples::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}, modelssds::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}}}

mutable struct GOFChain
    cgtree::CGTree
    curr_cgtree::CGTree
    params::ComponentArray{Float64}
    slicelogphis::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}
    cgtree_chain::Vector{CGTree}
    G_chain::Vector{Float64}
    rng::AbstractRNG
    accepted::Int
    rejected::Int
    maxsubtree::Number
end

function GOFChain(cgtree, params; maxsubtree=Inf, seed=nothing)
    cgtree = deepcopy(cgtree)
    if isfinite(maxsubtree)
        for ssd in values(cgtree)
            filter!(ks -> ks[1] <= maxsubtree, ssd)
        end
    end
    curr_cgtree = deepcopy(cgtree)
    params = deepcopy(params)
    slicelogphis = logphis(cgtree, params...)
    G = Gstatistic(cgtree, slicelogphis)
    if seed === nothing
        rng = Random.GLOBAL_RNG
    elseif seed isa Int
        rng = Random.MersenneTwister(seed)
    elseif seed isa AbstractRNG
        rng = seed
    else
        throw(ArgumentError("seed must be an integer or an AbstractRNG"))
    end
    return GOFChain(cgtree, curr_cgtree, params, slicelogphis, [deepcopy(cgtree)], [G], rng, 0, 0, maxsubtree)
end

function GOFChain(chain::Chain, params=bestsample(chain); maxsubtree=chain.maxsubtree, seed=nothing)
    return GOFChain(chain.cgtree, params, maxsubtree=maxsubtree, seed=seed)
end

function acceptancerate(chain::GOFChain)
    return chain.accepted / (chain.accepted + chain.rejected)
end

@doc raw"""
    Gstatistic(ssd::T, logphis) where {T <: AbstractDict{Int, Int}}

Compute the G-statistic for a single subtree size distribution, e.g. a single slice of a coarse-grained tree.
The G-statistic is given by
```math
G = 2\sum_k O_k\ln\frac{O_k}{E_k}.
```
where `O_k` are observed frequencies of subtrees of size `k` and `E_i` are expected frequencies taken from `logphis[k]` calculated from the model.

### Arguments
- `ssd::AbstractDict{Int, Int}`: the subtree size distribution
- `logphis::Vector{Float64}`: the precomputed logphis for the slice

### Returns
- `G::Float64`: the G-statistic of the subtree size distribution
"""
function Gstatistic(ssd::T, logphis) where {T <: AbstractDict{Int, Int}}
    freqs = values(ssd)
    mass = sum(freqs)
    return 2 * sum(freqs .* (log.(freqs) .- log(mass) .- getindex.(Ref(logphis), keys(ssd))))
end

@doc raw"""
    Gstatistic(cgtree::CGTree, slicelogphis)

Compute the G-statistic for a coarse-grained tree which is the sum of the G-statistics of each slices.
```math
G = \sum_{(t, s)} G(ssd(t, s))
```

### Arguments
- `cgtree::CGTree`: the coarse-grained tree
- `slicelogphis`: the precomputed logphis for each slice `(t, s)`

### Returns
- `G::Float64`: the G-statistic of the whole coarse-grained tree
"""
function Gstatistic(cgtree::CGTree, slicelogphis; perslice=false)
    slice_Gs = Dict()
    G = 0.0
    for (ts, ssd) in cgtree
        sliceG = Gstatistic(ssd, slicelogphis[ts])
        G += sliceG
        slice_Gs[ts] = G
    end
    if perslice
        return (; G, slice_Gs)
    else
        return G
    end
end

"""
    advance_chain!(chain::GOFChain, nbiter; savecgtree=false, progressoutput=:repl)

Advance the goodness-of-fit chain by `nbiter` iterations and save the G-statistic at each iteration.

At each iteration the chain is advanced by as many sub-iterations as the
number of subtrees in the current state of the coarse-grained tree.

### Arguments
- `nbiter::Int`: the number of iterations to advance the chain by.
- `savecgtree::Bool=false`: if `true`, the current state of the coarse-grained tree is saved at each iteration.

### Returns
- `chain::GOFChain`: the chain

# Notes
- If a file named `stopgof` is found in the current directory, the chain is stopped gracefully.
- Each iteration consists of as many sub-iterations as the number of subtrees in the current coarse-grained tree at the beginning of the iteration.
"""
function advance_chain!(chain::GOFChain, nbiter; ess50target=200, savecgtree=false, progressoutput=:repl)

    if progressoutput === :repl
        progressio = stderr
    elseif progressoutput === :file
        progressoutput = "progress_pid$(getpid()).txt"
        progressio = open(progressoutput, "w")
    end

    if !(nbiter isa Int)
        prog = ProgressUnknown(showspeed=true, output=progressio)
        _nbiter = typemax(Int64)
    else
        prog = Progress(nbiter; showspeed=true, output=progressio)
        _nbiter = nbiter
    end

    for n in 1:_nbiter
        crown_resizes, trunk_resizes = 0, 0
        for _ in 1:nbsubtrees(chain.curr_cgtree)
            bouquet = _popbouquet!(chain.curr_cgtree)
            nbtips = sum(bouquet.crown.ks)
            proposed_crown = randompartitionAD5(nbtips, rng=chain.rng)
            proposed_trunk = length(proposed_crown)
            proposed_bouquet = (trunk=(i=bouquet.trunk.i, t=bouquet.trunk.t, s=bouquet.trunk.s, k=proposed_trunk),
                                crown=(i=bouquet.crown.i, t=bouquet.crown.t, s=bouquet.crown.s, ks=proposed_crown))

            # Make sure there are enough logphis for all new subtrees
            crownts = (t=bouquet.crown.t, s=bouquet.crown.s)
            crownlogphis = chain.slicelogphis[crownts]
            if 2 * maximum(proposed_crown) > length(crownlogphis)
                crown_resizes += 1
                # println("Resizing logphis for crown slice $(map(x->round(x, digits=3), crownts))")
                crownlogphis = logphis(2 * maximum(proposed_crown), crownts.t, crownts.s, chain.params...)
                chain.slicelogphis[crownts] = crownlogphis
            end
            trunkts = (t=bouquet.trunk.t, s=bouquet.trunk.s)
            trunklogphis = chain.slicelogphis[trunkts]
            if 2 * proposed_trunk > length(chain.slicelogphis[trunkts])
                trunk_resizes += 1
                # println("Resizing logphis for trunk slice $(trunkts)")
                trunklogphis = logphis(2 * proposed_trunk, trunkts.t, trunkts.s, chain.params...)
                chain.slicelogphis[trunkts] = trunklogphis
            end

            # Compute prob ratio
            logprob_ratio = trunklogphis[proposed_trunk] - trunklogphis[bouquet.trunk.k]
            logprob_ratio += sum(crownlogphis[proposed_crown]) - sum(crownlogphis[bouquet.crown.ks])

            # Compute Hastings ratio
            #  (gammaln(proposed_nb_parts + 1) - gammaln(np.array(list(Counter(proposed_part_sizes).values())) + 1).sum()
            # - gammaln(current_nb_parts + 1) + gammaln(np.array(list(Counter(current_part_sizes).values())) + 1).sum())
            loghastings = loggamma(length(proposed_crown) + 1) - sum(loggamma.(values(countmap(proposed_crown)) .+ 1))
            loghastings -= loggamma(length(bouquet.crown.ks) + 1) - sum(loggamma.(values(countmap(bouquet.crown.ks)) .+ 1))

            logacceptance_ratio = logprob_ratio + loghastings

            if log(rand(chain.rng)) < logacceptance_ratio
                _pushbouquet!(chain.curr_cgtree, proposed_bouquet)
                chain.accepted += 1
            else
                _pushbouquet!(chain.curr_cgtree, bouquet)
                chain.rejected += 1
            end

            isfile("stopgof") && break

        end

        G = Gstatistic(chain.curr_cgtree, chain.slicelogphis)
        push!(chain.G_chain, G)
        if savecgtree
            push!(chain.cgtree_chain, deepcopy(chain.curr_cgtree))
        end

        conv = "wait 40"
        if length(chain) > 40
            conv = ess_rhat(chain.G_chain[div(end, 2):end])
            conv = map(x->round(x, digits=3), conv)
            if nbiter === :ess && conv.ess >= ess50target
                println("Convergence target reached!")
                break
            end
        end

        if trunk_resizes > 0 || crown_resizes > 0
            note = "resized logphis: trunk=$trunk_resizes, crown=$crown_resizes"
        else
            note = ""
        end

        next!(prog, showvalues=[
            ("iteration", n),
            ("chain length", length(chain)),
            ("empirical G", @sprintf("%.3f", first(chain.G_chain))),
            ("current G", @sprintf("%.3f", last(chain.G_chain))),
            ("convergence (burn 50%)", conv),
            ("note", note)
        ])

        isfile("stopgof") && break

    end

    if progressoutput === :file
        close(progressio)
        rm(progressoutput)
    end

end

Base.length(chain::GOFChain) = length(chain.G_chain)

function burn!(chain::GOFChain, burn=0)
    chain.G_chain = chain.G_chain[_burnlength(length(chain), burn)+1:end]
    return chain
end

burn(chain::GOFChain, burn=0) = burn!(deepcopy(chain), burn)

function gof_null(nbsamples, f, b, d, rho, g, eta, alpha, beta; nbslices=8, age=1.0, treesize=1000, verbose=false, rng=default_rng())

    K = 2 * treesize

    if age isa Float64
        ts = LinRange(0.0, age, nbslices + 1)
        tss = @NamedTuple{s::Float64, t::Float64}.(collect(zip(ts[1:end-1], ts[2:end])))
    elseif collect(age) isa Vector{@NamedTuple{t::Float64, s::Float64}}
        tss = sort(collect(age), by=x->x.t)
    end

    modelssds = ModelSSDs()
    for(i, ts) in enumerate(tss)
        verbose && println("Computing model SSD for slice $i/$(length(tss)) (t=$(ts.t), s=$(ts.s))")
        modelssds[(; t=ts.t, s=ts.s)] = logphis(K, ts.t, ts.s, f, b, d, rho, g, eta, alpha, beta)
    end

    G_samples = Float64[]
    slice_Gs_samples = DefaultDict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}(Vector{Float64})

    for k in 1:nbsamples
        verbose && print("\r$k/$nbsamples")
        _cgtree, G, slice_Gs = generate_cgtree(modelssds; treesize=treesize, verbose=false, rng=rng)
        G_samples = [G_samples; G]
        for (ts, slice_G) in slice_Gs
            push!(slice_Gs_samples[ts], slice_G)
        end
    end
    println()

    return (; G_samples, slice_Gs_samples=Dict(slice_Gs_samples), modelssds)
end

gof_null(nbsamples, params::ComponentArray; nbslices=8, age=1.0, treesize=1000, verbose=false, rng=default_rng()) = gof_null(nbsamples, params.f, params.b, params.d, params.i.rho, params.i.g, params.h.eta, params.h.alpha, params.h.beta; nbslices=nbslices, age=age, treesize=treesize, verbose=verbose, rng=rng)

function gof(cgtree, params, nbsamples=200; verbose=true, rng=default_rng())

    _gof_null = gof_null(nbsamples, params; age=keys(cgtree), treesize=size(cgtree), verbose=verbose, rng=rng)

    slice_ps = Dict{@NamedTuple{t::Float64, s::Float64}, @NamedTuple{p::Float64, G::Float64}}()
    cgtree_Gstat = 0.0
    for (ts, G_samples) in _gof_null.slice_Gs_samples
        slice_Gstat = Gstatistic(cgtree[ts], _gof_null.modelssds[ts])
        slice_ps[ts] = (p=quantilerank(G_samples, slice_Gstat), G=slice_Gstat)
        cgtree_Gstat += slice_Gstat
    end
    cgtree_p = quantilerank(_gof_null.G_samples, cgtree_Gstat)

    return (; cgtree_p, cgtree_Gstat, slice_ps, gof_null=_gof_null)
end

function Base.show(io::IO, gof::GOF)
    println(io, "GOF")
    println(io, "cgtree p-value: $(round(gof.cgtree_p, digits=3))")
    println(io, "cgtree G-statistic: $(round(gof.cgtree_Gstat, digits=3))")
    println(io, "slice p-values:")
    for (ts, ps) in sort(gof.slice_ps, by=x->x[1])
        println(io, "  t=$(round(ts.t, digits=3)) s=$(round(ts.s, digits=3)):   p=$(round(ps.p, digits=3)),   G=$(round(ps.G, digits=3))")
    end
end