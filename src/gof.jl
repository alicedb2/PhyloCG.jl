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
function Gstatistic(cgtree::CGTree, slicelogphis)
    G = 0.0
    for (ts, ssd) in cgtree
        G += Gstatistic(ssd, slicelogphis[ts])
    end
    return G
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
- If a file named `stop` is found in the current directory, the chain is stopped gracefully.
- Each iteration consists of as many sub-iterations as the number of subtrees in the current coarse-grained tree at the beginning of the iteration.
"""
function advance_chain!(chain::GOFChain, nbiter; savecgtree=false, progressoutput=:repl)
    
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
        for _ in 1:size(chain.curr_cgtree)
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

            isfile("stop") && break

        end
        
        G = Gstatistic(chain.curr_cgtree, chain.slicelogphis)
        push!(chain.G_chain, G)
        if savecgtree
            push!(chain.cgtree_chain, deepcopy(chain.curr_cgtree))
        end

        conv = nothing
        if length(chain) > 40
            conv = ess_rhat(chain.G_chain[div(end, 2):end])
            conv = map(x->round(x, digits=3), conv)
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
        
        isfile("stop") && break

    end

    if progressoutput === :file
        close(progressio)
        rm(progressoutput)
    end
    
end

function Base.length(chain::GOFChain)
    return length(chain.G_chain)
end

function burn!(chain::GOFChain, burn=0)
    chain.G_chain = chain.G_chain[_burnpos(length(chain), burn)+1:end]
    return chain
end

function burn(chain::GOFChain, burn=0)
    chain = deepcopy(chain)
    return burn!(chain, burn)
end