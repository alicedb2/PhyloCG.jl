mutable struct GOFChain
    cgtree::CGTree
    params::ComponentArray{Float64}
    slicelogphis::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}
    cgtree_chain::Vector{CGTree}
    G_chain::Vector{Float64}
    rng::AbstractRNG
    accepted::Int
    rejected::Int
end

function GOFChain(cgtree, params; seed=nothing)
    cgtree = deepcopy(cgtree)
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
    return GOFChain(cgtree, params, slicelogphis, [deepcopy(cgtree)], [G], rng, 0, 0)
end

function Gstatistic(ssd::DefaultDict{Int, Int, Int}, logphis)
    freqs = values(ssd)
    mass = sum(freqs)
    return 2 * sum(freqs .* (log.(freqs) .- log(mass) .- getindex.(Ref(logphis), keys(ssd))))
end

function Gstatistic(cgtree, slicelogphis)
    G = 0.0
    for (ts, ssd) in cgtree
        G += Gstatistic(ssd, slicelogphis[ts])
    end
    return G
end

function advance_chain!(chain::GOFChain, nbiter; sampleevery=1)
    
    prog = Progress(nbiter, showspeed=true)
    
    for k in 1:nbiter
        isfile("stop") && break
        bouquet = popbouquet!(chain.cgtree)
        nbtips = sum(bouquet.crown.ks)
        proposed_crown = randompartitionAD5(nbtips, rng=chain.rng)
        proposed_trunk = length(proposed_crown)
        proposed_bouquet = (trunk=(i=bouquet.trunk.i, t=bouquet.trunk.t, s=bouquet.trunk.s, k=proposed_trunk),
                            crown=(i=bouquet.crown.i, t=bouquet.crown.t, s=bouquet.crown.s, ks=proposed_crown))

        # Make sure there are enough logphis for all new subtrees
        crownts = (t=bouquet.crown.t, s=bouquet.crown.s)
        crownlogphis = chain.slicelogphis[crownts]
        if 2 * maximum(proposed_crown) > length(crownlogphis)
            println("Resizing logphis for crown slice $(crownts)")
            crownlogphis = logphis(2 * maximum(proposed_crown), crownts.t, crownts.s, chain.params...)
            chain.slicelogphis[crownts] = crownlogphis
        end
        trunkts = (t=bouquet.trunk.t, s=bouquet.trunk.s)
        trunklogphis = chain.slicelogphis[trunkts]
        if 2 * proposed_trunk > length(chain.slicelogphis[trunkts])
            println("Resizing logphis for trunk slice $(trunkts)")
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
            pushbouquet!(chain.cgtree, proposed_bouquet)
            chain.accepted += 1
        else
            pushbouquet!(chain.cgtree, bouquet)
            chain.rejected += 1
        end

        G = Gstatistic(chain.cgtree, chain.slicelogphis)
        push!(chain.cgtree_chain, deepcopy(chain.cgtree))
        push!(chain.G_chain, G)
        next!(prog)

    end
end

function Base.length(chain::GOFChain)
    return length(chain.G_chain)
end

function burn!(chain::GOFChain, burn=0)
    # chain.accepted = 0
    # chain.rejected = 0
    if burn > 0
        if 0 < burn < 1
            burn = round(Int, burn * length(chain))
        end
        chain.cgtree_chain = chain.cgtree_chain[burn+1:end]
        chain.G_chain = chain.G_chain[burn+1:end]
    elseif burn < 0
        if -1 < burn < 0
            burn = round(Int, -burn * length(chain))
        end
        chain.cgtree_chain = chain.cgtree_chain[1:end+burn]
        chain.G_chain = chain.G_chain[1:end+burn]
    end
    return chain
end