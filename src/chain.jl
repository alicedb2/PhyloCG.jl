mutable struct Chain
    sampler::Sampler
    cgtree::CGTree
    params_chain::Vector{ComponentArray{Float64}}
    logprob_chain::Vector{Float64}
    maxsubtree::Real
end

function Chain(cgtree, sampler; maxsubtree=Inf)
    chain = Chain(sampler, cgtree, [], [], maxsubtree)
    chain.sampler.current_logprob = logdensity(cgtree, sampler.params, maxsubtree=maxsubtree)
    return chain
end

function chainsamples(chain::Chain, syms...; burn=0)
    if burn < 0
        slice = (length(chain.logprob_chain)+burn)+1:length(chain.logprob_chain)
    elseif burn > 0
        slice = burn+1:length(chain.logprob_chain)
    else
        slice = 1:length(chain.logprob_chain)
    end

    if first(syms) isa Int
        return [p[first(syms)] for p in chain.params_chain[slice]]
    else
        if syms === (:logdensity,)
            return chain.logprob_chain[slice]
        else
            return [reduce((x, s) -> getproperty(x, s), syms, init=p) for p in chain.params_chain[slice]]
        end
    end
end

Base.length(chain::Chain) = length(chain.logprob_chain)

function Base.argmax(chain::Chain, logprob=:map)
    if logprob === :map
        return argmax(chain.logprob_chain)
    elseif logprob === :mle
        return argmax(chain.logprob_chain .- log_priors.(chain.params_chain))
    elseif logprob === :prior
        return argmax(log_priors.(chain.params_chain))
    else
        throw(ArgumentError("logprob must be :map, :mle, or :prior"))
    end
end

bestsample(chain::Chain, logprob=:map) = chain.params_chain[argmax(chain, logprob)]

function ess_rhat(chain::Chain, syms...; burn=0)
    if isempty(syms)
        NamedTuple(ComponentArray(ess_rhat.(chainsamples.(Ref(chain), 1:length(chain.sampler.params), burn=burn)), getaxes(chain.sampler.params)))
    else
        return ess_rhat(chainsamples(chain, syms...; burn=burn))
    end
end

function burn!(chain, burn=0)
    if burn > 0
        if 0 < burn < 1
            burn = round(Int, burn * length(chain))
        end
        chain.logprob_chain = chain.logprob_chain[burn+1:end]
        chain.params_chain = chain.params_chain[burn+1:end]
        chain.sampler.current_logprob = chain.logprob_chain[end]
        chain.sampler.params = chain.params_chain[end]
    elseif burn < 0
        if -1 < burn < 0
            burn = round(Int, -burn * length(chain))
        end
        chain.logprob_chain = chain.logprob_chain[1:end+burn]
        chain.params_chain = chain.params_chain[1:end+burn]
        chain.sampler.current_logprob = chain.logprob_chain[end]
        chain.sampler.params = chain.params_chain[end]
    end
    return chain
end

function burn(chain, burn=0)
    return burn!(deepcopy(chain), burn)
end

function advance_chain!(chain::Chain, nbiter)
    s = chain.sampler

    prog = Progress(nbiter, showspeed=true)

    for n in 1:nbiter
        isfile("stop") && break
        advance!(chain.sampler, chain.cgtree, maxsubtree=chain.maxsubtree)
        push!(chain.logprob_chain, s.current_logprob)
        push!(chain.params_chain, deepcopy(s.params))
        next!(prog, desc="$n")
    end
    finish!(prog)
    return chain
end