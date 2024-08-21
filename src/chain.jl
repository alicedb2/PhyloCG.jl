mutable struct Chain
    sampler::Sampler
    cgtree::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{@NamedTuple{k::Int64, n::Int64}}}
    params_chain::Vector{ComponentArray{Float64}}
    logprob_chain::Vector{Float64}
    maxsubtree
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
Base.argmax(chain::Chain) = argmax(chain.logprob_chain)
bestsample(chain::Chain) = chain.params_chain[argmax(chain.logprob_chain)]

function ess_rhat(chain::Chain, syms...; burn=0)
    return ess_rhat(chainsamples(chain, syms...; burn=burn))
end

function burn!(chain, burn=0)
    if burn > 0
        chain.logprob_chain = chain.logprob_chain[burn+1:end]
        chain.params_chain = chain.params_chain[burn+1:end]
        chain.sampler.current_logprob = chain.logprob_chain[end]
        chain.sampler.params = chain.params_chain[end]
    elseif burn < 0
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

function advance_chain!(chain, n_iter)
    s = chain.sampler

    prog = Progress(n_iter, showspeed=true)

    for n in 1:n_iter
        isfile("stop") && break
        advance!(chain.sampler, chain.cgtree, maxsubtree=chain.maxsubtree)
        push!(chain.logprob_chain, s.current_logprob)
        push!(chain.params_chain, deepcopy(s.params))
        next!(prog, desc="$n")
    end
    finish!(prog)
    return chain
end