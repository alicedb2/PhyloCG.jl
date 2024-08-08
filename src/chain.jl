mutable struct Chain
    sampler::Sampler
    cgtree::Dict{@NamedTuple{t::Float64, s::Float64}, Vector{@NamedTuple{k::Int64, n::Int64}}}
    params_chain::Vector{ComponentArray{Float64}}
    logprob_chain::Vector{Float64}
end

Chain(cgtree, sampler) = Chain(sampler, cgtree, [], [])

function chainsamples(chain::Chain, syms...; burn=0)
    if burn < 0
        burn = length(chain.logprob_chain) + burn
    end
    
    if first(syms) isa Int
        return [p[first(syms)] for p in chain.params_chain[burn+1:end]]
    else
        if syms === (:logdensity,)
            return chain.logprob_chain[burn+1:end]
        else
            return [reduce((x, s) -> getproperty(x, s), syms, init=p) for p in chain.params_chain[burn+1:end]]
        end
    end
end

Base.length(chain::Chain) = length(chain.logprob_chain)
Base.argmax(chain::Chain) = argmax(chain.logprob_chain)
bestsample(chain::Chain) = chain.params_chain[argmax(chain.logprob_chain)]

function ess_rhat(chain::Chain, syms...; burn=0)
    return ess_rhat(chainsamples(chain, syms...; burn=burn))
end


function advance_chain!(chain, n_iter)
    s = chain.sampler

    prog = Progress(n_iter, showspeed=true)

    for n in 1:n_iter
        isfile("stop") && break
        advance!(chain.sampler, chain.cgtree)
        push!(chain.logprob_chain, s.current_logprob)
        push!(chain.params_chain, deepcopy(s.params))
        next!(prog)
    end

    return chain
end
