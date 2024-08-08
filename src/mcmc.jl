abstract type Sampler end

mutable struct AMWG <: Sampler
    current_logprob::Float64
    params::ComponentArray{Float64}
    mask::ComponentArray{Bool}
    logscales::ComponentArray{Float64}
    acc::ComponentArray{Int}
    rej::ComponentArray{Int}
    iter::Int
    batch_size::Int
    nb_batches::Int
    acceptance_target::Float64
    min_delta::Float64
end

function AMWG(; kwargs...)
    params = initparams(kwargs...)
    mask = similar(params, Bool) .= true

    return AMWG(
        -Inf,
        params,
        mask,
        similar(params, Float64) .= 0.0,
        similar(params, Int) .= 0,
        similar(params, Int) .= 0,
        0, 10, 0, 0.44, 0.01)
end

function setmodel!(sampler::AMWG, model::String="fbd")

    if contains(model, "f")
        sampler.mask.f = true
        if sampler.params.f == 1.0
            sampler.params.f = 0.999
        end
    else
        sampler.mask.f = false
        sampler.params.f = 1.0
    end

    if contains(model, "b")
        sampler.mask.b = true
        if sampler.params.b == 0.0
            sampler.params.b = 1.0
        end
    else
        sampler.mask.b = false
        sampler.params.b = 0.0
    end

    if contains(model, "d")
        sampler.mask.d = true
        if sampler.params.d == 0.0
            sampler.d = 1.0
        end
    else
        sampler.mask.d = false
        sampler.params.d = 0.0
    end

    if contains(model, "i")
        sampler.mask.i .= true
        if sampler.params.i.rho == 0.0
            sampler.params.i.rho = 1.0
        end
    else
        sampler.mask.i .= false
        sampler.params.i.rho = 0.0
        sampler.params.i.g = 0.5
    end

    if contains(model, "h")
        sampler.mask.h .= true
        if sampler.params.h.eta == 0.0
            sampler.params.h.eta = 1.0
            sampler.params.h.alpha = 5.0
            sampler.params.h.beta = 2.0
        end
    else
        sampler.mask.h .= false
        sampler.params.h.eta = 0.0
        sampler.params.h.alpha = 5.0
        sampler.params.h.beta = 2.0
    end

    return sampler
end

mutable struct Chain
    sampler::Sampler
    cgtree::Dict{Tuple{Float64, Float64}, Vector{Tuple{Int64, Int64}}}
    params_chain::Vector{ComponentArray{Float64}}
    logprob_chain::Vector{Float64}
end

function Chain(cgtree, model="fbd")
    sampler = AMWG()
    setmodel!(sampler, model)
    return Chain(sampler, cgtree, [], [])
end

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

function bestsample(chain::Chain)
    idx = argmax(chain.logprob_chain)
    return chain.params_chain[idx]
end

function ess_rhat(chain::Chain, syms...; burn=0)
    return ess_rhat(chainsamples(chain, syms...; burn=burn))
end

function adjustlogscales!(sampler::AMWG; clearrates=true)
    s = sampler
    
    delta_n = min(s.min_delta, 1/sqrt(s.nb_batches))

    acc_rates = s.acc[s.mask] ./ (s.acc[s.mask] .+ s.rej[s.mask])
    s.logscales[s.mask] .+= delta_n .* (acc_rates .> s.acceptance_target) .- delta_n .* (acc_rates .<= s.acceptance_target)

    if clearrates
        s.acc .= 0
        s.rej .= 0
    end

    return s
end


function advance!(sampler::AMWG, cgtree)
    s = sampler

    if s.iter > s.batch_size
        adjustlogscales!(s)
        s.iter = 1
    end

    for i in findall(s.mask[:])
        p_new = deepcopy(s.params)
        p_new[i] += exp(s.logscales[i]) * randn()
        logprob_new = logdensity(cgtree, p_new)
        accprob = logprob_new - s.current_logprob
        if log(rand()) < accprob
            s.params .= p_new
            s.current_logprob = logprob_new
            s.acc[i] += 1
        else
            s.rej[i] += 1
        end
    end

    s.iter += 1

    return s
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
