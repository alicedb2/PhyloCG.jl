abstract type Sampler end

dims(sampler::Sampler) = sum(sampler.mask)

function setmodel!(params::ComponentArray{Float64}, mask::ComponentArray{Bool}, model::String="fbd")

    if contains(model, "f")
        mask.f = true
        if params.f == 1.0
            params.f = 0.999
        end
    else
        mask.f = false
        params.f = 1.0
    end

    if contains(model, "b")
        mask.b = true
        if params.b == 0.0
            params.b = 1.0
        end
    else
        mask.b = false
        params.b = 0.0
    end

    if contains(model, "d")
        mask.d = true
        if params.d == 0.0
            params.d = 1.0
        end
    else
        mask.d = false
        params.d = 0.0
    end

    if contains(model, "i")
        mask.i .= true
        if params.i.rho == 0.0
            params.i.rho = 1.0
            params.i.g = 0.5
        end
    else
        mask.i .= false
        params.i.rho = 0.0
        params.i.g = 0.5
    end

    if contains(model, "h")
        mask.h .= true
        if params.h.eta == 0.0
            params.h.eta = 1.0
            params.h.alpha = 5.0
            params.h.beta = 2.0
        end
    else
        mask.h .= false
        params.h.eta = 0.0
        params.h.alpha = 5.0
        params.h.beta = 2.0
    end

    return params, mask
end

function setmodel!(sampler::Sampler, model::String="fbd")
    setmodel!(sampler.params, sampler.mask, model)
    return sampler
end

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

function AMWG(model="fbd")
    params = initparams()
    mask = similar(params, Bool)
    sampler = AMWG(
            -Inf,
            params,
            mask,
            similar(params, Float64) .= 0.0,
            similar(params, Int) .= 0,
            similar(params, Int) .= 0,
            0, 10, 0, 0.44, 0.01)
    setmodel!(sampler, model)
    return sampler
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

mutable struct AM <: Sampler
    current_logprob::Float64
    params::ComponentArray{Float64}
    mask::ComponentArray{Bool}
    iter::Int
    acc::Int
    rej::Int
    safety_beta::Float64
    safety_sigma::Float64
    empirical_x::Vector{Float64}
    empirical_xx::Matrix{Float64}
end

function AM(model="fbd")
    params = initparams()
    mask = similar(params, Bool)
    setmodel!(params, mask, model)
    d = sum(mask)
    sampler = AM(
            -Inf,
            params,
            mask,
            1,
            0,
            0,
            0.05,
            0.01,
            zeros(d),
            zeros(d, d))
    return sampler
end

function am_sigma(L::Int, x::Vector{Float64}, xx::Matrix{Float64}; correction=true, eps::Float64=1e-10)
    sigma = (xx - x * x' / L) / (L - 1)
    if correction
        sigma = (sigma + sigma') / 2 + eps * I
    end
    return sigma
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

function advance!(sampler::AM, cgtree)
    s = sampler
    d = dims(sampler)

    if s.iter <= 2 * d
        proposal_dist = MvNormal(s.safety_sigma^2 / d * I(d))
    else
        safety_component = MvNormal(s.safety_sigma^2 / d * I(d))
        
        empirical_sigma = am_sigma(s.iter, s.empirical_x, s.empirical_xx)
        empirical_component = MvNormal(2.38^2 / d * empirical_sigma)
        
        proposal_dist = MixtureModel([safety_component, empirical_component], [s.safety_beta, 1 - s.safety_beta])
    end

    p_new = deepcopy(s.params)
    p_new[s.mask] .+= rand(proposal_dist)

    logprob_new = logdensity(cgtree, p_new)
    accprob = logprob_new - s.current_logprob
    if log(rand()) < accprob
        s.params .= p_new
        s.current_logprob = logprob_new
        s.acc += 1
    else
        s.rej += 1
    end

    s.iter += 1
    
    return s

end