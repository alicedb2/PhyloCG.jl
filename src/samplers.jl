abstract type Sampler end

dims(sampler::Sampler) = sum(sampler.mask)

function _setmodel!(sampler::Sampler, model::String="fbd")
    _setmodel!(sampler.params, sampler.mask, model)
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


"""
    AMWG(model="fbd")

Create a new Adaptive-Metropolis-Within-Gibbs (AMWG) sampler with the given model.

### Arguments
- `model::String`: the model to use. Can be any combination of "f", "b", "d", "i", and "h".
  - "f": Incomplete lineage sampling. Adds a sampling rate `f` to the model, otherwise the rate is fixed to 1.
  - "b": Birth/speciation model. Adds a birth/speciation rate `b` to the model.
  - "d": Death/extinction model. Adds a death/extinction rate `d` to the model.
  - "i": (Geometric) innovation model. Adds an innovation rate `rho` and geometric shape parameter `g` to the model. 
  - "h": (Beta-geometric) heterogeneous innovation model. Adds an heterogeneous innovation rate `eta` and beta-geometric burst shape parameters `alpha` and `beta` to the model.

### Returns
- `sampler::AMWG`: the new AMWG sampler
"""
function AMWG(model="fbd")
    params = initparams()
    mask = similar(params, Bool)
    logscales = fill!(similar(params, Float64), -2.0)
    sampler = AMWG(
            -Inf,       # current_logprob
            params,     # params
            mask,       # mask
            logscales,  # logscales
            fill!(similar(params, Int), 0), # accepted
            fill!(similar(params, Int), 0), # rejected
            0, 20, 0, 0.44, 0.1) # iter, batch_size, nb_batches, acceptance_target, min_delta
    _setmodel!(sampler, model)
    return sampler
end

"""
    acceptancerate(sampler::AMWG)

Return the acceptance rate of the sampler for the parameters of the current model mask.
"""
function acceptancerate(sampler::AMWG)
    return sampler.acc[sampler.mask] ./ (sampler.acc[sampler.mask] .+ sampler.rej[sampler.mask])
end

function _adjustlogscales!(sampler::AMWG; clearrates=true)
    s = sampler

    delta_n = min(s.min_delta, 1/sqrt(s.nb_batches))

    acc_rates = acceptancerate(s)
    s.logscales[s.mask] .+= delta_n .* (acc_rates .> s.acceptance_target) .- delta_n .* (acc_rates .< s.acceptance_target)

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
    _setmodel!(params, mask, model)
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

mutable struct LatentSlice <: Sampler
    current_logprob::Float64
    params::ComponentArray{Float64}
    mask::ComponentArray{Bool}
    iter::Int
    latent_s::ComponentArray{Float64}
    scales::ComponentArray{Float64}
    lbnds::ComponentArray{Float64}
    ubnds::ComponentArray{Float64}
end

function LatentSlice(model="fbd")
    params = initparams()
    mask = similar(params, Bool)
    _setmodel!(params, mask, model)

    scales = fill!(similar(params), 1.0)
    # scales.f = 0.2
    # scales.i.g = 0.2

    lbnds = fill!(similar(params), 0.0)
    ubnds = fill!(similar(params), Inf)
    ubnds.f = 1.0
    ubnds.b = 12.0
    ubnds.d = 12.0
    ubnds.i.rho = 12.0
    ubnds.i.g = 1.0
    ubnds.h.eta = 12.0
    ubnds.h.alpha = 20.0
    ubnds.h.beta = 5.0

    return LatentSlice(
        -Inf,
        params,
        mask,
        1,
        fill!(similar(params), 0.5),
        scales,
        lbnds,
        ubnds
    )
end

"""
    advance!(sampler::AMWG, cgtree; maxsubtree=Inf)

Advance the AMWG sampler by one iteration.

### Returns
- `sampler::AMWG`: the updated sampler

# Notes
If the heterogeneous innovation process is included in the model, the sampler uses 
a hyperbolic change of variable `(eta, alpha, beta) -> (u, v, w)`, where 
`u = 1/2 * log(eta / alpha)`, `v = sqrt(eta * alpha)`, and `beta / sqrt(eta * alpha)`,
and adds the jacobian of the transformation to the acceptance probability. This
change of variable improves the mixing of the sampler by removing strong correlations
between the parameters of the heterogeneous innovation model. The inverse transformation
is given by `eta = v * exp(u)`, `alpha = v * exp(-u)`, and `beta = v * w`.
"""
function advance!(sampler::AMWG, cgtree; maxsubtree=Inf)
    s = sampler

    if s.iter > s.batch_size
        s.nb_batches += 1
        _adjustlogscales!(s)
        s.iter = 1
    end

    for i in shuffle!(findall(s.mask[:]))
        p_new = deepcopy(s.params)
        if i == 6
            du = exp(s.logscales[6]) * randn()
            p_new[6:8] .= _eab((_uvw(s.params.h.eta, s.params.h.alpha, s.params.h.beta) .+ [du, 0.0, 0.0])...)
        elseif i == 7
            dv = exp(s.logscales[7]) * randn()
            p_new[6:8] .= _eab((_uvw(s.params.h.eta, s.params.h.alpha, s.params.h.beta) .+ [0.0, dv, 0.0])...)
        elseif i == 8
            dw = exp(s.logscales[8]) * randn()
            p_new[6:8] .= _eab((_uvw(s.params.h.eta, s.params.h.alpha, s.params.h.beta) .+ [0.0, 0.0, dw])...)
        else
            p_new[i] += exp(s.logscales[i]) * randn()
        end
        accprob = 0.0
        if p_new.h.eta > 0.0
            accprob += logjac_deabduvw(p_new.h.eta, p_new.h.alpha, p_new.h.beta) - logjac_deabduvw(s.params.h.eta, s.params.h.alpha, s.params.h.beta)
        end

        logprob_new = logdensity(cgtree, p_new; maxsubtree=maxsubtree)
        if logprob_new > -Inf
            accprob += logprob_new - s.current_logprob
        else
            accprob = -Inf
        end

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

function advance!(sampler::AM, cgtree; maxsubtree=Inf)
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

    logprob_new = logdensity(cgtree, p_new; maxsubtree=maxsubtree)
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

# One parameter updated at a time
function advance!(sampler::LatentSlice, cgtree; maxsubtree=Inf)
    s = sampler

    for i in shuffle!(findall(s.mask[:]))
        log_w = s.current_logprob + log(rand())
        l = rand(Uniform(s.params[i] - s.latent_s[i]/2, s.params[i] + s.latent_s[i]/2))
        latent_s = 2 * abs(l - s.params[i]) + rand(Exponential(s.scales[i]))
        s.latent_s[i] = latent_s

        lbnd = max(s.lbnds[i], l - latent_s/2)
        ubnd = min(s.ubnds[i], l + latent_s/2)

        new_params = copy(s.params)
        new_params[i] = rand(Uniform(lbnd, ubnd))
        logprob = logdensity(cgtree, new_params, maxsubtree=maxsubtree)
        while log_w > logprob
            if new_params[i] < s.params[i]
                lbnd = max(lbnd, new_params[i])
            else
                ubnd = min(ubnd, new_params[i])
            end
            new_params[i] = rand(Uniform(lbnd, ubnd))
            logprob = logdensity(cgtree, new_params, maxsubtree=maxsubtree)
        end
        s.params = new_params
        s.current_logprob = logprob
    end

    s.iter += 1

    return s

end

# All parameters updated at once
function _advance!(sampler::LatentSlice, cgtree, maxsubtree=Inf)
    s = sampler
    mask = s.mask

    log_w = s.current_logprob + log(rand())

    l = rand.(Uniform.(s.params[mask] .- s.latent_s[mask]/2, s.params[mask] .+ s.latent_s[mask]/2))
    latent_s = 2 * abs.(l .- s.params[mask]) .+ rand.(Exponential.(s.scales[mask]))
    s.latent_s[mask] .= latent_s


    lbox = max.(s.lbnds[mask], l .- latent_s ./ 2)
    ubox = min.(s.ubnds[mask], l .+ latent_s ./ 2)

    new_params = copy(s.params)
    new_params[mask] .= rand.(Uniform.(lbox, ubox))
    logprob = logdensity(cgtree, new_params, maxsubtree=maxsubtree)
    while log_w > logprob
        for i in dims(s)
            if new_params[mask][i] < s.params[mask][i]
                lbox[i] = max(lbox[i], new_params[mask][i])
            else
                ubox[i] = min(ubox[i], new_params[mask][i])
            end
        end
        new_params[mask] .= rand.(Uniform.(lbox, ubox))
        println(new_params)
        logprob = logdensity(cgtree, new_params, maxsubtree=maxsubtree)
    end

    s.params = new_params
    s.current_logprob = logprob

    s.iter += 1

    return s

end