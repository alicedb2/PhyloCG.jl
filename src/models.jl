function bdih(u, p, t)
    b, d, rho, g, eta, alpha, beta = p
    du = 0
    if b > 0
        du += b * (u - 1) * u
    end
    if d > 0
        du += d * (1 - u)
    end
    if rho > 0
        du += rho * g * (u - 1) * u / (1 - g * u)
    end
    if eta > 0
        du += eta * alpha / (alpha + beta) * (u - 1) * u * _₂F₁(1, alpha + 1, alpha + beta + 1, u)
        # du += eta * alpha / (alpha + beta) * (u - 1) * u * hyp2f1a1(alpha + 1, alpha + beta + 1, u)
    end
    return du
end

function bdih!(du, u, p, t)
    b, d, rho, g, eta, alpha, beta = p
    du[1] = 0
    if b > 0
        du[1] += b * (u[1] - 1) * u[1]
    end
    if d > 0
        du[1] += d * (1 - u[1])
    end
    if rho > 0
        du[1] += rho * g * (u[1] - 1) * u[1] / (1 - g * u[1])
    end
    if eta > 0
        # du[1] += eta * alpha / (alpha + beta) * (u[1] - 1) * u[1] * _₂F₁(1, alpha + 1, alpha + beta + 1, u[1])
        du[1] += eta * alpha / (alpha + beta) * (u[1] - 1) * u[1] * hyp2f1a1(alpha + 1, alpha + beta + 1, u[1])
    end
    return nothing
end

# function bdih_jac(u, p, t)
#     b, d, rho, g, eta, alpha, beta = p
#     J = 0
#     if b > 0
#         du += b * (u - 1) * u
#     end
#     if d > 0
#         du += d * (1 - u)
#     end
#     if rho > 0
#         du += rho * g * (u - 1) * u / (1 - g * u)
#     end
#     if eta > 0
#         du += eta * alpha / (alpha + beta) * (u - 1) * u * _₂F₁(1, alpha + 1, alpha + beta + 1, u)
#         # du += eta * alpha / (alpha + beta) * (u - 1) * u * hyp2f1a1(alpha + 1, alpha + beta + 1, u)
#     end
#     return du
# end

const _bdihprob = ODEProblem(bdih, 0 + 0im, (0, 1))

const _bdihprob_inplace = ODEProblem(bdih!, 0 + 0im, (0, 1))

function Ubdih(z, t, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)::Complex
    if t == 0
        return z
    end
    # sol = solve(prob, Tsit5(); 
    #     u0=z, tspan=(0.0, t), 
    #     p=[b, d, rho, g, eta, alpha, beta],
    #     reltol=1e-8, abstol=1e-8)
    # return sol.u[end]
    sol = solve(prob, Tsit5(); 
        u0=[z], tspan=(0.0, t), 
        p=[b, d, rho, g, eta, alpha, beta],
        reltol=1e-8, abstol=1e-8)
    return sol.u[end][1]
end

function Phi(y, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)::Complex
    @assert t >= s >= 0
    if y == 0
        return 0
    end
    Ut1f = Ubdih(1 - f, t, b, d, rho, g, eta, alpha, beta, prob)
    Us1f = Ubdih(1 - f, s, b, d, rho, g, eta, alpha, beta, prob)
    return (Ubdih(Us1f + y * (1 - Us1f), t - s, b, d, rho, g, eta, alpha, beta, prob) - Ut1f) / (1 - Ut1f)
end

function _powerof2ceil(n)
    return 2^ceil(Integer, log2(n))
end

function logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; gap=1/_powerof2ceil(truncN), optimizeradius=false)::Vector{Real}
    # Make sure the total number of state
    # is the closest power of two larger than n
    
    n = _powerof2ceil(truncN)

    if t < s || s < 0.0 || !(0 <= f <= 1) || b < 0 || d < 0 || rho < 0 || !(0 <= g <= 1) || eta < 0 || alpha < 0 || beta < 0
        return fill(-Inf, n)
    end

    if optimizeradius
        r = bdihPhi_optimal_radius(n, t, s, f, b, d, rho, g, eta, alpha, beta, prob)
    else 
        r = bdihPhi_singularity(t, s, f, b, d, rho, g, eta, alpha, beta, prob) - gap
    end

    complex_halfcircle = r * exp.(2pi * im * (0:div(n, 2)) / n)

    # try
        Phi_samples = [Phi(z, t, s, f, b, d, rho, g, eta, alpha, beta, prob) for z in complex_halfcircle]
        # Phi_samples = Phi.(complex_halfcircle, t, s, f, b, d, rho, g, eta, alpha, beta, Ref(prob))
        upks = irfft(conj(Phi_samples), n) # Hermitian FFT
        # correct for 0
        # upks .-= upks[1]
        log_pks = [upk > 0 ? log(upk) : -Inf for upk in upks] - (0:n-1) .* log(r)
        # log_pks = log.(upks) - (0:n-1) .* log(r)

        return log_pks
    # catch e
        # return fill(-Inf, n)
    # end

end

function slicelogprob(subtreesizedistribution, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; maxsubtree=1000)

    truncN = maximum(first.(subtreesizedistribution)) + 1

    phiks = logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta, prob)[2:end]

    logprob = sum([n * phiks[k] for (k, n) in subtreesizedistribution])

    return logprob
end

function cgtreelogprob(cgtree, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; dropfirstslice=false, normalize=false, maxsubtree=Inf)
    logprob = 0.0
    N = 0
    for ((t, s), ssd) in cgtree
        if s == 0 && dropfirstslice
            continue
        end
        origN = sum(getfield.(ssd, :n))
        ssd = filter(st -> st.k <= maxsubtree, ssd)
        filteredN = sum(getfield.(ssd, :n))
        # logprob += slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta, prob)
        logprob += origN / filteredN * slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta, prob)
        N += origN
    end

    if normalize
        logprob /= N
    end

    return logprob
end

function log_jeffreys_betadist(a, b)
    d = (polygamma(1, a) - polygamma(1, a + b)) * (polygamma(1, b) - polygamma(1, a + b))
    od = polygamma(1, a + b)^2
    return 1/2 * log(d - od)
end

function logdensity(cgtree, p::ComponentArray; maxsubtree=Inf, prob=_bdihprob_inplace)
    lp = 0.0
    
    try
        if 0.001 < p.f < 1.0
            lp += logpdf(Truncated(Beta(0.5, 0.5), 0.001, 0.999), p.f)
        end
        
        if p.b > 0.0
            lp -= 1/2 * log(p.b)
        end

        if p.d > 0.0
            lp -= 1/2 * log(p.d)
        end
        
        if p.i.rho > 0.0
            lp -= 1/2 * log(p.i.rho)
            # lp += logpdf(Gamma(p.priors.b.alpha, p.priors.b.beta), p.b)
            # lp += logpdf(Gamma(p.priors.d.alpha, p.priors.d.beta), p.d)
            # lp += logpdf(Gamma(p.priors.i.rho.alpha, p.priors.i.rho.beta), p.i.rho)
            # lp += logpdf(Truncated(Beta(p.priors.i.g.alpha, p.priors.i.g.beta), 0, 0.999), p.i.g)
            lp -= 1/2 * log(p.i.g) + log(1 - p.i.g)
        end

        if p.h.eta > 0.0
            lp -= 1/2 * log(p.h.eta)
            # lp += logpdf(Gamma(p.priors.h.eta.alpha, p.priors.h.eta.beta), p.h.eta)
            # lp += logpdf(Gamma(p.priors.h.alpha.alpha, p.priors.h.alpha.beta), p.h.alpha)
            # lp += logpdf(Gamma(p.priors.h.beta.alpha, p.priors.h.beta.beta), p.h.beta)
            lp += log_jeffreys_betadist(p.h.alpha, p.h.beta)
        end

        if lp > -Inf # Don't bother if we are out of range
            lp += cgtreelogprob(cgtree, p.f, p.b, p.d, p.i.rho, p.i.g, p.h.eta, p.h.alpha, p.h.beta, prob, maxsubtree=maxsubtree)
        end

        return lp
    catch _
        return -Inf
    end

end

function initparams(;
    f=0.999, b=1.0, d=1.0, rho=0.0, g=0.5, eta=0.0, alpha=5.0, beta=2.0,
    # fpriora=0.5, fpriorb=0.5, # Bernoulli Jeffreys Beta(1/2, 1/2)
    # bpriora=1.0, bpriorb=5.0,
    # dpriora=1.0, dpriorb=5.0,
    # rhopriora=1.0, rhopriorb=5.0,
    # gpriora=0.5, gpriorb=0.01, # Approx Jeffreys Beta(1/2, 0)
    # etapriora=1.0, etapriorb=5.0, 
    # alphapriora=1.1, alphapriorb=4.0, 
    # betapriora=1.1, betapriorb=4.0
    )
    return ComponentArray(f=f, b=b, d=d, i=(rho=rho, g=g), h=(eta=eta, alpha=alpha, beta=beta))
end