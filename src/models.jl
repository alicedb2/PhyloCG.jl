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
        du[1] += eta * alpha / (alpha + beta) * (u[1] - 1) * u[1] * _₂F₁(1, alpha + 1, alpha + beta + 1, u[1])
        # du[1] += eta * alpha / (alpha + beta) * (u[1] - 1) * u[1] * hyp2f1a1(alpha + 1, alpha + beta + 1, u[1])
    end
    return nothing
end

# Decomplexified ODE to use with stiff solvers
function _bdih!(du, u, p, t)
    b, d, rho, g, eta, alpha, beta = p
    u = u[1] + 1.0im * u[2]
    _du = 0.0im
    if b > 0
        _du += b * (u - 1) * u
    end
    if d > 0
        _du += d * (1 - u)
    end
    if rho > 0
        _du += rho * g * (u - 1) * u / (1 - g * u)
    end
    if eta > 0
        _du += eta * alpha / (alpha + beta) * (u - 1) * u * _₂F₁(1, alpha + 1, alpha + beta + 1, u)
        # du += eta * alpha / (alpha + beta) * (u - 1) * u * hyp2f1a1(alpha + 1, alpha + beta + 1, u)
    end
    du .= [real(_du), imag(_du)]
    return nothing
end

const _bdihprob = ODEProblem(bdih, 0.0im, (0.0, 1.0))

# const _bdihprob_inplace = ODEProblem(bdih!, 0.0im, (0.0, 1.0))
const _bdihprob_inplace = ODEProblem(_bdih!, [0.0, 0.0], (0.0, 1.0))

function Ubdih(z, t, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)::Complex
    if t == 0
        return z
    end
    sol = solve(prob, AutoTsit5(Rodas5P()), 
        u0=[real(z), imag(z)], tspan=(0.0, t), 
        p=[b, d, rho, g, eta, alpha, beta],
        reltol=1e-6)
        # abstol=1e-7)
    return sol.u[end][1] + 1.0im * sol.u[end][2]
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
    return 2^ceil(Int, log2(n))
end

function logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; gap=1/_powerof2ceil(truncN), optimizeradius=false)::Vector{Real}
    # Make sure the total number of state
    # is the closest power of two larger than n

    n = _powerof2ceil(truncN)

    # If the largest empirical k is too
    # close to the next power of two, double n
    # to avoid the numerical instabilities
    # that sometimes appear in the tail of
    # logphis
    if truncN/n >= 0.75
        n *= 2
    end

    if t < s || s < 0.0 || !(0 < f <= 1) || b < 0 || d < 0 || rho < 0 || !(0 < g < 1) || eta < 0 || alpha <= 0 || beta <= 0
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
        upks = irfft(conj(Phi_samples), n) # Hermitian FFT
        log_pks = [upk > 0 ? log(upk) : -Inf for upk in upks] - (0:n-1) .* log(r)
        return log_pks
    # catch e
        # return fill(-Inf, n)
    # end

end

function slicelogprob(subtreesizedistribution, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; maxsubtree=1000)

    truncN = maximum(getfield.(subtreesizedistribution, :k)) + 1

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
    offd = polygamma(1, a + b)^2
    return 1/2 * log(d - offd)
end

function logdensity(cgtree, p::ComponentArray; maxsubtree=Inf, prob=_bdihprob_inplace)
    lp = 0.0
    
    try
        if p.f < 1.0
            # Jeffreys prior on Bernoulli
            # sampling probability f
            # π(p) = Beta(1/2, 1/2)
            lp += logpdf(Truncated(Beta(0.5, 0.5), 0.01, 0.999), p.f)
            # We reserve f = 1 for when the model does
            # not include incomplete lineage sampling. 
            # We truncate the prior at the lower end
            # to avoid numerical instabilities.
        elseif p.f < 0.01
            return -Inf
        end
        
        # Jeffreys prior on Poisson birth/death
        # rates π(b) ∝ 1/sqrt(b) and π(d) ∝ 1/sqrt(d)
        if p.b > 0.0
            lp -= 1/2 * log(p.b)
        elseif p.b < 0.0
            return -Inf
        end

        if p.d > 0.0
            lp -= 1/2 * log(p.d)
        elseif p.d < 0.0
            return -Inf
        end
        
        if p.i.rho > 0.0
            # Jeffreys prior on Poisson rate 1/sqrt(rho) (improper)
            
            # Innovation process burst size distribution
            # is geometric P(k) = (1 - g)g^(k-1), k ≥ 1
            # Jeffrets prior on geometric g (improper)
            # given by π(g) ∝ 1/sqrt(g) * 1/(1-g)
            if !(0.0 < p.i.g < 1.0)
                return -Inf
            else
                lp -= 1/2 * log(p.i.rho)
                lp -= 1/2 * log(p.i.g) + log(1 - p.i.g)
            end
        elseif p.i.rho < 0.0
            return -Inf
        end

        if p.h.eta > 0.0
            if !(0.0 < p.h.alpha && 0.0 < p.h.beta)
                return -Inf
            else
                # Jeffreys prior on Poisson rate η (improper)
                lp -= 1/2 * log(p.h.eta)
                lp += log_jeffreys_betadist(p.h.alpha, p.h.beta)
            end
            # Jeffreys prior on α, β of Beta(α, β) (proper)
            # Could not find Jeffreys prior on full Beta-Geometric distribution
        elseif p.h.eta < 0.0
            return -Inf
        end

        if isfinite(lp) # Don't bother if we are out of range
            lp += cgtreelogprob(cgtree, p.f, p.b, p.d, p.i.rho, p.i.g, p.h.eta, p.h.alpha, p.h.beta, prob, maxsubtree=maxsubtree)
        else
            println("Out of prior support")
            println(p)
        end

        return lp
    catch e
        println(e)
        println("Log density failed")
        println(p)
        return -Inf
    end

end

# By default this is a full fbdih model
function initparams(;
    f=0.999, b=1.0, d=1.0, rho=1.0, g=0.5, eta=1.0, alpha=5.0, beta=2.0,
    )
    return ComponentArray(f=f, b=b, d=d, i=(rho=rho, g=g), h=(eta=eta, alpha=alpha, beta=beta))
end

_uvw(et, a, b) = [1/2 * log(et / a), sqrt(et * a), b / sqrt(et * a)]
_eab(u, v, w) = [v * exp(u), v * exp(-u), v * w ]
logjac_deabduvw(eta, alpha, beta) = log(2) + log(eta) + log(alpha)