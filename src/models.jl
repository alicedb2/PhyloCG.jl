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

function _bdih_jac!(J, u, p, t)
    # println("in jac [J]=$(size(J)) [u]=$(size(u))")
    b, d, rho, g, eta, alpha, beta = p
    _u = u[1] + im * u[2]
    _J = 0.0im
    if b > 0
        _J += b * (2 * _u - 1)
    end
    if d > 0
        _J += -d
    end
    if rho > 0
        _J += rho * ((1 - g) / (1 - g * _u)^2 - 1)
    end
    if eta > 0
        _J += eta * alpha / (alpha + beta) * ((2 * _u - 1) * _₂F₁(1, alpha + 1, alpha + beta + 1, _u) + (alpha + 1) / (alpha + beta + 1) * (_u - 1) * _u * _₂F₁(2, alpha + 2, alpha + beta + 2, _u))
    end
    J[1, 1] = real(_J)
    J[2, 2] = real(_J)
    J[1, 2] = -imag(_J)
    J[2, 1] = imag(_J)
    return nothing
end

const _bdih_fun = ODEFunction(_bdih!)
const _bdih_prob_flat = ODEProblem(_bdih_fun, zeros(2), (0.0, 1.0))

const _bdih_fun_wjac = ODEFunction(_bdih!, jac=_bdih_jac!)
const _bdih_prob_flat_wjac = ODEProblem(_bdih_fun_wjac, zeros(2), (0.0, 1.0))

function Ubdih(z, t::Float64, b, d, rho, g, eta, alpha, beta)::ComplexF64
    if t == 0.0
        return z
    end
    u, = Ubdih(z, [t], b, d, rho, g, eta, alpha, beta)
    return u
end

function Ubdih(z, ts::Vector{Float64}, b, d, rho, g, eta, alpha, beta)
    sol = solve(
        _bdih_prob_flat_wjac,
        AutoTsit5(Rodas5P()),
        u0=[real(z), imag(z)], tspan=(0.0, maximum(ts)), 
        p=[b, d, rho, g, eta, alpha, beta],
        saveat=ts,
        reltol=1e-6)
    return [x[1] + im * x[2] for x in sol.u]
end

function Phi(y, t, s, f, b, d, rho, g, eta, alpha, beta)
    @assert t >= s >= 0
    Us1f, Ut1f = Ubdih(1 - f, [s, t], b, d, rho, g, eta, alpha, beta)
    return (Ubdih(Us1f .+ y .* (1 - Us1f), t - s, b, d, rho, g, eta, alpha, beta) .- Ut1f) ./ (1 - Ut1f)
end

function _powerof2ceil(n)
    return 2^ceil(Int, log2(n))
end

# function logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta; gap=1/_powerof2ceil(truncN), optimizeradius=false)
function logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta; gap=1/2/truncN, optimizeradius=false)
    
    # # Make sure the total number of state
    # # is the closest power of two larger than n
    # n = _powerof2ceil(truncN)
    # # If the largest empirical k is too
    # # close to the next power of two, double n
    # # to avoid the numerical instabilities
    # # that sometimes appear in the tail of
    # # logphis
    # if truncN/n >= 0.75
    #     n *= 2
    # end

    N = 2 * truncN

    if t < s || s < 0.0 || !(0 < f <= 1) || b < 0 || d < 0 || rho < 0 || !(0 < g < 1) || eta < 0 || alpha <= 0 || beta <= 0
        return fill(-Inf, N)
    end

    if optimizeradius
        r = bdihPhi_optimal_radius(N, t, s, f, b, d, rho, g, eta, alpha, beta)
    else 
        r = bdihPhi_singularity(t, s, f, b, d, rho, g, eta, alpha, beta) - gap
    end

    complex_halfcircle = r * exp.(2pi * im * (0:div(N, 2)) / N)

    Us1f, Ut1f = Ubdih(1 - f, [s, t], b, d, rho, g, eta, alpha, beta)
    Phi_samples = [(Ubdih(Us1f .+ z .* (1 - Us1f), t - s, b, d, rho, g, eta, alpha, beta) .- Ut1f) ./ (1 - Ut1f) for z in complex_halfcircle]

    upks = irfft(conj(Phi_samples), N) # Hermitian FFT
    log_pks = [upk > 0 ? log(upk) : -Inf for upk in upks] - (0:N-1) .* log(r)

    return log_pks
end

function slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta; maxsubtree=Inf)

    ssd = filter(st -> st.k <= maxsubtree, ssd)

    truncN = maximum(getfield.(ssd, :k)) + 1

    _logphiks = logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta)[2:end]

    logprob = sum([n * _logphiks[k] for (k, n) in ssd])

    return logprob
end

function cgtreelogprob(cgtree, f, b, d, rho, g, eta, alpha, beta; dropfirstslice=false, normalize=false, maxsubtree=Inf)
    logprob = 0.0
    nbsubtrees = 0
    for ((t, s), ssd) in cgtree
        if s == 0.0 && dropfirstslice
            continue
        end
        nbsubtrees += sum(getfield.(ssd, :n))
        logprob += slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta, maxsubtree=maxsubtree)
    end

    if normalize
        logprob /= nbsubtrees
    end

    return logprob
end

function log_jeffreys_betadist(a, b)
    if a <= 0.0 || b <= 0.0
        return -Inf
    end
    d = (polygamma(1, a) - polygamma(1, a + b)) * (polygamma(1, b) - polygamma(1, a + b))
    offd = polygamma(1, a + b)^2
    return 1/2 * log(d - offd)
end

# Jeffreys prior on Poisson rate π(r) ∝ 1/sqrt(r),
# We truncate at rate_upper_bound to both avoid
# numerical instability and to make the prior proper
function log_jeffreys_rate(r; rate_upper_bound=50.0)
    if 0.0 < r <= rate_upper_bound
        return -1/2 * log(r) - log(2) - 1/2 * log(rate_upper_bound)
    else
        return -Inf
    end
end

# Innovation process burst size distribution
# is geometric P(k) = (1 - g)g^(k-1), k ≥ 1
# Jeffrets prior on geometric g (improper)
# given by π(g) ∝ 1/sqrt(g) * 1/(1-g)
function log_jeffreys_geom(g)
    if 0.0 < g < 1.0
        return -1/2 * log(g) - log(1 - g)
    else
        return -Inf
    end
end

function log_jeffreys_samplingrate(f; f_lower_bound=0.01)
    return logpdf(Truncated(Beta(0.5, 0.5), f_lower_bound, 1.0), f)
end

function logdensity(cgtree, p::ComponentArray; maxsubtree=Inf)
    lp = 0.0

    try
        # f = 1.0 signals the exclusion
        # of the sampling rate as a parameter
        # of the model
        if p.f < 1.0
            lp += log_jeffreys_samplingrate(p.f)
        end
        
        # Rates b, d, rho, eta = 0.0 signal
        # the exclusion of the corresponding
        # process in the model

        if p.b != 0.0
            lp += log_jeffreys_rate(p.b)
        end

        if p.d != 0.0
            lp += log_jeffreys_rate(p.d)
        end
        
        if p.i.rho != 0.0
            lp += log_jeffreys_rate(p.i.rho)
            lp += log_jeffreys_geom(p.i.g)
        end
        
        if p.h.eta != 0.0
            lp += log_jeffreys_rate(p.h.eta)
            lp += log_jeffreys_betadist(p.h.alpha, p.h.beta)
        end

        # Don't bother if one of the above
        # parameter was out of bounds
        if isfinite(lp)
            lp += cgtreelogprob(cgtree, p.f, p.b, p.d, p.i.rho, p.i.g, p.h.eta, p.h.alpha, p.h.beta, maxsubtree=maxsubtree)
        end

        return lp
    catch e
        return -Inf
    end

end

# By default this is a full fbdih model
function initparams(;
    f=0.999, b=1.0, d=1.0, rho=1.0, g=0.5, eta=1.0, alpha=5.0, beta=2.0,
    )
    return ComponentArray(f=f, b=b, d=d, i=(rho=rho, g=g), h=(eta=eta, alpha=alpha, beta=beta))
end

_uvw(eta, alpha, beta) = [1/2 * log(eta / alpha), sqrt(eta * alpha), beta / sqrt(eta * alpha)]
_eab(u, v, w) = [v * exp(u), v * exp(-u), v * w ]
function logjac_deabduvw(eta, alpha, beta)
    try
        return log(2) + log(eta) + log(alpha)
    catch _
        return -Inf
    end
end