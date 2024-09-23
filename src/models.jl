function _bdih!(du, u, p, t)
    b, d, rho, g, eta, alpha, beta = p
    u = u[1] + im * u[2]
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
        # _₂F₁ is not stable at large alpha
        # _du += eta * alpha / (alpha + beta) * (u - 1) * u * _₂F₁(1, alpha + 1, alpha + beta + 1, u)
        _du += eta * alpha / (alpha + beta) * (u - 1) * u * hyp2f1a1(alpha + 1, alpha + beta + 1, u)
    end
    du .= [real(_du), imag(_du)]
    return nothing
end

function _bdih(u, p, t)
    ret = similar(u)
    _bdih!(ret, u, p, t)
    return ret
end

# Abandoning the use of _₂F₁ for stability
# and we don't have a stable implementation
# for a = 2. We'd need to use one of Gauss'
# contiguous relations to express d₂F₁(1, b, c; z)/dz
# in terms of ₂F₁(1, b, c; z) and ₂F₁(1, b+1, c,; z)
# instead of ₂F₁(2, b, c; z)
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

# Jacobian needs ₂F₁(2, b, c; z), for which we
# don't have a stable implementation, do not use.
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
    if isinteger(beta)
        @warn "For numerical reasons beta cannot be an integer, adding a small perturbation"
        beta += 1e-6
    end

    # Normalize rates and times
    # Not always faster at a given tolerate
    # maxrate = maximum([b, d, rho, eta])
    # b, d, rho, eta = [b, d, rho, eta] ./ maxrate
    # ts = maxrate .* ts

    sol = solve(
        _bdih_prob_flat,
        # Tsit5(),
        AutoTsit5(Vern9()),
        u0=[real(z), imag(z)], tspan=(0.0, maximum(ts)),
        p=[b, d, rho, g, eta, alpha, beta],
        saveat=ts,
        reltol=1e-13)
    return [x[1] + im * x[2] for x in sol.u]
end

# Phi is already vectorized so we can
# avoid recomputing Us1f and Ut1f for
# each y in the halfcircle passed
# by logphis
function Phi(y, t, s, f, b, d, rho, g, eta, alpha, beta)
    @assert t >= s >= 0
    Us1f, Ut1f = Ubdih(1 - f, [s, t], b, d, rho, g, eta, alpha, beta)
    return (Ubdih.(Us1f .+ y .* (1 - Us1f), t - s, b, d, rho, g, eta, alpha, beta) .- Ut1f) ./ (1 - Ut1f)
end

function dPhi(y, t, s, f, b, d, rho, g, eta, alpha, beta)
    @assert t >= s >= 0
    Us1f, Ut1f = Ubdih(1 - f, [s, t], b, d, rho, g, eta, alpha, beta)
    z = Us1f + y * (1 - Us1f)
    Utsz = Ubdih(z, t - s, b, d, rho, g, eta, alpha, beta)
    num = ComplexF64(_bdih([real(Utsz), imag(Utsz)], [b, d, rho, g, eta, alpha, beta], 0.0)...)
    denum = ComplexF64(_bdih([real(z), imag(z)], [b, d, rho, g, eta, alpha, beta], 0.0)...)
    return num * (1 - Us1f)  / denum / (1 - Ut1f)
end

function _powerof2ceil(n)
    return 2^ceil(Int, log2(n))
end

function logphis(K, t, s, f, b, d, rho, g, eta, alpha, beta; gap=1/(K+1), optimizeradius=false)

    K += 1

    if t < s || s < 0.0 || !(0 < f <= 1) || b < 0 || d < 0 || rho < 0 || !(0 < g < 1) || eta < 0 || alpha <= 0 || beta <= 0
        return fill(-Inf, K)
    end

    if optimizeradius
        # We recommend not to use optimizeradius
        # and stick with the 1/K gap for now.
        # There's something strange with
        # xH models where the optimal radius
        # appears to not lie on or within the
        # unit circle, but the imaginary part
        # is not 0 anymore beyond the unit circle.
        # This is a problem with the current 
        # implementation which uses the complex-step
        # derivative to calculate dPhi(r)/dr.
        r = Phi_optimal_radius(K, t, s, f, b, d, rho, g, eta, alpha, beta)
    else
        r = Phi_singularity(t, s, f, b, d, rho, g, eta, alpha, beta) - gap
    end

    # We know the coefficients are probabilities
    # and therefore positive real numbers, so
    # we can safely use the Hermitian FFT.
    # Hermitian FFT only requires K/2 + 1 samples
    # in the upper half of the the complex
    # half-circle (including 0 and π) to recover
    # all K coefficients.
    complex_halfcircle = r * exp.(2pi * im * (0:div(K, 2)) / K)
    Phi_samples = Phi(complex_halfcircle, t, s, f, b, d, rho, g, eta, alpha, beta)

    # Hermitian FFT
    upks = irfft(conj(Phi_samples), K)
    log_phiks = [upk > 0 ? log(upk) : -Inf for upk in upks] - (0:K-1) .* log(r)

    # phi_0 = 0 by construction
    # and Julia's 1-based indexing
    # will make it so we can index
    # logphis directly from the data
    return log_phiks[2:end]
end

function slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta; maxsubtree=Inf)

    ssd = filter(st -> first(st) <= maxsubtree, ssd)

    truncK = 2 * maximum(keys(ssd))
    gap = 1 / truncK

    _logphiks = logphis(truncK, t, s, f, b, d, rho, g, eta, alpha, beta, gap=gap)

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
        nbsubtrees += sum(values(ssd))
        logprob += slicelogprob(ssd, t, s, f, b, d, rho, g, eta, alpha, beta, maxsubtree=maxsubtree)
    end

    if normalize
        logprob /= nbsubtrees
    end

    return logprob
end

function logphis(cgtree, f, b, d, rho, g, eta, alpha, beta; maxsubtree=Inf)
    _logphis = ModelSSDs()
    for ((t, s), ssd) in cgtree
        truncK = 2 * maximum(keys(ssd))
        truncK = Int(min(truncK, 2 * maxsubtree))
        gap = 1 / truncK
        _logphis[(t=t, s=s)] = logphis(truncK, t, s, f, b, d, rho, g, eta, alpha, beta, gap=gap)[1:div(truncK, 2)]
    end
    return _logphis
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
        lp = -1/2 * log(r)
        if isfinite(rate_upper_bound)
            lp += -log(2) - 1/2 * log(rate_upper_bound)
        end
        return lp
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

# Change of variable
# (eta, alpha, beta) -> (u, v, w)
# to remove the strong correlation between
# eta and alpha. We also need to add a third
# variable w to the bunch because with the change
# (eta, alpha, beta) -> (u, v, beta)
# v is then further strongly correlated with beta.
_uvw(eta, alpha, beta) = [1/2 * log(eta / alpha), sqrt(eta * alpha), beta / sqrt(eta * alpha)]
_eab(u, v, w) = [v * exp(u), v * exp(-u), v * w ]
function logjac_deabduvw(eta, alpha, _beta)
    try
        return log(2) + log(eta) + log(alpha)
    catch _
        return -Inf
    end
end

function log_priors(p::ComponentArray)
    lp = 0.0
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

    return lp
end

function logdensity(cgtree, p::ComponentArray; maxsubtree=Inf)
    lp = 0.0

    try
        lp += log_priors(p)

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
# Do not set beta to an integer value,
# Our implementation of ₂F₁ is not defined there.
# This is not a problem in general since the
# probability the chain will hit an integer
# is pretty virtually zero.
function initparams(;
    f=0.999, b=1.0, d=1.0, rho=1.0, g=0.5, eta=1.0, alpha=5.0, beta=2.1,
    )
    if isinteger(beta)
        @warn "For numerical reasons beta cannot be an integer, adding a small perturbation"
        beta += 1e-6
    end
    return ComponentArray(f=f, b=b, d=d, i=(rho=rho, g=g), h=(eta=eta, alpha=alpha, beta=beta))
end

function setmodel!(params::ComponentArray{Float64}, mask::ComponentArray{Bool}, model::String="fbd")

    if contains(model, "f")
        mask.f = true
        if params.f == 1.0
            params.f = initparams().f
        end
    else
        mask.f = false
        params.f = 1.0
    end

    if contains(model, "b")
        mask.b = true
        if params.b == 0.0
            params.b = initparams().b
        end
    else
        mask.b = false
        params.b = 0.0
    end

    if contains(model, "d")
        mask.d = true
        if params.d == 0.0
            params.d = initparams().d
        end
    else
        mask.d = false
        params.d = 0.0
    end

    if contains(model, "i")
        mask.i .= true
        if params.i.rho == 0.0
            params.i .= initparams().i
        end
    else
        mask.i .= false
        params.i.rho = 0.0
    end

    if contains(model, "h")
        mask.h .= true
        if params.h.eta == 0.0
            params.h .= initparams().h
        end
    else
        mask.h .= false
        params.h.eta = 0.0
    end

    return params, mask
end