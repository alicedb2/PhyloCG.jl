function init_params(; f=1.0, b=1.0, d=1.0, rho=0.0, g=0.5, eta=1.0, alpha=1.0, beta=1.0)
    return ComponentArray(f=f, b=b, d=d, rho=rho, g=g, eta=eta, alpha=alpha, beta=beta)
end

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
        # du += eta * alpha / (alpha + beta) * (u - 1) * u * hyp2f1a1(alpha + 1, alpha + beta + 1, u)
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

# function logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace; gap=1/_powerof2ceil(truncN), optimizeradius=false)::Vector{Real}
function logphis(truncN::Int, t::Real, s::Real, p::ComponentArray, prob=_bdihprob_inplace; gap=1/_powerof2ceil(truncN), optimizeradius=false)::Vector{Real}
    # Make sure the total number of state
    # is the closest power of two larger than n
    
    n = _powerof2ceil(truncN)

    if t < s || s < 0.0 || !(0 <= p.f <= 1) || p.b < 0 || p.d < 0 || p.rho < 0 || !(0 <= p.g <= 1) || p.eta < 0 || p.alpha < 0 || p.beta < 0
        return fill(-Inf, n)
    end

    if optimizeradius
        r = bdihPhi_optimal_radius(n, t, s, p.f, p.b, p.d, p.rho, p.g, p.eta, p.alpha, p.beta, prob)
    else 
        r = bdihPhi_singularity(t, s, p.f, p.b, p.d, p.rho, p.g, p.eta, p.alpha, p.beta, prob) - gap
    end

    complex_halfcircle = r * exp.(2pi * im * (0:div(n, 2)) / n)

    # try
        Phi_samples = [Phi(z, t, s, p.f, p.b, p.d, p.rho, p.g, p.eta, p.alpha, p.beta, prob) for z in complex_halfcircle]
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

function slicelogprob(subtreesizedistribution, t::Real, s::Real, p::ComponentArray, prob=_bdihprob_inplace)::Real

    truncN = maximum(first.(subtreesizedistribution)) + 1

    phiks = logphis(truncN, t, s, p, prob)[2:end]

    logprob = sum([n * phiks[k] for (k, n) in subtreesizedistribution])

    return logprob
end

function cgtreelogprob(cgtree, p::ComponentArray, prob=_bdihprob_inplace; dropfirstslice=false, normalize=false)::Real
    logprob = 0.0
    N = 0
    for ((t, s), ssd) in cgtree
        if s == 0 && dropfirstslice
            continue
        end
        N += sum(getindex.(ssd, 2))
        logprob += slicelogprob(ssd, t, s, p, prob)
    end
    if normalize
        logprob /= N
    end
    return logprob
end

# @model function BDIH(cgtree, model="fbdih", prob=_bdihprob_inplace)
    
#     f ~ Beta(1.05, 1.0)
#     b ~ Gamma(1.0, 4.0)
#     d ~ Gamma(1.0, 4.0)
#     rho ~ Gamma(1.0, 4.0)
#     g ~ Beta(1.05, 1.05)
#     eta ~ Gamma(1.0, 4.0)
#     alpha ~ Gamma(1.1, 4.0)
#     beta ~ Gamma(1.1, 4.0)

#     print(" $(round(f, digits=4)), $(round(b, digits=4)), $(round(d, digits=4)), $(round(rho, digits=4)), $(round(g, digits=4)), $(round(eta, digits=4)), $(round(alpha, digits=4)), $(round(beta, digits=4))")

#     datalikelihood = cgtreelogprob(cgtree, 
#             contains(model, 'f') ? f : 1.0,
#             contains(model, 'b') ? b : 0.0,
#             contains(model, 'd') ? d : 0.0, 
#             contains(model, 'i') ? rho : 0.0,
#             contains(model, 'i') ? g : 0.5, 
#             contains(model, 'h') ? eta : 0.0, 
#             contains(model, 'h') ? alpha : 1.0,
#             contains(model, 'h') ? beta : 1.0,
#             prob
#         )
#     println(" $(round(datalikelihood, digits=4))")
#     DynamicPPL.@addlogprob! datalikelihood
#     return (; f, b, d, rho, g, eta, alpha, beta, datalikelihood)

# end

# @model function BDIH_gaussian(cgtree, model="fbdih", prob=_bdihprob_inplace; bounds=true)
    
#     # logitf ~ Flat()
#     # logb ~ Flat()
#     # logd ~ Flat()
#     # logrho ~ Flat()
#     # logitg ~ Flat()
#     # logeta ~ Flat()
#     # logalpha ~ Flat()
#     # logbeta ~ Flat()

#     # logitf ~ Normal(0, 1.7)
#     # logb ~ Normal(log(5), 2)
#     # logd ~ Normal(log(5), 2)
#     # logrho ~ Normal(log(5), 2)
#     # logitg ~ Normal(0, 1.7)
#     # logeta ~ Normal(log(5), 2)
#     # logalpha ~ Normal(log(5), 2)
#     # logbeta ~ Normal(log(5), 2)
    
#     params ~ MvNormal([0, 0, 0, 0, 0, 0, 0, 0], 
#                       diagm([2.6, 2, 2, 2, 1.7, 2, 2, 2]))

#     f = logistic(params[1])
#     b = exp(params[2])
#     d = exp(params[3])
#     rho = exp(params[4])
#     g = logistic(params[5])
#     eta = exp(params[6])
#     alpha = exp(params[7])
#     beta = exp(params[8])
    
#     # Jacobian from changes of variables
#     @addlogprob! -log(abs(f)) - log(abs(1 - f))
#     @addlogprob! -log(abs(b))
#     @addlogprob! -log(abs(d))
#     @addlogprob! -log(abs(rho)) - log(abs(g)) - log(abs(1 - g))
#     @addlogprob! -log(abs(eta)) - log(abs(alpha)) - log(abs(beta))

#     print(" $(round(f, digits=4)), $(round(b, digits=4)), $(round(d, digits=4)), $(round(rho, digits=4)), $(round(g, digits=4)), $(round(eta, digits=4)), $(round(alpha, digits=4)), $(round(beta, digits=4))")
    
#     if bounds && (!(0.001 <= f <= 1) || !(0.0 <= b <= 10) || !(0.0 <= d <= 10) || !(0.0 <= rho <= 10) || !(0.001 <= g <= 0.999) || !(0.0 <= eta <= 10) || !(0.001 <= alpha <= 16) || !(0.001 <= beta <= 3))
#         println(" -Inf")
#         datalikelihood = -Inf
#         DynamicPPL.@addlogprob! datalikelihood
#         return (; f, b, d, rho, g, eta, alpha, beta, datalikelihood)
#     else
#         try
#             datalikelihood = cgtreelogprob(cgtree, 
#                     contains(model, 'f') ? f : 1.0,
#                     contains(model, 'b') ? b : 0.0,
#                     contains(model, 'd') ? d : 0.0, 
#                     contains(model, 'i') ? rho : 0.0,
#                     contains(model, 'i') ? g : 0.5, 
#                     contains(model, 'h') ? eta : 0.0, 
#                     contains(model, 'h') ? alpha : 1.0,
#                     contains(model, 'h') ? beta : 1.0,
#                     prob
#                 )
#             println(" $(round(datalikelihood, digits=4))")
#             @addlogprob! datalikelihood
#             return (; f, b, d, rho, g, eta, alpha, beta, datalikelihood)
#         catch e
#             println(" -Inf")
#             datalikelihood = -Inf
#             @addlogprob! datalikelihood
#             return (; f, b, d, rho, g, eta, alpha, beta, datalikelihood=Inf)
#         end
#     end

# end