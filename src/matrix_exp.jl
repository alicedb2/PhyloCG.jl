function Lbdih(b, d, rho, g, eta, alpha, beta; truncN=1024, leaky=true)
    L = zeros(truncN, truncN)

    for n in 0:truncN-2
        L[n+1, n+1] -= b * n
        L[n+2, n+1] += b * n
    end
    L[truncN, truncN] += -b * (truncN - 1)

    for n in 1:truncN-1
        L[n, n+1] += d * n
        L[n+1, n+1] -= d * n
    end

    if rho > 0 && 0 < g < 1
        for n in 1:truncN-2
            for m in n+1:truncN-1
                L[m+1, n+1] += rho * n * (1 - g) * g^(m - n)
            end
        end
        for n in 1:truncN-1
            L[n+1, n+1] -= rho * g * n
        end
    end

    if eta > 0 && alpha > 0 && beta > 0
        for n in 1:truncN-2
            for m in n+1:truncN-1
                L[m+1, n+1] += eta * n * beta * exp(loggamma(alpha + beta) - loggamma(alpha) + loggamma(alpha + m - n) - loggamma(alpha + beta + m - n + 1))
            end
        end
        for n in 1:truncN-1
            L[n+1, n+1] -= eta * alpha / (alpha + beta) * n
        end
    end

    if !leaky
        L[diagind(L)] .= 0.0
        L[diagind(L)] .= -sum(L, dims=1)[1, :]
    end

    return L
end

function logphis_exp(truncN::Integer, t::Real, s::Real, f::Real=1, b::Real=0, d::Real=0, rho::Real=0, g::Real=0, eta::Real=0, alpha::Real=0, beta::Real=0; leaky=true)
    L = Lbdih(b, d, rho, g, eta, alpha, beta, truncN=truncN, leaky=leaky)
    Ps = exp(s * L)[:, 2]
    Pt = exp(t * L)[:, 2]
    Pts = exp((t - s) * L)[:, 2]

    logbinomial(n, k) = loggamma(n + 1) - loggamma(n - k + 1) - loggamma(k + 1)

    missing_prob_s = sum(Ps .* (1 - f).^(0:truncN-1))
    missing_prob_t = sum(Pt .* (1 - f).^(0:truncN-1))

    log_survival_prob_s = log1p(-missing_prob_s)
    log_survival_prob_t = log1p(-missing_prob_t)

    uncond_logphis = fill(-Inf, truncN, truncN)
    for k in 0:truncN-1
        for n in 0:truncN-1-k
            uncond_logphis[k+1, n+1] = log(Pts[n + k + 1]) + logbinomial(n + k, k) + n * log(missing_prob_s)
        end
    end

    uncond_logphis = logsumexp(uncond_logphis, dims=2)
    # println(uncond_logphis)
    uncond_logphis .+= (0:truncN-1) * log_survival_prob_s 

    cond_logphis = uncond_logphis .- log_survival_prob_t
    cond_logphis[1] = -Inf

    return cond_logphis
end