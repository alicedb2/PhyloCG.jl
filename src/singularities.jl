b_pole(t, b)::Real = exp(b * t) / (exp(b * t) - 1)

function bd_pole(t, b, d)::Real
    if b == d
        return 1 + 1 / (b * t)
    else
        return (exp((b - d) * t) - d / b) / (exp((b - d) * t) - 1)
    end
end

function i_branchpoint(t, rho, g)::Real
    c = g^g*(1-g)^(1-g)*exp(-rho*g*t)
    return find_zero(z -> abs(z-1)^(1-g)/z - c, (1, 1/g), Bisection())
end

function di_branchpoint(t, d, rho, g)::Real
    c = (1/g-1)^(1-g)/(1/g-d/g/(rho+d))^(rho/(rho+d))
    c *= exp(-(rho*g - d*(1-g))*t)
    lower_bound = max(1, d/g/(rho+d))
    function fun(z)
        ret = (z-1)^(1-g)/abs(z-d/g/(rho+d))^(rho/(rho+d))
        return isfinite(ret) ? ret : Inf
    end
    return find_zero(z -> fun(z) - c, (lower_bound, 1/g), Bisection())
end

function bi_branchpoint(t, b, rho, g)::Real
    a0 = -(rho * g + b*(1 - g)) / (rho * g + b)
    a2 = rho * g^2 / (rho * g + b)
    b2 = (rho/b + 1/g)

    c = (1/g)^a0 * (1/g - 1)^(1-g) * abs(1/g - b2)^a2
    c *= exp(-(rho * g + b*(1 - g))*t)

    return find_zero(z -> abs(z)^a0 * abs(z - 1)^(1-g) * abs(z - b2)^a2 - c, (1, 1/g), Bisection())
end


function _bdi_fun(z, p)::Real
    c, a0, a1, a2, Δ1, Δ2 = p
    ret = abs(z - 1)^a0 * abs(z - Δ1)^a1 * abs(z - Δ2)^a2
    isfinite(ret) ? ret : Inf
    return ret - c
end

function bdi_branchpoint(t, b, d, rho, g)::Real
    A = 1 - g
    B = 1 - 1/g - rho/b
    Δ = sqrt((rho/b + 1/g + d/b)^2 - 4*d/b/g)
    Δ1 = 1/2*(rho/b + 1/g + d/b + Δ)
    Δ2 = 1/2*(rho/b + 1/g + d/b - Δ)
    a0 = 1 - g
    a1 = -(A * Δ1 + B) / Δ
    a2 = (A * Δ2 + B) / Δ
    r = rho * g + (b - d)*(1 - g)

    # z = rand()
    # println(1 / (b*(z - 1)*z + d*(1 - z) + rho*g*(z - 1)*z/(1 - g*z)))
    # println(1/r * (a0/(z - 1) + a1/(z - Δ1) + a2/(z - Δ2)))

    c = (1/g - 1)^a0 * abs(1/g - Δ1)^a1 * abs(1/g - Δ2)^a2 * exp(-r*t)

    lower_bound = maximum([1, Δ2])
    # println("$lower_bound, $(1/g)")
    # println("$(round(Δ2, digits=4))")
    branchpoint = find_zero(_bdi_fun, (lower_bound, 1/g), Bisection(), (c, a0, a1, a2, Δ1, Δ2))
    # branchpoint = find_zero(z -> fun(z) - c, (lower_bound, 1/g), Bisection())
    return branchpoint
end

function bdih_singularity(t, b, d, rho, g, eta)::Real
    if eta == 0
        if b > 0 && d == 0 && rho == 0
            # println("B model")
            return b_pole(t, b)
        end
        if b == 0 && d > 0 && rho == 0
            # println("D model")
            return Inf
        end
        if b > 0 && d > 0 && rho == 0
            # println("BD model")
            return bd_pole(t, b, d)
        end
        if b == 0 && d == 0 && rho > 0
            # println("I model")
            return i_branchpoint(t, rho, g)
        end
        if b > 0 && d == 0 && rho > 0
            # println("BI model")
            return bi_branchpoint(t, b, rho, g)
        end
        if b == 0 && d > 0 && rho > 0
            # println("DI model")
            return di_branchpoint(t, d, rho, g)
        end
        # println("BDI model")
        return bdi_branchpoint(t, b, d, rho, g)
    else
        # Any model with heterogeneous innovation rate
        return 1.0
    end
end

function bdihPhi_singularity(t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)
    Usf = Ubdih(1 - f, s, b, d, rho, g, eta, alpha, beta, prob)
    return (bdih_singularity(t - s, b, d, rho, g, eta) - Usf) / (1 - Usf)
end

function _saddlepoint_cond(n, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)
    complex_step = 1e-13
    function condition(r)::Float64
        u = Phi(r + complex_step * im, t, s, f, b, d, rho, g, eta, alpha, beta, prob)
        Phir = real(u)
        dPhir = imag(u)/complex_step
        return dPhir/n - Phir/r
        # return n/dPhir - r/Phir
        # return dPhir/abs(Phir) - (n+1)/abs(r)
    end
    return condition
end

function bdihPhi_optimal_radius(n, t, s, f, b, d, rho, g, eta, alpha, beta, prob=_bdihprob_inplace)
    max_radius = bdihPhi_singularity(t, s, f, b, d, rho, g, eta, alpha, beta, prob)
    return find_zero(_saddlepoint_cond(n, t, s, f, b, d, rho, g, eta, alpha, beta, prob), (0.0, max_radius), Bisection())
end
