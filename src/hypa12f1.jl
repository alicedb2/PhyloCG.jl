# This representation will not work
# if c is an integer away from b.

function hyp2f1a1(b, c, z; maxiter=1000)

    EPS = 1e-16

    if z == 0; return 1; end
    if z == 1; return (c - 1) / (c - b - 1); end
    if c == 1; return (1 - z)^-b; end

    tn = 1
    rhon = 0

    n = 1

    res, lres, llres = tn, 0, 0

    while true
        n += 1

        if mod(n, 2) == 1 # Odd n
            k = (n - 1)/2
            an = k * (c - 1 - b + k) * z / (c + n - 3) / (c + n - 2)
        else # Even n
            k = (n - 2)/2
            an = (c - 1 + k) * (b + k) * z / (c + n - 3) / (c + n - 2)
        end

        rhon = an * (1 + rhon) / (1 - an * (1 + rhon))
        tn = rhon * tn

        res, lres, llres = res + tn, res, lres

        if ((abs(tn) < EPS * abs(res))
            || ((maxiter <= 0) && (n >= -maxiter)))
            # negative maxiter prevent switching to the analytically continued expression
            # println("n=$n")
            return res
        elseif (maxiter > 0) && (n >= maxiter)
            # println("maxiter reached")
            # If we hit maxiter and maxiter is positive, then switch to
            # the analytically continued expression with z->1-z
            return continued_hyp2f1a1(b, c, z, maxiter=maxiter)
        end
    end
end


function continued_hyp2f1a1(b, c, z; maxiter=1000)

    logACF = log(hyp2f1a1(b, 2 + b - c, 1 - z, maxiter=-maxiter))
    lgcb1, lgcb1sgn = logabsgamma(c - b - 1)
    lgc1, lgc1sgn = logabsgamma(c - 1)
    lgcb, lgcbsgn = logabsgamma(c - b)

    logf1 = logACF + lgcb1 - lgc1 - lgcb
    s1 = lgcb1sgn * lgc1sgn * lgcbsgn
    # logf1 = logACF + loggamma(c - b - 1.0) - loggamma(c - 1.0) - loggamma(c - b)
    # s1 = gammasgn(c - b - 1.0) * gammasgn(c - 1.0) * gammasgn(c - b)

    lg1bc, lg1bcsgn = logabsgamma(1 + b - c)
    lgb, lgbsgn = logabsgamma(b)

    logf2 = lg1bc - lgb + (c - b - 1) * log(1 - z) + (1 - c) * log(z)
    s2 = lg1bcsgn * lgbsgn
    # logf2 = loggamma(1.0 + b - c) - loggamma(b) + (c - b - 1.0)*log(1 - z) + (1.0 - c)*log(z)
    # s2 = gammasgn(1.0 + b - c) * gammasgn(b)

    if real(logf1) >= real(logf2)
        logf = logf1 + log(s1 + s2 * exp(logf2 - logf1))
    else
        logf = logf2 + log(s1 * exp(logf1 - logf2) + s2)
    end

    lgc, lgcsgn = logabsgamma(c)

    return lgcsgn * exp(lgc + logf)
    # return gammasgn(c) * exp(loggamma(c) + logf)
end