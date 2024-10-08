mutable struct Pop
    t::Float64
    k::Int64
end

function advance_gillespie_bdih(tk::Pop, b, d, rho, g, eta, alpha, beta; max_age=1.0, max_size=10000)
    t, k = tk.t, tk.k
    if k == 0 || t >= max_age
        return Pop(max_age, k)
    end

    rates = [b * k, d * k, rho * k, eta * k]
    total_rate = sum(rates)
    r1 = rand()
    delta_age = -log(r1)/total_rate
    if t + delta_age >= max_age
        return Pop(max_age, k)
    end
    event = sample([1, 2, 3, 4], Weights(rates))
    if event == 1
        return Pop(t + delta_age, k + 1)
    elseif event == 2
        return Pop(t + delta_age, k - 1)
    elseif event == 3
        delta_k = rand(Geometric(1 - g))
        return Pop(t + delta_age, k + delta_k)
    elseif event == 4
        _g = rand(Beta(alpha, beta))
        delta_k = rand(Geometric(1 - _g))
        return Pop(t + delta_age, k + delta_k)
    end
end