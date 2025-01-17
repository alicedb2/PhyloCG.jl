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

function generate_ssd(crownsize, modelssd; condition_on_size=false, rng=default_rng())

    @assert length(modelssd) >= crownsize
    if length(modelssd) < 2 * crownsize
        @warn "Model SSD may be too small for crown size, ideally it should be at least twice the crown size"
    end

    ssd = DefaultDict{Int, Int, Int}(0)
    remaining = crownsize
    while remaining > 0
        _modelssd = modelssd[1:remaining]
        _modelssd .-= logsumexp(_modelssd)
        k = rand(rng, Categorical(exp.(_modelssd)))
        remaining -= k
        ssd[k] += 1
    end
    if condition_on_size
        G = Gstatistic(ssd, modelssd[1:crownsize] .- logsumexp(modelssd[1:crownsize]))
    else
        G = Gstatistic(ssd, modelssd)
    end
    return (; ssd, G)
end

function generate_cgtree(modelssds; treesize=1000, verbose=false, rng=default_rng())

    modelssds = sort(modelssds, by=x->x.t)

    cgtree = CGTree()

    slice_Gs = Dict{@NamedTuple{t::Float64, s::Float64}, Float64}()
    G = 0.0
    crownsize = 0
    for (i, (ts, ssd)) in enumerate(modelssds)
        t, s = ts
        if i == 1
            crownsize = treesize
        end
        verbose && println("Sampling slice $i t=$(round(t, digits=4))  s=$(round(s, digits=4)) crown size=$crownsize")
        cgtree[(; t, s)], ssdG = generate_ssd(crownsize, ssd, rng=rng)
        crownsize = sum(values(cgtree[(; t, s)]))
        slice_Gs[(; t, s)] = ssdG
        G += ssdG
    end

    return (; cgtree, G, slice_Gs)

end

function generate_cgtree(f, b, d, rho, g, eta, alpha, beta; nbslices=8, age=1.0, treesize=1000, verbose=false, rng=default_rng())

    K = 2 * treesize
    ts = LinRange(0.0, age, nbslices + 1)

    slice_Gs = Dict{@NamedTuple{t::Float64, s::Float64}, Float64}()
    modelssds = ModelSSDs()
    cgtree = CGTree()

    G = 0.0
    crownsize = 0
    for (i, (s, t)) in enumerate(zip(ts[1:end-1], ts[2:end]))
        if i == 1
            crownsize = treesize
        end
        verbose && println("Calculating slice $i t=$(round(t, digits=4))  s=$(round(s, digits=4)) crownsize=$crownsize...sampling")
        modelssds[(; t, s)] = logphis(K, t, s, f, b, d, rho, g, eta, alpha, beta)
        cgtree[(; t, s)], ssdG = generate_ssd(crownsize, modelssds[(; t, s)], rng=rng)
        crownsize = sum(values(cgtree[(; t, s)]))
        K = 2 * crownsize
        slice_Gs[(; t, s)] = ssdG
        G += ssdG
    end

    return (; cgtree, G, slice_G)
end