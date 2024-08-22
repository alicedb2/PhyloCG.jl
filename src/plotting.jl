function plotssd!(ax, ssd, params=nothing; cumulative=false, modelN=Inf)
    (t, s), ssd = ssd
    mass = sum(getfield.(ssd, :n))
    ks = getfield.(ssd, :k)
    max_empirical_k = maximum(ks)

    ps = exp.(log.(getfield.(ssd, :n)) .- log(mass))
    if cumulative
        # ps = 1 .- vcat(0, cumsum(ps))[1:end-1]
        ps = reverse(cumsum(reverse(ps)))
    end
    barplot!(ax, ks, ps, fillto=1/2/mass, gap=0.0, alpha=0.4, width=1)

    if params !== nothing
        if !isfinite(modelN)
            truncN = 2 * (max_empirical_k + 1)
        else
            truncN = 2 * modelN
        end
        _logphis = logphis(truncN, t, s, params...)[2:end];
        # _logphis = logphis_exp(truncN, t, s, params...)[2:end];
        if cumulative
            _logcumulsumexp = reduce((x, y) -> vcat(x, logaddexp(x[end], y)), _logphis, init=[-Inf])
            _logphis = log1mexp.(_logcumulsumexp)
            # _ps = 1 .- vcat(0, cumsum(_ps))[1:end-1]
            # _ps = reverse(cumsum(reåverse(_ps)))
        end
        _ps = exp.(_logphis)
        stairs!(ax, 1:length(_ps), _ps, step=:center, color=Cycled(2), linewidth=5);
        ylims!(ax, minimum(filter(x-> x > 0, _ps))/4, ℯ)
        # ylims!(ax, 1/2/mass, ℯ);
    else
        xlims!(ax, 1, 2 * max_empirical_k)
    end
    ylims!(ax, exp(-20), exp(1))
    # xlims!(ax, 1, 2 * maxsubtree)
    
    return ax
end

function plotssd(ssd, params=nothing; cumulative=false, modelN=Inf)
    with_theme(theme_minimal()) do
        fig = Figure(size=(800, 600), fontsize=30);
        ax = Axis(fig[1, 1], 
        title="subtree size distribution ($(round(ssd[1][1], digits=3)), $(round(ssd[1][2], digits=3)))", 
        xlabel="subtree size", ylabel="probability",
        xscale=log2, yscale=log);
        plotssd!(ax, ssd, params, cumulative=cumulative, modelN=modelN);
        return fig
    end
end

function plotssds(ssds, params=nothing; cumulative=false, modelN=Inf)
    with_theme(theme_minimal()) do
        ssds = sort(ssds, by=x->x[1])
        nbslices = length(ssds)
        fig = Figure(size=(1800, 600 * div(nbslices+1, 2)), fontsize=30);
        for (i, ((t, s), ssd)) in enumerate(ssds)
            println("$i t=$(round(t, digits=4))  s=$(round(s, digits=4))  maxk=$(maximum(getfield.(ssd, :k)))")
            ax = Axis(fig[div(i-1, 2)+1, mod1(i, 2)], 
            title="subtree size distribution ($(round(t, digits=3)), $(round(s, digits=3)))", 
            xlabel="subtree size", ylabel="probability",
            xscale=log2, yscale=log);
            plotssd!(ax, ((t, s), ssd), params, cumulative=cumulative, modelN=modelN);
        end
        return fig
    end
end

function plot(chain; burn=0)
    with_theme(theme_minimal()) do
        nbparams = sum(chain.sampler.mask[:])
        fig = Figure(size=(1600, 400 * (1 + nbparams)), fontsize=30);

        _labels = labels(chain.sampler.params)

        axtrace = Axis(fig[1, 1], title="log density", xlabel="iteration", ylabel="log density");
        _samples = chainsamples(chain, :logdensity, burn=burn)       
        lines!(axtrace, _samples, color=Cycled(1), linewidth=2);
        axmarginal = Axis(fig[1, 2], title="log density", xlabel="log density", ylabel="probability");
        hist!(axmarginal, _samples, bins=50, color=Cycled(1), normalization=:pdf);

        for (i, k) in enumerate(findall(chain.sampler.mask[:]))
            axtrace = Axis(fig[i+1, 1], xlabel="iteration", ylabel=_labels[k]);
            _samples = chainsamples(chain, k, burn=burn)       
            lines!(axtrace, _samples, color=Cycled(1), linewidth=2);
            axmarginal = Axis(fig[i+1, 2], xlabel=_labels[k], ylabel="probability");
            hist!(axmarginal, _samples, bins=50, color=Cycled(1), normalization=:pdf);
        end

        return fig
    end
end