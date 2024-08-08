function plotssd!(ax, ssd, params=nothing; cumulative=false)
    (t, s), ssd = ssd
    mass = sum(getfield.(ssd, :n))
    ks = getfield.(ssd, :k)
    maxk = _powerof2ceil(maximum(ks))
    logps = exp.(log.(getfield.(ssd, :n)) .- log(mass))
    if cumulative
        logps = reverse(cumsum(reverse(logps)))
    end
    barplot!(ax, ks, logps, fillto=1/2/mass, gap=0.0, alpha=0.4, width=1)

    if params !== nothing
        _logphis = logphis(maxk, t, s, params...)[2:end];
        _logps = exp.(_logphis)
        if cumulative
            _logps = 1 .- vcat(0, cumsum(_logps))[1:end-1]
            # _logps = reverse(cumsum(reverse(_logps)))
        end
        stairs!(ax, 1:length(_logphis), _logps, step=:center, color=Cycled(2), linewidth=5);
    end

    xlims!(ax, 1, maxk);
    ylims!(ax, 1/2/mass, 1.5);

    return ax
end

function plotssd(ssd, params=nothing; cumulative=false)
    with_theme(theme_minimal()) do
        fig = Figure(size=(800, 600), fontsize=30);
        ax = Axis(fig[1, 1], 
        title="subtree size distribution ($(round(ssd[1][1], digits=3)), $(round(ssd[1][2], digits=3)))", 
        xlabel="subtree size", ylabel="probability",
        xscale=log2, yscale=log);
        plotssd!(ax, ssd, params, cumulative=cumulative);
        return fig
    end
end

function plotssds(ssds, params=nothing; cumulative=false)
    with_theme(theme_minimal()) do
        ssds = sort(ssds, by=x->x[1])
        nbslices = length(ssds)
        fig = Figure(size=(1800, 600 * div(nbslices+1, 2)), fontsize=30);
        for (i, ((t, s), ssd)) in enumerate(ssds)
            println("$i t=$(round(t, digits=4))  s=$(round(s, digits=4))")
            ax = Axis(fig[div(i-1, 2)+1, mod1(i, 2)], 
            title="subtree size distribution ($(round(t, digits=3)), $(round(s, digits=3)))", 
            xlabel="subtree size", ylabel="probability",
            xscale=log2, yscale=log);
            plotssd!(ax, ((t, s), ssd), params, cumulative=cumulative);
        end
        return fig
    end
end

function plot(chain; burn=0)
    with_theme(theme_minimal()) do
        nbparams = sum(chain.sampler.mask[:])
        fig = Figure(size=(1600, 400 * (1 + nbparams)), fontsize=30);

        _labels = replace.(labels(chain.sampler.params), Ref("" => ""))

        axtrace = Axis(fig[1, 1], title="log density", xlabel="iteration", ylabel="log density");
        _samples = chainsamples(chain, :logdensity, burn=burn)       
        lines!(axtrace, _samples, color=Cycled(1), linewidth=2);
        axmarginal = Axis(fig[1, 2], title="log density", xlabel="log density", ylabel="probability");
        hist!(axmarginal, _samples, bins=50, color=Cycled(1), normalization=:pdf);

        for (i, k) in enumerate(findall(chain.sampler.mask[:]))
            axtrace = Axis(fig[i+1, 1], title=_labels[k], xlabel="iteration", ylabel=last(split(_labels[k], ".")));
            _samples = chainsamples(chain, k, burn=burn)       
            lines!(axtrace, _samples, color=Cycled(1), linewidth=2);
            axmarginal = Axis(fig[i+1, 2], title=_labels[k], xlabel=last(split(_labels[k], ".")), ylabel="probability");
            hist!(axmarginal, _samples, bins=50, color=Cycled(1), normalization=:pdf);
        end

        return fig
    end
end