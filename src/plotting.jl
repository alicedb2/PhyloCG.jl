function plotssd!(ax, ssd, params=nothing)
    (t, s), ssd = ssd
    mass = sum(getindex.(ssd, 2))
    ks = getindex.(ssd, 1)
    maxk = _powerof2ceil(maximum(ks))
    logps = exp.(log.(getindex.(ssd, 2)) .- log(mass))
    logps = reverse(cumsum(reverse(logps)))
    barplot!(ax, ks .+ 0.5, logps, fillto=1/2/mass, gap=0.0, alpha=0.4, width=Makie.automatic);

    if params !== nothing
        _logphis = logphis(maxk, t, s, params.cgmodel...)[2:end];
        # map_logphis = logphis(maxn, t, s, map_est...)[2:end];
        _logps = exp.(_logphis)
        _logps = reverse(cumsum(reverse(_logps)))
        stairs!(ax, 1:length(_logphis) .+ 0.5, _logps, step=:post, color=Cycled(2), linewidth=3);
    end

    xlims!(ax, 1, maxk);
    ylims!(ax, 1/2/mass, 1);

    return ax
end

function plot(chain; burn=0)
    with_theme(theme_minimal()) do
        nbparams = sum(chain.sampler.mask[:])
        fig = Figure(size=(1600, 400 * (1 + nbparams)), fontsize=30);

        _labels = replace.(labels(chain.sampler.params), Ref("cgmodel." => ""))

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