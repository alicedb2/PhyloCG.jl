function plotssd!(ax, slice, params=nothing; cumulative=false, modelK=Inf)
    (t, s), ssd = slice

    sidx = sortperm(collect(keys(ssd)))
    ks = collect(keys(ssd))[sidx]
    ns = collect(values(ssd))[sidx]

    mass = sum(ns)
    max_empirical_k = maximum(ks)

    ps = exp.(log.(ns) .- log(mass))
    if cumulative
        # ps = 1 .- vcat(0, cumsum(ps))[1:end-1]
        ps = reverse(cumsum(reverse(ps)))
    end

    bpcolor = wong_colors()[5]
    mcolor = wong_colors()[6]

    if params !== nothing
        if params isa ComponentArray
            if !isfinite(modelK)
                modelK = max_empirical_k
            end
            truncK = 2 * modelK
            _logphis = logphis(truncK, t, s, params...)[1:modelK];
        elseif params isa ModelSSDs
            _logphis = params[(t=t, s=s)]
        else
            throw(ArgumentError("params must be a ComponentArray or a ModelSSDs"))
        end
        if cumulative
            _logcumulsumexp = reduce((x, y) -> vcat(x, logaddexp(x[end], y)), _logphis, init=[-Inf])
            _logphis = log1mexp.(_logcumulsumexp)
            # _ps = 1 .- vcat(0, cumsum(_ps))[1:end-1]
            # _ps = reverse(cumsum(reåverse(_ps)))
        end
        _ps = exp.(_logphis)
        lb = minimum(vcat(ps, _ps)/2)
        barplot!(ax, ks, ps, fillto=lb, gap=0.0, width=1.0, strokewidth=1.0, color=bpcolor, strokecolor=bpcolor);
        stairs!(ax, 1:length(_ps), _ps, step=:center, color=mcolor, linewidth=5);
        # ylims!(ax, minimum(filter(x-> x > 0, _ps))/4, ℯ)
        xlims!(ax, 1, max_empirical_k)
        ylims!(ax, lb, exp(1))
    else
        lb = minimum(ps)/2
        barplot!(ax, ks, ps, fillto=lb, gap=0.0, width=1.0, strokewidth=1.0, color=bpcolor, strokecolor=bpcolor);
        ylims!(ax, lb, ℯ);
        xlims!(ax, 1, max_empirical_k)
    end

    return ax
end

function plotssd(slice, params=nothing; cumulative=false, modelK=Inf)
    (t, s), _ = slice
    with_theme(theme_minimal()) do
        fig = Figure(size=(800, 600), fontsize=30);
        ax = Axis(fig[1, 1],
        title="SSD of slice (t=$(round(t, digits=3)), s=$(round(s, digits=3)))",
        xlabel="subtree size", ylabel="probability",
        xscale=log2, yscale=log);
        plotssd!(ax, slice, params, cumulative=cumulative, modelK=modelK);
        return fig
    end
end

function plotssds(cgtree, params=nothing; cumulative=false, modelK=Inf)
    with_theme(theme_minimal()) do
        cgtree = sort(cgtree, by=x->x[1])
        nbslices = length(cgtree)
        fig = Figure(size=(1800, 600 * div(nbslices+1, 2)), fontsize=30);
        for (i, ((t, s), ssd)) in enumerate(cgtree)
            println("$i t=$(round(t, digits=4))  s=$(round(s, digits=4))  maxk=$(maximum(keys(ssd)))")
            ax = Axis(fig[div(i-1, 2)+1, mod1(i, 2)],
            title="SSD of slice (t=$(round(t, digits=3)), s=$(round(s, digits=3)))",
            xlabel="subtree size", ylabel="probability",
            xscale=log2, yscale=log);
            plotssd!(ax, ((t, s), ssd), params, cumulative=cumulative, modelK=modelK);
        end
        return fig
    end
end

function plot(chain; burn=0)
    if (burn isa Int && (abs(burn) > length(chain)))        @error "burn must be less than the length of the chain (len=$(length(chain)))"
        return nothing
    end

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