function plotssd!(ax, slice, params=nothing; cumulative=false, modelK=Inf, ssdcolor=wong_colors()[5], ssdalpha=1.0, modelcolor=wong_colors()[6], setlims=true)
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
        barplot!(ax, ks, ps, fillto=lb, gap=0.0, width=1.0, strokewidth=1.0, color=(ssdcolor, ssdalpha), strokecolor=(ssdcolor, ssdalpha));
        stairs!(ax, 1:length(_ps), _ps, step=:center, color=modelcolor, linewidth=5);
        # ylims!(ax, minimum(filter(x-> x > 0, _ps))/4, ℯ)
        if setlims
            xlims!(ax, 1, max_empirical_k)
            ylims!(ax, lb, exp(1))
        end
    else
        lb = minimum(ps)/2
        barplot!(ax, ks, ps, fillto=lb, gap=0.0, width=1.0, strokewidth=1.0, color=(ssdcolor, ssdalpha), strokecolor=(ssdcolor, ssdalpha));
        if setlims
            ylims!(ax, lb, ℯ);
            xlims!(ax, 1, max_empirical_k)
        end
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

"""
    plotssds(cgtree::CGTree; params=nothing, cumulative=false, modelK=Inf)

Plot all subtree size distributions of a coarse-grained tree.
If `params` is provided, the model SSDs are plotted as well. 

### Arguments
- `cgtree::CGTree`: the coarse-grained tree
- `params::Nothing|ComponentArray{Float64}`: the parameters of the model
- `cumulative::Bool`: whether to plot the cumulative distributions
- `modelK::Real`: the maximum subtree size to consider when calculating the model SSDs

### Returns
- `fig::Figure`
"""
function plotssds(cgtree, params=nothing; cumulative=false, modelK=Inf, secondcgtree=nothing)
    with_theme(theme_minimal()) do
        cgtree = sort(cgtree, by=x->x[1])
        nbslices = length(cgtree)
        fig = Figure(size=(1800, 600 * div(nbslices+1, 2)), fontsize=30);
        for (i, ((t, s), ssd)) in enumerate(cgtree)
            if !isnothing(params)
                println("Calculating slice $i t=$(round(t, digits=4))  s=$(round(s, digits=4))  maxk=$(maximum(keys(ssd)))")
            end
            ax = Axis(fig[div(i-1, 2)+1, mod1(i, 2)],
                      title="SSD of slice (t=$(round(t, digits=3)), s=$(round(s, digits=3)))",
                      xlabel="subtree size", ylabel="probability",
                      xscale=log2, yscale=log);
            plotssd!(ax, ((t, s), ssd), params, cumulative=cumulative, modelK=modelK);
            if !isnothing(secondcgtree)
                plotssd!(ax, ((t, s), secondcgtree[(t=t, s=s)]), cumulative=cumulative, ssdcolor=wong_colors()[3], ssdalpha=0.5, setlims=false);
            end
        end
        return fig
    end
end

function plotssds(chain::Chain; bestparams=true, cumulative=false, modelK=Inf)
    if bestparams
        params = bestsample(chain)
    else
        params = nothing
    end
    if isfinite(chain.maxsubtree)
        return plotssds(chain.cgtree, params, cumulative=cumulative, modelK=modelK, secondcgtree=chain.truncated_cgtree)
    else
        return plotssds(chain.cgtree, params, cumulative=cumulative, modelK=modelK)
    end
end

"""
    plot(chain::Chain; burn=0)

Plot the trance and marginal distributions of the log density and parameters of a chain.

### Arguments
- `chain::Chain`: the chain
- `burn::Int|Float64`: the number or proportion of samples to discard from the beginning of the chain. Can be negative to keep samples from the end.

### Returns
- `fig::Figure`
"""
function plot(chain::Chain; burn=0)
    
    burnidx = _burnpos(length(chain), burn)

    with_theme(theme_minimal()) do
        nbparams = sum(chain.sampler.mask[:])
        fig = Figure(size=(1600, 400 * (1 + nbparams)), fontsize=30);

        _labels = labels(chain.sampler.params)

        axtrace = Axis(fig[1, 1], title="log density", xlabel="iteration", ylabel="log density");
        samples = chainsamples(chain, :logdensity, burn=burn)
        bestidx = argmax(samples)
        lines!(axtrace, burnidx+1:length(chain), samples, color=Cycled(1), linewidth=2);
        vlines!(axtrace, [bestidx], color=Cycled(6), linestyle=:dash, linewidth=4);
        axmarginal = Axis(fig[1, 2], title="log density", xlabel="log density", ylabel="probability");
        hist!(axmarginal, samples, bins=50, color=Cycled(1), normalization=:pdf);

        for (i, k) in enumerate(findall(chain.sampler.mask[:]))
            axtrace = Axis(fig[i+1, 1], xlabel="iteration", ylabel=_labels[k]);
            samples = chainsamples(chain, k, burn=burn)
            bestval = samples[bestidx]
            lines!(axtrace, burnidx+1:length(chain), samples, color=Cycled(1), linewidth=2);
            vlines!(axtrace, [bestidx], color=Cycled(6), linestyle=:dash, linewidth=4);
            axmarginal = Axis(fig[i+1, 2], xlabel=_labels[k], ylabel="probability");
            hist!(axmarginal, samples, bins=50, color=Cycled(1), normalization=:pdf);
            vlines!(axmarginal, [bestval], color=Cycled(2), linestyle=:dash, linewidth=4);
        end

        return fig
    end
end

function plot(chain::GOFChain; burn=0)
    empiricalG = first(chain.G_chain)
    chain = burn!(deepcopy(chain), burn)
    with_theme(theme_minimal()) do
        fig = Figure(size=(800, 900), fontsize=30);
        ax = Axis(fig[1, 1], title="G-statistic chain", xlabel="iteration", ylabel="G");
        lines!(ax, chain.G_chain, color=Cycled(1), linewidth=2);
        ax = Axis(fig[2, 1], title="G-statistic distribution", xlabel="G", ylabel="Frequency");
        hist!(ax, chain.G_chain, color=Cycled(1), normalization=:pdf, label="Null distribution");
        vlines!(ax, [empiricalG], color=Cycled(2), linestyle=:dash, linewidth=2, label="Empirical G");
        return fig
    end
end