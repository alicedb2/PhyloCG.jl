mutable struct Chain
    sampler::Sampler
    cgtree::CGTree
    truncated_cgtree::CGTree
    params_chain::Vector{ComponentArray{Float64}}
    logprob_chain::Vector{Float64}
    maxsubtree::Real
    # _probs::Vector{ODEProblem}
end

"""
    Chain(cgtree, sampler; maxsubtree=Inf)

Create a chain

### Arguments
- `cgtree::CGTree`: the coarse-grained tree
- `sampler::Sampler`: the sampler specifying the model
- `maxsubtree::Real`: the maximum subtree size to consider when calculating the log probability of the model

### Returns
- `chain::Chain`: the chain
"""
function Chain(cgtree, sampler; maxsubtree=Inf)
    cgtree = deepcopy(cgtree)
    truncated_cgtree = deepcopy(cgtree)
    if isfinite(maxsubtree)
        truncated_cgtree = truncate!(truncated_cgtree, maxsubtree)
    end
    init_logprob = logdensity(cgtree, sampler.params, maxsubtree=maxsubtree)
    init_params = deepcopy(sampler.params)
    chain = Chain(sampler, cgtree, truncated_cgtree, [init_params], [init_logprob], maxsubtree)
    return chain
end

function newmaxsubtree!(chain::Chain, maxsubtree=Inf)
    maxsubtree > 0 || throw(ArgumentError("maxsubtree must be positive"))
    maxsubtree == chain.maxsubtree && return chain
    chain.truncated_cgtree = truncate(chain.cgtree, maxsubtree)
    chain.maxsubtree = maxsubtree
    chain.sampler.current_logprob = logdensity(chain.truncated_cgtree, chain.sampler.params)
    return chain
end

Base.length(chain::Chain) = length(chain.logprob_chain)

function chainsamples(chain::Chain, syms...; burn=0)

    burn = _burnlength(length(chain), burn)

    if isempty(syms)
        return reduce(hcat, chain.params_chain[burn+1:end])
    elseif first(syms) isa Int
        return [p[first(syms)] for p in chain.params_chain[burn+1:end]]
    elseif syms === (:logdensity,)
        return chain.logprob_chain[burn+1:end]
    else
        return [reduce((x, s) -> getproperty(x, s), syms, init=p) for p in chain.params_chain[burn+1:end]]
    end
end

"""
    argmax(chain::Chain, logprob=:logdensity)

Return the index of the sample with the highest logprob

### Arguments
- `chain::Chain`: the chain
- `logprob::Symbol`: the logprob to maximize. Can be `:logdensity` (default), `:map` (equiv to :logdensity), `:mle`, or `:prior`.
"""
function Base.argmax(chain::Chain, logprob=:logdensity)
    if logprob === :map || logprob === :logdensity
        return argmax(chain.logprob_chain)
    elseif logprob === :mle
        return argmax(chain.logprob_chain .- log_priors.(chain.params_chain))
    elseif logprob === :prior
        return argmax(log_priors.(chain.params_chain))
    else
        throw(ArgumentError("logprob must be :logdensity (equiv :map), :mle, or :prior"))
    end
end


"""
    bestsample(chain::Chain, logprob=:map)

Return the sample with the highest logprob

### Arguments
- `chain::Chain`: the chain
- `logprob::Symbol`: the logprob to maximize. Can be `:logdensity` (default), `:map` (equiv to :logdensity), `:mle`, or `:prior`.
"""
bestsample(chain::Chain, logprob=:map) = chain.params_chain[argmax(chain, logprob)]

_mapround(y, digits=4) = map(er->map(x->round(x, digits=digits), er), y)

function ess_rhat(chain::Chain, syms...; burn=0, digits=4)
    if isempty(syms)
        NamedTuple(ComponentArray(_mapround(ess_rhat.(chainsamples.(Ref(chain), 1:length(chain.sampler.params), burn=burn))), getaxes(chain.sampler.params)))
    else
        return _mapround(ess_rhat(chainsamples(chain, syms...; burn=burn)))
    end
end

function convergence(chain::Chain; burn=0.5, digits=4)
    paramsamples = chainsamples(chain, burn=burn)
    # Remove parameters absent from the model
    paramsamples = paramsamples[.!allequal.(eachrow(paramsamples)), :]
    # Append logprob chain
    logprobs = chainsamples(chain, :logdensity, burn=0.5)'
    chains = vcat(logprobs, paramsamples)'
    # Standardize chains
    chains = (chains .- mean(chains, dims=1)) ./ std(chains, dims=1)
    # Reshape chains for ess_rhat such that each parameter + logprob
    # chain is considered as a separate chain of the same process
    # instead of a single chain with multiple parameters
    chains = reshape(chains, size(chains, 1), size(chains, 2), 1)
    return map(x->round(x[1], digits=digits), ess_rhat(chains))
end

function _burnlength(len, burn)
    if burn > 0
        if 0 < burn < 1
            burn = round(Int, burn * len)
        end
    elseif burn < 0
        if -1 < burn < 0
            burn = round(Int, -burn * len)
        end
        burn = len + burn
    elseif iszero(burn)
        return 0
    end
    if burn > len
        throw(ArgumentError("burn must be less than the chain length"))
    end
    return burn
end

"""
    burn!(chain::Chain, burn=0)
    burn!(chain::GOFChain, burn=0)

In-place burn-in

### Arguments
- If `burn` is a positive integer, the first `burn` samples are removed.
- If `burn` is a negative integer, only the last `|burn|` samples are kept.
- If `burn` is a float between 0 and 1, the fist `burn * length(chain)` samples are removed.
- If `burn` is a float between -1 and 0, the last `|burn| * length(chain)` samples are kept.
"""
function burn!(chain::Chain, burn=0)

    burn = _burnlength(length(chain), burn)

    # Set end before burn in case we are burning
    # the whole chain
    chain.sampler.current_logprob = chain.logprob_chain[end]
    chain.sampler.params = chain.params_chain[end]
    chain.logprob_chain = chain.logprob_chain[burn+1:end]
    chain.params_chain = chain.params_chain[burn+1:end]

    return chain
end

"""
    burn(chain, burn=0)

Burn-in a copy of the chain, see `burn!`
"""
burn(chain, burn=0) = burn!(deepcopy(chain), burn)

"""
    advance_chain!(chain::Chain, nbiter; ess50target=100, essparam=:logensity, progressoutput=:repl)

Advance the chain by `nbiter` iterations

### Arguments
- `nbiter::Int|Symbol`: the number of iterations to advance the chain by.
  - If `nbiter` is an integer, the chain is advanced by `nbiter` iterations.
  - If `nbiter` is `:ess`, the chain is advanced until the effective sample size (ESS) of the chain is greater than or equal to `ess50target`.
- `progressoutput::Symbol`: Where to output the progress meter
  - `:repl` to show progress in the REPL
  - `:file` to write progress to a file `"progress_pid\$(getpid()).txt`. The file is deleted once `advance_chain!` is done.

### Returns
- `chain::Chain`: the chain

# Notes
- If a file named `stop` is found in the current directory, the chain is stopped gracefully.
"""
function advance_chain!(chain::Chain, nbiter; ess50target=100, progressoutput=:repl)
    s = chain.sampler

    if progressoutput === :repl
        progressio = stderr
    elseif progressoutput === :file
        progressoutputfn = "progress_pid$(getpid()).txt"
        progressio = open(progressoutputfn, "w")
    end

    if !(nbiter isa Int)
        prog = ProgressUnknown(showspeed=true, output=progressio)
        _nbiter = typemax(Int64)
    else
        prog = Progress(nbiter; showspeed=true, output=progressio)
        _nbiter = nbiter
    end

    _conv = "wait 40"
    conv = (ess=NaN, rhat=NaN)
    for n in 1:_nbiter
        isfile("stop") && break
        # advance!(chain.sampler, chain.cgtree, maxsubtree=chain.maxsubtree)
        advance!(chain.sampler, chain.truncated_cgtree)
        push!(chain.logprob_chain, s.current_logprob)
        push!(chain.params_chain, deepcopy(s.params))

        if length(chain) >= 100 && (mod(n-1, 10) == 0)
            conv = convergence(chain, burn=0.5)
            _conv = "$(conv)"
        end

        next!(prog, showvalues=[
            ("iteration", n),
            ("chain length", length(chain)),
            ("logprob (best @ idx)", @sprintf("%.2f (%.2f @ %i)", s.current_logprob, maximum(chain.logprob_chain), argmax(chain))),
            ("convergence (burn 50%)", _conv)
        ])

        if nbiter === :ess && conv.ess >= ess50target
            println("Convergence target reached!")
            break
        end

    end
    finish!(prog)

    if progressoutput === :file
        close(progressio)
        rm(progressoutputfn)
    end

    return chain
end