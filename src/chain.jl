mutable struct Chain
    sampler::Sampler
    cgtree::CGTree
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
    chain = Chain(sampler, cgtree, [], [], maxsubtree)
    chain.sampler.current_logprob = logdensity(cgtree, sampler.params, maxsubtree=maxsubtree)
    return chain
end

function chainsamples(chain::Chain, syms...; burn=0)

    if 0 < burn < 1
        burn = round(Int, burn * length(chain))
    elseif -1 < burn < 0
        burn = round(Int, -burn * length(chain))
    end

    if burn < 0
        burn = length(chain) + burn
    end

    if burn > length(chain)
        @error "burn must be less than the chain length"
        return nothing
    end

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

Base.length(chain::Chain) = length(chain.logprob_chain)


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
    if burn > 0
        if 0 < burn < 1
            burn = round(Int, burn * length(chain))
        end
    elseif b < 0
        if -1 < burn < 0
            burn = round(Int, -burn * length(chain))
        end
        burn = length(chain) + burn
    end
    if burn > length(chain)
        @error "burn must be less than the chain length"
        return chain
    end

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

Burn-in a copy of the chain

### Arguments
- If `burn` is a positive integer, the first `burn` samples are removed.
- If `burn` is a negative integer, only the last `|burn|` samples are kept. 
- If `burn` is a float between 0 and 1, the fist `burn * length(chain)` samples are removed. 
- If `burn` is a float between -1 and 0, the last `|burn| * length(chain)` samples are kept.
"""
function burn(chain, burn=0)
    return burn!(deepcopy(chain), burn)
end

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
        progressoutput = "progress_pid$(getpid()).txt"
        progressio = open(progressoutput, "w")
    end

    if !(nbiter isa Int)
        prog = ProgressUnknown(showspeed=true, output=progressio)
        _nbiter = typemax(Int64)
    else
        prog = Progress(nbiter; showspeed=true, output=progressio)
        _nbiter = nbiter
    end

    conv = nothing
    for n in 1:_nbiter
        isfile("stop") && break
        advance!(chain.sampler, chain.cgtree, maxsubtree=chain.maxsubtree)
        push!(chain.logprob_chain, s.current_logprob)
        push!(chain.params_chain, deepcopy(s.params))

        if length(chain) >= 20 && (mod(n-1, 10) == 0)
            conv = convergence(chain, burn=0.5)
        end

        next!(prog, showvalues=[
            ("iteration", n),
            ("chain length", length(chain)),
            ("logprob", @sprintf("%.2f", s.current_logprob)),
            ("convergence (burn 50%)", conv)
        ])

        if nbiter === :ess && conv.ess >= ess50target
            println("Convergence target reached!")
            break
        end

    end
    
    if progressoutput === :file
        close(progressio)
        rm(progressoutput)
    end

    return chain
end