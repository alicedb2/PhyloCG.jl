using Distributed
using PrettyTables
addprocs(96)

@everywhere begin
    using Revise
    using PhyloCG
    using JLD2
    using JSON
    include("misc/loadsubtrees.jl")
end

@everywhere begin
    function workerfun(ontology; model="fbdh", maxsubtree=Inf, initmaxsubtree=200, nbinititer=777)
        include("misc/loadsubtrees.jl")
        data = unique_data[ontology]


        chain = Chain(data, AMWG(model), maxsubtree=initmaxsubtree)
        # Fast-initialize using a small maxsubtree
        # to quickly find a good spot in parameter space
        # to start the slower MCMC chain
        advance_chain!(chain, nbinititer, pretty_progress=:file)
        burn!(chain, nbinititer)
        chain.maxsubtree = maxsubtree
        chain.sampler.current_logprob = logdensity(chain.cgtree, chain.sampler.params, maxsubtree=chain.maxsubtree)
        chain.params_chain[1] = chain.sampler.current_logprob

        prefix = "results/$(ontology).$(model)_length$(length(chain))_maxmax$(maxmax(data))_maxsubtree$(maxsubtree)"

        results = Dict(
            "ontology" => ontology,
            "model" => model,
            "bestsample" => NamedTuple(bestsample(chain)),
            "ess_rhat" => ess_rhat(chain),
            "maxmax" => maxmax(data),
            "maxsubtree" => maxsubtree
        )

        jldsave(prefix * ".jld2",
            chain=chain,
            ontology=ontology,
            results=results
            )

        open(prefix * ".json", "w") do io
            write(io, json(results, 4))
        end

        run(`gzip -f $(resultfile)`)

        return results
    end
end

jobs = []
for ontology in keys(unique_data)
    push!(jobs, remotecall(workerfun, wp,
                           ontology,
                           nbiter=100,
                           model="fbdh",
                           maxsubtree=512
                           )
         )
    break
end