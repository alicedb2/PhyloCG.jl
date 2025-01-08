#!/usr/bin/env -S julia --color=yes --startup-file=no

using Distributed
using PhyloCG
using JLD2
using JSON

if restartworkers
    rmprocs(workers())
    addprocs(96, 
            topology=:master_worker,
            env=["JULIA_NUM_THREADS"=>"1"]
            )
    wp = WorkerPool(workers())
end

@everywhere begin
    using PhyloCG
    using JLD2
    using JSON
    include("$(pwd())/misc/loadsubtrees.jl")
end

@everywhere begin
    function workerfun(ontology; model="fbdh", nbiter=:ess, maxsubtree=Inf, ess50target=200, initmaxsubtree=400, nbinititer=777)
        # include("misc/loadsubtrees.jl")
        data = unique_data[ontology]
        chain = Chain(data, AMWG(model), maxsubtree=initmaxsubtree)
        # Fast-initialize using a small maxsubtree
        # to quickly find a good spot in parameter space
        # to start the slower MCMC chain
        advance_chain!(chain, nbinititer, progressoutput=:file)
        burn!(chain, nbinititer)
        newmaxsubtree!(chain, maxsubtree)
        chain.logprob_chain[1] = chain.sampler.current_logprob
        advance_chain!(chain, nbiter, ess50target=ess50target, progressoutput=:file)

        prefix = "results/$(ontology).$(model)_length$(length(chain))_maxmax$(maxmax(data))_maxsubtree$(maxsubtree)"
        chainfile = prefix * ".jld2"
        resultfile = prefix * ".json"

        results = Dict(
            "length" => length(chain),
            "ontology" => ontology,
            "model" => model,
            "bestsample" => NamedTuple(bestsample(chain)),
            "ess_rhat" => ess_rhat(chain),
            "maxmax" => maxmax(data),
            "maxsubtree" => maxsubtree
        )

        jldsave(chainfile,
            chain=chain,
            ontology=ontology,
            results=results
            )

        open(resultfile, "w") do io
            write(io, json(results, 4))
        end

        run(`gzip -f $(chainfile)`)

        return results
    end
end

jobs = []
for ontology in keys(unique_data)
    push!(jobs, remotecall(workerfun, wp,
                           ontology,
                           model="fbdh",
                           maxsubtree=5000,
                           nbiter=:ess, ess50target=200,
                           initmaxsubtree=400, nbinititer=777
                           )
         )
end