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
    function workerfun(ontology; nbiter=200, model="fbdh", maxsubtree=Inf)
        include("misc/loadsubtrees.jl")
        data = unique_data[ontology]
        chain = Chain(data, AMWG(model), maxsubtree=maxsubtree)
        advance_chain!(chain, nbiter, pretty_progress=:file)
        resultfile = "results/$(ontology).$(model)_length$(length(chain))_maxmax$(maxmax(data))_maxsubtree$(maxsubtree).jld2"
        results = Dict(
            "ontology" => ontology,
            "model" => model,
            "bestsample" => NamedTuple(bestsample(chain)),
            "ess_rhat" => ess_rhat(chain),
            "maxmax" => maxmax(data),
            "maxsubtree" => maxsubtree
        )
        
        jldsave(resultfile, 
            chain=chain, 
            ontology=ontology,
            results=results
            )
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