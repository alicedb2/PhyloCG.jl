using JSON, Glob, DataStructures

verbose = false

ssdfiles = glob("*subtree*", "data")
data = Dict{String, CGTree}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        data[ontology] = CGTree((t=ssd["t"], s=ssd["s"]) => DefaultDict(0, [Pair(kn...) for kn in ssd["subtree_size_distribution"]]...) for ssd in ssds)
    end
end
verbose && println("Loaded $(length(data)) ontologies")

# Use full CGTrees as keys to bin ontologies
# Feels like a hack, but it's actually
# clever and totally fine
function remove_trailing_nan(s::String)
    while occursin(r"__nan$", s)
        s = replace(s, r"__nan$" => "")
    end
    return s
end

binned_unique = Dict()
for (ontology, ssds) in data
    if ssds in keys(binned_unique)
        binned_unique[ssds] = push!(binned_unique[ssds], ontology)
    else
        binned_unique[ssds] = [ontology]
    end
end
sort!.(collect(values(binned_unique)))
sort!(collect(values(binned_unique)))
# for onts in values(binned_unique)
#     if contains(last(onts), "nan")
#         println("$(length(onts)), $(filter(contains("nan"), onts)))")
#         println()
#     end
# end

function treeify(ontologies)

    function treeify!(parent, path)
        print("$(path[1])")
        if length(path) == 1
            parent[path[1]] = Dict{Any, Any}(nothing => nothing)
            println()
        else
            if !(path[1] in keys(parent))
                parent[path[1]] = Dict{Any, Any}()
            end
            print("__")
            treeify!(parent[path[1]], path[2:end])
        end
    end

    tree = Dict()
    for ontology in ontologies
        println(ontology)
        treeify!(tree, String.(split(ontology, "__")))
        println()
    end

    return tree
end
ontology_tree = treeify(last.(sort!(collect(values(binned_unique)))))

function nbleaves(node)
    isnothing(node) && return 1
    return sum([minimalontologies(v) for v in values(node)])
end

function trim!(node)
end

verbose && println()
verbose && println("Maximum K for each unique ontology:")

unique_data = Dict()
unique_bins = Dict()
for bin in values(binned_unique)
    ontology = last(bin)
    unique_data[ontology] = data[ontology]
    unique_bins[ontology] = bin
end
ontologies_maxmax = sort!(map(kv -> (kv[1] => (maxmax(kv[2]), nbsubtrees(kv[2]))), collect(unique_data)), by=x->x[2][1])
ontologies_maxmax

# Output produced:
# data: Dict{String, CGTree}
#       All ontologies with their CGTree
# binned_unique: Dict{CGTree, Vector{String}}
#       All unique CGTree with bins of ontologies sharing the same CGTree
#       Each bin is sorted by length of the ontologies
#       We verify that all ontologies in a bin share the same prefix
# unique_bins = Dict{String, Vector{String}}
#       Ontology bins sharing the same CGTree
# unique_data: Dict{String, CGTree}
#       All unique ontologies with their CGTree
#       We pick the last ontology in each bin as
#       the representative so that it contains
#       the most information (longest ontology)
#
# unique_data is the required output for
# processing the full dataset and load
# into Chain/GOFChain