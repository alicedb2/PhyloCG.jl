using JSON, Glob, DataStructures

ssdfiles = glob("*subtree*", "data")
data = Dict{String, CGTree}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        data[ontology] = CGTree((t=ssd["t"], s=ssd["s"]) => DefaultDict(0, [Pair(kn...) for kn in ssd["subtree_size_distribution"]]...) for ssd in ssds)
    end
end
println("Loaded $(length(data)) ontologies")

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
# Remove all trailing `__nan` from all ontologies
# and sort bins by length. Check that all remaining
# Check that all prefixes are the same
_nodiscr = 0
for (ssds, ontologies) in binned_unique
    nanless_bin = sort!(collect(Set(remove_trailing_nan.(ontologies))), by=length)
    binned_unique[ssds] = nanless_bin
    global _nodiscr
    unique_prefixes = Set([o[1:length(first(nanless_bin))] for o in nanless_bin])
    if length(unique_prefixes) > 1
        _nodiscr += 1
        print("*** ")
    end
end
for ontologies in sort!(collect(values(binned_unique)), by=first)
    println("Coarse: $(first(ontologies))")
    println("  Fine: $(last(ontologies))")
    println()
end

println("Found $(length(binned_unique)) unique ontology bins")
if iszero(_nodiscr)
    println("No prefix discrepencies found")
else
    println("Some bins contain ontologies with prefix discrepencies")
end

println()
println("Maximum K for each unique ontology:")
unique_data = Dict()
unique_bins = Dict()
for bin in values(binned_unique)
    ontology = last(bin)
    unique_data[ontology] = data[ontology]
    unique_bins[ontology] = bin
end
ontologies_maxmax = sort!(map(kv -> (kv[1] => maxmax(kv[2])), collect(unique_data)), by=x->x[2])
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