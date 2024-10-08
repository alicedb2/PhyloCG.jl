using JSON, Glob, DataStructures

maxmax(t) = maximum(maximum.(map(kn -> keys(kn), values(t))))

ssdfiles = glob("*subtree*", "/Users/alice/Documents/PhD/phyloinverse_lite/data")
data = Dict{String, CGTree}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        data[ontology] = Dict((t=ssd["t"], s=ssd["s"]) => DefaultDict(0, [Pair(kn...) for kn in ssd["subtree_size_distribution"]]...) for ssd in ssds)
    end
end
data

binned_unique = Dict()
for (ontology, ssds) in data
    if ssds in keys(binned_unique)
        binned_unique[ssds] = push!(binned_unique[ssds], ontology)
    else
        binned_unique[ssds] = [ontology]
    end
end
for (ssds, ontologies) in binned_unique
    sort!(ontologies, by=length)
end

unique_data = Dict()
for ontology in first.(values(binned_unique))
    unique_data[ontology] = data[ontology]
end

sort!(map(kv -> (kv[1] => maxmax(kv[2])), collect(unique_data)), by=x->x[2])