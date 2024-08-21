using JSON, Glob

maxmax(t) = maximum(maximum.(map(kn -> getfield.(kn, :k), values(t))))

ssdfiles = glob("*subtree*", "/Users/alice/Documents/PhD/phyloinverse_lite/data")

# data = Dict{String, Dict{@NamedTuple{t::Float64, s::Float64}, Tuple}}()
data = Dict{String, Dict{@NamedTuple{t::Float64, s::Float64}, Vector{@NamedTuple{k::Int64, n::Int64}}}}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        # data[ontology] = Dict((t=ssd["t"], s=ssd["s"]) => Tuple(@NamedTuple{k::Int64, n::Int64}.(ssd["subtree_size_distribution"])) for ssd in ssds)
        data[ontology] = Dict((t=ssd["t"], s=ssd["s"]) => @NamedTuple{k::Int64, n::Int64}.(ssd["subtree_size_distribution"]) for ssd in ssds)
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