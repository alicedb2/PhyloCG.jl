using JSON, Glob

ssdfiles = glob("*subtree*", "/Users/alice/Documents/PhD/phyloinverse_lite/data")

data = Dict{String, Dict{Tuple{Float64, Float64}, Vector{Tuple{Int64, Int64}}}}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        data[ontology] = Dict((ssd["t"], ssd["s"]) => Tuple{Int64, Int64}.(ssd["subtree_size_distribution"]) for ssd in ssds)
    end
end
data