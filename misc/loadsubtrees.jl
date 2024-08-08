using JSON, Glob

ssdfiles = glob("*subtree*", "/Users/alice/Documents/PhD/phyloinverse_lite/data")

data = Dict{String, Dict{@NamedTuple{t::Float64, s::Float64}, Vector{@NamedTuple{k::Int64, n::Int64}}}}()
for fn in ssdfiles
    ontology = split(split(fn, '/')[end], '.')[1]
    open(fn, "r") do f
        ssds = JSON.parse(f)
        data[ontology] = Dict((t=ssd["t"], s=ssd["s"]) => @NamedTuple{k::Int64, n::Int64}.(ssd["subtree_size_distribution"]) for ssd in ssds)
    end
end
data