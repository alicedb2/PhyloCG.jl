using PhyloCG
using CairoMakie

include("misc/loadsubtrees.jl")

# cgtree = data["envo__terrestrial_biome__shrubland_biome__subtropical_shrubland_biome"]
# cgtree = data["envo__terrestrial_biome__desert_biome__polar_desert_biome"]
# cgtree = data["envo__terrestrial_biome__mangrove_biome"]
cgtree = data["envo__terrestrial_biome__desert_biome__polar_desert_biome"]
maximum(maximum.(map(kn -> getfield.(kn, :k), values(cgtree))))


chain = Chain(cgtree, AMWG("fbd"))
chain = Chain(cgtree, AM("fbd"))
advance_chain!(chain, 1000)
plot(chain, burn=500)