const CGTree = Dict{@NamedTuple{t::Float64, s::Float64}, DefaultDict{Int, Int, Int}}
const Bouquet = @NamedTuple{trunk::@NamedTuple{i::Int, t::Float64, s::Float64, k::Int}, crown::@NamedTuple{i::Int, t::Float64, s::Float64, ks::Vector{Int}}}
const ModelSSDs = Dict{@NamedTuple{t::Float64, s::Float64}, Vector{Float64}}


function Base.isvalid(cgtree::CGTree)
    cgtree = sort!(collect(cgtree), by=tsssd->tsssd.first.t)
    first(cgtree).first.s == 0.0 || @warn "crown tip time != 0"
    for (((cr_t, cr_s), crown_ssd), ((tr_t, tr_s), trunk_ssd)) in zip(cgtree[1:end-1], cgtree[2:end])
        if cr_t != tr_s
            @warn "crown root time != trunk tip time, $(cr_s) != $(tr_t)"
            return false
        end
        if sum(values(crown_ssd)) != sum(keys(trunk_ssd) .* values(trunk_ssd))
            @warn "number of subtree in crown slice differs from number of tips in trunk slice"
            return false
        end
    end
    return true
end

# Note that popbouquet! leaves cgtree in an invalid state
# because it does not recursively update the crown of the crown
function popbouquet!(cgtree::CGTree; rng=Random.GLOBAL_RNG)::Bouquet
    slices = sort!(collect(cgtree), by=tsssd->tsssd.first.s)

    # Pick the trunk from slices except the one closest to the crown
    slice_masses = [sum(values(ssd)) for ((t, s), ssd) in slices[2:end]]
    any(slice_masses .> 0) || throw(ArgumentError("cgtree is empty"))
    slice_masses = slice_masses ./ sum(slice_masses)
    trunkslice_idx = rand(rng, Categorical(slice_masses)) + 1
    trunk_ts, trunk_ssd = slices[trunkslice_idx]
    crown_ts, crown_ssd = slices[trunkslice_idx-1]

    # Pick the trunk subtree
    trunk_mass = sum(values(trunk_ssd))
    trunk_mass > 0 || throw(ArgumentError("trunk slice is empty"))
    trunk_ks, trunk_ns = collect(keys(trunk_ssd)), collect(values(trunk_ssd))
    trunk_ps = trunk_ns / trunk_mass
    trunk_idx = rand(rng, Categorical(trunk_ps))
    trunk_k = trunk_ks[trunk_idx]
    trunk_ssd[trunk_k] -= 1
    if trunk_ssd[trunk_k] == 0
        delete!(trunk_ssd, trunk_k)
    end

    # Pick trunk_k crown subtrees
    # We have to do it without replacement
    crown_ssd_ks = collect(keys(crown_ssd))
    crown_ssd_mass = sum(values(crown_ssd))
    crown_ssd_mass > 0 || throw(ArgumentError("crown slice is empty"))
    crown_ssd_ns = collect(values(crown_ssd))
    crown_ks = Int[]
    for i in 1:trunk_k
        crown_ssd_ps = crown_ssd_ns / crown_ssd_mass
        crown_k_idx = rand(rng, Categorical(crown_ssd_ps))
        crown_k = crown_ssd_ks[crown_k_idx]
        push!(crown_ks, crown_k)
        crown_ssd_ns[crown_k_idx] -= 1
        crown_ssd_mass -= 1
        crown_ssd[crown_k] -= 1
        if crown_ssd[crown_k] == 0
            delete!(crown_ssd, crown_k)
        end
    end

    return (trunk=(i=trunkslice_idx, t=trunk_ts.t, s=trunk_ts.s, k=trunk_k), crown=(i=trunkslice_idx - 1, t=crown_ts.t, s=crown_ts.s, ks=crown_ks))

end

function pushbouquet!(cgtree::CGTree, bouquet::Bouquet)::CGTree
    trunk = bouquet.trunk
    crown = bouquet.crown

    trunk_ssd = cgtree[(t=trunk.t, s=trunk.s)]
    trunk_ssd[trunk.k] += 1

    crown_ssd = cgtree[(t=crown.t, s=crown.s)]
    for k in crown.ks
        crown_ssd[k] += 1
    end

    return cgtree
end