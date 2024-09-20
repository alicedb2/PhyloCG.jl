struct Subtree
    t::Float64
    s::Float64
    k::Int64
end

const Bouquet = @NamedTuple{trunk::Subtree, crown::Vector{Subtree}}
