using BenchmarkTools
using PhyloCG
using CairoMakie

include("misc/loadsubtrees.jl")
maxmax(t) = maximum(maximum.(map(kn -> getfield.(kn, :k), values(t))))

# cgtree = data["envo__terrestrial_biome__shrubland_biome__subtropical_shrubland_biome"]
# cgtree = data["envo__terrestrial_biome__desert_biome__polar_desert_biome"]
cgtree = data["envo__terrestrial_biome__mangrove_biome"]
cgtree = data["envo__aquatic_biome__marine_biome__marginal_sea_biome"]
cgtree = data["envo__terrestrial_biome__shrubland_biome__tropical_shrubland_biome"]
cgtree = data["envo__terrestrial_biome__grassland_biome__tropical_grassland_biome"]
cgtree = data["envo__terrestrial_biome__anthropogenic_terrestrial_biome__village_biome"]
sort(map(kv -> (kv[1] => maxmax(kv[2])), collect(data)))
sort(map(kv -> (kv[1] => maxmax(kv[2])), collect(data)), by=x->x[2])

chain = Chain(cgtree, AMWG("fbdh"))
advance_chain!(chain, 5000)
chain2 = Chain(cgtree, AMWG("fbdh"))
advance_chain!(chain2, 5000)
# chain = Chain(cgtree, AM("fbd"))
# chain = Chain(cgtree, LatentSlice("fbdh"))
plot(chain, burn=500)


chain.logprob_chain = chain.logprob_chain[1:7000]
chain.params_chain = chain.params_chain[1:7000]
chain.sampler.current_logprob = chain.logprob_chain[end]
chain.sampler.params = chain.params_chain[end]

fig = Figure(size=(500, 500), fontsize=20);
# fig = Figure(size=(3*300, 20*300));
ax = Axis(fig[1, 1], xlabel="eta", ylabel="alpha")
# eabs = [(eta=p.h.eta, alpha=p.h.alpha, beta=p.h.beta) for p in chain.params_chain[div(end, 2):end]]
eabs = [(eta=p.h.eta, alpha=p.h.alpha, beta=p.h.beta) for p in chain.params_chain]
# _eabs = eabs[1:div(end, 300):end]
# _eabs = eabs[1000:end]
# hexbin!(ax, [(p.eta, p.alpha) for p in eabs], bins=50)
# hexbin!(ax, [Tuple([_uv(p.h.eta, p.h.alpha).v, p.h.beta]) for p in chain.params_chain], bins=50)
hexbin!(ax, [Tuple(_uvw(p.h.eta, p.h.alpha, p.h.beta)[[2, 3]]) for p in chain.params_chain], bins=30)
fig
s = bestsample(chain)
suv = @NamedTuple{u, v}(_uv(s.h.eta, s.h.alpha))
_pts = [Point2(_ea(u, suv.v)) for u in LinRange(-3, suv.u+1, 100)]
_pts2 = [Point2(_ea(suv.u, v)) for v in LinRange(1, suv.v+1, 100)]
scatter!(ax, _pts, markersize=9, color=Cycled(2))
scatter!(ax, _pts2, markersize=9, color=Cycled(2))
ax = Axis(fig[1, 2], xlabel="u", ylabel="v")
hexbin!(ax, [Point2(_uv(p.eta, p.alpha)) for p in eabs], bins=50)
fig


# hexbin([Tuple(((x,y)->(log(x),y))(_uv(p.eta, p.alpha, p.beta)...)) for p in eabs], bins=50)

# hexbin!(ax, [Tuple(_uv(p.eta, p.alpha, p.beta)) for p in eabs], bins=50)
# # _pts = [Point2(sqrt(u * _bs_uv[2] / _bs.h.beta), sqrt(_bs.h.beta * _bs_uv[2] / u)) for u in LinRange(0, 5, 100)]
# suv = (u=s.h.eta * s.h.beta / s.h.alpha, v=s.h.eta * s.h.alpha)
# _pts = [Point2(sqrt(s.h.beta * suv.u * v), sqrt(v / s.h.beta / suv.u)) for v in LinRange(0, 10, 100)]
# _v = suv.v
# _v = 10
# _pts = [Point2(sqrt(s.h.beta * u * _v), sqrt(_v / s.h.beta / u)) for u in LinRange(0, 1, 1000)]
_pts = [Point2(_uv(e, a, beta)) for e in 0.5 .+ rand(10) for a in 0.5 .+ rand(10)]
scatter!(ax, _pts, markersize=9, color=Cycled(2))
fig
# # streamplot!(ax, (eta, alpha) -> Point2f(-1.44*eta/alpha/sqrt(1+(1.44*eta/alpha)^2), 1/sqrt(1+(1.44 * eta/alpha)^2)), 0..2, 0..20)
# # streamplot!(ax, (eta, alpha) -> Point2f(-1.44*eta*), 1/sqrt(1+(1.44 * eta/alpha)^2)), 0..2, 0..20)
# # arrows!(ax, getfield.(_eabs, :eta), getfield.(_eabs, :alpha), -getfield.(_eabs, :beta) .* getfield.(_eabs, :eta) ./ getfield.(_eabs, :alpha) ./ sqrt.(1 .+ (getfield.(_eabs, :beta) .* getfield.(_eabs, :eta) ./ getfield.(_eabs, :alpha)).^2), 1 ./ sqrt.(1 .+ (getfield.(_eabs, :beta) .* getfield.(_eabs, :eta) ./ getfield.(_eabs, :alpha)).^2))
# arrows!(ax, Point2.(getfield.(_eabs, :eta), getfield.(_eabs, :alpha)), grad.(_eabs), linecolor=Cycled(1), arrowcolor=Cycled(1)) 
# arrows!(ax, Point2.(getfield.(_eabs, :eta), getfield.(_eabs, :alpha)), orthgrad.(_eabs), linecolor=Cycled(2), arrowcolor=Cycled(2)) 
_ea = sample(_eabs)

fig


c = []
eab = [1.0, 1.0, 1.0]
acc, rej = [0, 0, 0], [0, 0, 0]
batchsize = 30
nbbatch = 0
ls = [0.0, 0.0, 0.0]
_acc = [0.0, 0.0, 0.0]
neab = nothing
nuvw = nothing
nbiter = 10000
for k in 1:nbiter
    for _ in 1:batchsize
        for i in 1:3
            uvw = _uvw(eab...)
            nuvw = _uvw(eab...)
            nuvw[i] = uvw[i] + exp(ls[i]) * randn()
            neab = _eab(nuvw...)
            if !all(0 .< neab .< 5)
                acc_logp = -Inf
            else
                acc_logp = PhyloCG.logjac_deabduvw(neab...) - PhyloCG.logjac_deabduvw(eab...)
                # acc_logp = log(nuv[2]) - log(uv[2])
            end
            if log(rand()) < acc_logp
                eab .= neab
                acc[i] += 1
            else
                rej[i] += 1
            end
        end
    end
    push!(c, eab[:])
    
    nbbatch += 1
    delta_n = min(0.01, 1/sqrt(nbbatch))
    _acc = acc ./ (acc + rej)
    ls .+= delta_n .* (_acc .> 0.44) .- delta_n .* (_acc .< 0.44)
    if k < nbiter
        acc, rej = [0, 0, 0], [0, 0, 0]
    end
end
scatter(getindex.(c, 1), getindex.(c, 2), markersize=4)
scatter(getindex.(c, 1), getindex.(c, 3), markersize=4)
scatter(getindex.(c, 2), getindex.(c, 3), markersize=4)
hexbin(getindex.(c, 1), getindex.(c, 2))
_acc



_pts = [Point2(_uv(e, a, beta)) for e in 1 .+ rand(100) for a in 1 .+ rand(100)]
scatter(_pts)