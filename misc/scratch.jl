using BenchmarkTools
using PhyloCG
using CairoMakie

include("misc/loadsubtrees.jl")

# cgtree = data["envo__terrestrial_biome__shrubland_biome__subtropical_shrubland_biome"]; maxmax(cgtree)
cgtree = data["envo__terrestrial_biome__desert_biome__polar_desert_biome"]; maxmax(cgtree)
cgtree = data["envo__terrestrial_biome__mangrove_biome"]; maxmax(cgtree)
cgtree = data["envo__aquatic_biome__marine_biome__marginal_sea_biome"]; maxmax(cgtree)
cgtree = data["envo__terrestrial_biome__anthropogenic_terrestrial_biome__village_biome"]; maxmax(cgtree)
cgtree = data["envo__terrestrial_biome__grassland_biome__tropical_grassland_biome"]; maxmax(cgtree)

cgtree = data["envo__terrestrial_biome__shrubland_biome__tropical_shrubland_biome"]; maxmax(cgtree)

sort(map(kv -> (kv[1] => maxmax(kv[2])), collect(data)))
sort(map(kv -> (kv[1] => maxmax(kv[2])), collect(data)), by=x->x[2])

chain = Chain(cgtree, AMWG("fbdh"))
advance_chain!(chain, 50)
advance_chain!(chain, 100)

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


rs = LinRange(0.999:0.00001:1.001)
phs = Phi.(rs, 0.25, 0.125, bestsample(chain2)...)
lines(rs, real(phs) .- 1);
lines!(rs, imag(phs));
current_figure()

function meanbdih(t, b, d, rho, g, eta, alpha, beta)
    # rate = b - d + rho / (1 - g) + eta * alpha / (alpha + beta)
    rate = b - d + rho * g / (1 - g) + eta * alpha / (beta - 1)
    return exp(rate * t)
end

t, s = 0.9, 0.0
b, d, rho, g, eta, alpha, beta = 0.5, 0.0, 0.0, 0.5, 0.0, 2.0, 1.5
lphs = logphis(1024, t, s, 1.0, b, d, rho, g, eta, alpha, beta)
sum((0:length(lphs)-1) .* exp.(lphs))
meanbdih(t - s, b, d, rho, g, eta, alpha, beta)


N = 128
f(x) = 1 ./ (1 .- x) .- 1
rs = 0.0001:0.0001:0.9999
foo = []
for r in rs
    cc = r * exp.(2pi * im * (0:div(N, 2)) / N)
    s = f(cc)
    coeffs = irfft(conj(s), N)
    push!(foo, log(mean(abs.(vcat(s, s[2:end-1])))) - log(abs(last(coeffs) - (N-1)*log(r))) - (N-1) * log(r))
end
kappa, idx = findmin(abs.(foo)); r = rs[idx]
r = 0.99;
r = 1 - 1/N
# cc = r * exp.(2pi * im * (0:div(N, 2)) / N);
cc = r * exp.(2pi * im * (1:N) / N);
s = f.(cc);
s[end] = abs(s[end]);
# coeffs = irfft(conj(s), N);
coeffs = ifft(reverse(s))
coeffs = abs.(coeffs)
a = [c > 0 ? log(c) : -Inf for c in coeffs] .- (0:N-1) .* log(r)
a .- a[1]

foo = copy(empo.sampler.params)
foo.i.rho = 0.0
plotssds(empo.cgtree, foo, maxsubtree=4000)




problematic = ComponentVector{Float64}(f = 0.9987439882710604, b = 0.005685406623455595, d = 0.0022785453482634128, i = (rho = 0.09356531091384332, g = 0.5654596214606173), h = (eta = 0.0674255835353483, alpha = 23.710672150516874, beta = 1.0341308074309457))
fig = Figure();
ax = Axis(fig[1, 1])
N = 16_000
gap = 1/N
lps = logphis(N, 0.9, 0.5, empo.sampler.params..., gap=gap)
# @time lps = logphis(N, 0.9, 0.5, 1.0, 0.5, 1.0, 0.9, 0.5, 0.5, 2.0, 1.5, gap=gap)
# lps = logphis(N, 0.9, 0.5, problematic..., gap=gap)
replace!(lps, -Inf => NaN)
ks = 1:length(lps)
lines!(ax, log2.(ks), lps)
# xlims!(ax, log2(N)-1.1, log2(N)+0.1)
# ylims!(ax, lps[N-256]-2, lps[N-256]+5)
fig



t = 0.9
eta, alpha, beta = 0.1, 2.1, 1.5
zs = collect(LinRange(0.9, 1.1, 100)) .+ 1e-13im
us = Ubdih.(zs, t, 0.0, 0.0, 0.0, 0.0, eta, alpha, beta)
func(z, t, eta, alpha, beta) = z + eta * t * (alpha / (beta - 1) * (z - 1) - pi * csc(pi * beta) * exp(loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta)) * (1 - z)^beta)
func2(z, t, eta, alpha, beta) = z + eta * t * (alpha / (beta - 1) / (beta - 1) * (z - 1) * (z * (alpha + beta - 1) - alpha - 1) + pi * csc(pi * beta) * exp(loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta)) * (1 - z)^beta * (z * (alpha + beta - 1) - alpha - beta))

fig = Figure();
ax = Axis(fig[1, 1])
lines!(ax, real(zs), real.(us) .- 1)
# lines!(ax, real(zs), imag.(us))
lines!(ax, real(zs), real(func.(zs, t, eta, alpha, beta)) .- 1)
# lines!(ax, real(zs), imag(func.(zs, t, eta, alpha, beta)))
fig