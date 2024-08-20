using CairoMakie
using PhyloInverse
using Turing
using DifferentialEquations
using Optim, CMAEvolutionStrategy
import StatsPlots

include("misc/loadsubtrees.jl")

cgtree = data["envo__terrestrial_biome__desert_biome__polar_desert_biome"]
cgtree = sort([[collect(k), v] for (k, v) in cgtree])
maximum(maximum.([getindex.(sl, 1) for sl in getindex.(cgtree, 2)]))

cgtree = data["envo__aquatic_biome__freshwater_biome__freshwater_river_biome__Large_river_biome"]
cgtree = sort([[collect(k), v] for (k, v) in cgtree])
maximum(maximum.([getindex.(sl, 1) for sl in getindex.(cgtree, 2)]))

cgtree = data["empo__Free-living__Non-saline__Soil_non-saline"]
cgtree = sort([[collect(k), v] for (k, v) in cgtree])
maximum(maximum.([getindex.(sl, 1) for sl in getindex.(cgtree, 2)]))

cgtree = data["envo__terrestrial_biome__shrubland_biome__subtropical_shrubland_biome__mediterranean_shrubland_biome"]
cgtree = sort([[collect(k), v] for (k, v) in cgtree])
maximum(maximum.([getindex.(sl, 1) for sl in getindex.(cgtree, 2)]))


prob = ODEProblem(bdih, 0 + 0im, (0, 1), save_everystep=false, dense=false)
prob = ODEProblem(bdih!, 0 + 0im, (0, 1), save_everystep=false, dense=false)

# chain = sample(model, NUTS(0.65), 100)

model = BDIH(cgtree, "fbdh", prob)

invtransf = x -> [logistic(x[1]), exp(x[2]), exp(x[3]), exp(x[4]), logistic(x[5]), exp(x[6]), exp(x[7]), exp(x[8])]
model = BDIH_gaussian(cgtree, "fbdih", prob)
chain = sample(model, Turing.ESS(), 1000)

nbparams = length(chain.name_map.parameters);
_, maxldidx = findmax(chain.value[chain=1, var=[:lp]]);
best_sample = chain.value[chain=1][maxldidx[1], 1:nbparams]
optres = optimize(model, MAP(), best_sample, NelderMead(), 
                  Optim.Options(iterations=1000, show_trace=true,
                                g_abstol=1e-2, g_reltol=1e-2)
)

params_samples = invtransf.(eachrow(chain.value[chain=1][:, 1:end-1]))

datalikelihood_samples = chain.value[chain=1][:, end]
optres = optimize(model, MAP(), Optim.Options(iterations=1000, show_trace=true))

opt_ldf = DynamicPPL.LogDensityFunction(model);
objfun = x -> -DynamicPPL.LogDensityProblems.logdensity(opt_ldf, x)

opt_ldf = DynamicPPL.LogDensityFunction(model);
objfun = x -> -DynamicPPL.LogDensityProblems.logdensity(opt_ldf, x)

bdih_loglikelihood_gaussian = x -> -cgtreelogprob(cgtree, logistic(x[1]), exp(x[2]), exp(x[3]), exp(x[4]), logistic(x[5]), exp(x[6]), exp(x[7]), exp(x[8]), exp(x[8]), normalize=false)

bdih_objfun = x -> -cgtreelogprob(cgtree, x..., normalize=true)
bdh_objfun = x -> -cgtreelogprob(cgtree, vcat(x[1:3], [0.0, 0.5], x[4:6])..., PhyloInverse._bdihprob_inplace, normalize=true)
bdi_objfun = x -> -cgtreelogprob(cgtree, vcat(x, [0.0, 0.0, 0.0])..., normalize=true)
bd_objfun = x -> -cgtreelogprob(cgtree, vcat(x, [0.0, 0.5, 0.0, 1.0, 1.0])..., normalize=true)
# dh_objfun = x -> -cgtreelogprob(cgtree, vcat(x[1], 0.0, x[2], [0.0, 0.5], x[3:5])..., normalize=true)
function dh_objfun(x)
    try
        return -cgtreelogprob(cgtree, vcat(x[1], 0.0, x[2], [0.0, 0.5], x[3:5])..., normalize=true)
    catch e
        return Inf
    end
end

h_objfun = x -> -cgtreelogprob(cgtree, vcat(x[1], [0.0, 0.0, 0.0, 0.5], x[2:4])..., normalize=true)

opt_res_bdih = minimize(bdih_objfun, rand(8), 0.5, 
    lower=fill(0.0, 8),
    upper=[1.0, Inf, Inf, Inf, 1.0, Inf, Inf, Inf],
    ftol=1e-5, maxiter=1000)
bdih_map_est = xbest(opt_res_bdih)

opt_res_bdh = minimize(bdh_objfun, rand(6), 0.5, 
    lower=fill(0.0, 6),
    upper=[1.0, Inf, Inf, Inf, Inf, Inf],
    ftol=1e-5, maxiter=1000)
bdh_map_est = xbest(opt_res_bdh)
bdh_map_est = vcat(bdh_map_est[1:3], [0.0, 0.5], bdh_map_est[4:6])

opt_res_bdi = minimize(bdi_objfun, rand(5), 0.5, 
    lower=fill(0.0, 5),
    upper=[1.0, Inf, Inf, Inf, 1.0],
    ftol=1e-5, maxiter=1000)
bdi_map_est = vcat(xbest(opt_res_bdi), [0.0, 1.0, 1.0])

opt_res_dh = minimize(dh_objfun, rand(5), 1.0, 
    lower=fill(0.0, 5),
    upper=[1.0, Inf, Inf, Inf, Inf],
    ftol=1e-5, maxiter=1000)
dh_map_est = xbest(opt_res_dh)
dh_map_est = vcat(dh_map_est[1], 0.0, dh_map_est[2], [0.0, 0.5], dh_map_est[3:5])

opt_res_dh = minimize(dh_objfun, rand(5), 0.5, 
    lower=fill(0.0, 5),
    upper=[1.0, Inf, Inf, Inf, Inf],
    ftol=1e-5, maxiter=1000)
dh_map_est = xbest(opt_res_dh)
dh_map_est = vcat(dh_map_est[1], 0.0, dh_map_est[2], [0.0, 0.5], dh_map_est[3:5])
map_est = dh_map_est

map_est = bdih_map_est
map_est = bdi_map_est

map_est = [0.0177, 1.495, 0.0155, 0.144, 0.1593, 0.1072, 15.9959, 1.0809]
map_est = invtransf(best_sample)
maxn = maximum(maximum.(map(x -> getindex.(x, 1), getindex.(cgtree, 2))))
ncol = 4
nrow = div(length(cgtree)-1, ncol) + 1
fig = Figure(size=(ncol * 400, nrow * 400));
for i in 1:nrow, j in 1:ncol
    idx = (i-1) * nrow + j
    if idx > length(cgtree)
        break
    end
    (t, s), ssd = cgtree[idx]
    sort!(ssd)
    mass = sum(getindex.(ssd, 2))

    ax = Axis(fig[i, j]; 
    xlabel="k", ylabel="log ϕₖ", 
    title="t=$t, s=$s, mass=$mass",
    xscale=log2, yscale=log)
    
    # println(exp.(log.(getindex.(ssd, 2)) .- log(mass)))
    # println(getindex.(ssd, 1))
    # println(mass)
    # println(idx, " ", mass, " ", getindex.(ssd, 1), " ", getindex.(ssd, 2), " ", exp.(log.(getindex.(ssd, 2)) .- log(mass)))
    ks = getindex.(ssd, 1)
    logps = exp.(log.(getindex.(ssd, 2)) .- log(mass))
    logps = reverse(cumsum(reverse(logps)))
    barplot!(ax, ks .+ 0.5, logps, gap=0.0, alpha=0.4, width=Makie.automatic);
    
    map_logphis = logphis(2^14, t, s, map_est...)[2:end];
    # map_logphis = logphis(maxn, t, s, map_est...)[2:end];
    map_logps = exp.(map_logphis)
    map_logps = reverse(cumsum(reverse(map_logps)))
    stairs!(ax, 1:length(map_logphis) .+ 0.5, map_logps, step=:post, color=Cycled(2), linewidth=3);

    xlims!(ax, 1, maxn+1);
    ylims!(ax, 1/2/mass, 1);
end
fig