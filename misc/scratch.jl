# using Pkg; Pkg.activate(".")
# using Revise
# using PhyloInverse
# using Plots

# t, s = 1.0, 0.3
# b, d = 3.0, 2.0
# rho, g = 0.9, 0.8
# f = 0.9

# n = 2^11
# ks = (0:n-1)
# logphis_cont = logphis(t, s, n, f=f, b=b, d=d, rho=rho, g=g)
# logphis_matexp = logphis_exp(t, s, truncN=n, f=f, b=b, d=d, rho=rho, g=g, leaky=true)

# # scatter(log2.(ks), logphis_cont, ms=2, msw=0);
# # plot!(log2.(ks), logphis_matexp, linetype=:stepmid, lw=2);
# # ylims!(-100, 0)


# plot();
# for i in 11:15
#     n = 2^i
#     ks = (0:n-1)
#     logphis_cont = logphis(n, t, s, f, b, d, rho, g; gap=1/n)
#     plot!(log2.(ks), logphis_cont, linetype=:stepmid, lw=2);

# end
# ylims!(-100, 0)

# b, d = 1.0, 1.6
# rho, g = 0.1, 0.7
# zs = LinRange(0.0, 4.0, 1000)
# us = Ubdi.(zs.+0.00001im, 3.4, b, d, rho, g)
# bi_pole_0 = rho/b + 1/g
# bdi_pole_0 = rho/b + 1/g + d/b
# A = 1 - g
# B = 1 - 1/g - rho/b
# Δ = sqrt((rho/b + 1/g + d/b)^2 - 4*d/b/g)
# Δ1 = 1/2*(rho/b + 1/g + d/b + Δ)
# Δ2 = 1/2*(rho/b + 1/g + d/b - Δ)
# a0 = 1 - g
# a1 = -(A * Δ1 + B) / Δ
# a2 = (A * Δ2 + B) / Δ
# r = rho * g + (b - d)*(1 - g)


# plot(zs, real(us))
# vline!([Δ1], label="Δ1")
# vline!([Δ2], label="Δ2")
# ylims!(-2, 5)

# pop_ensemble = []
# max_age = 2.0
# b, d, rho, g = 0.2, 0.0, 0.5, 0.3
# for i in 1:10000
#     tk = Pop(0.0, 1)
#     while tk.t < max_age
#         tk = advance_gillespie_bdi(tk, b, d, rho, g; max_age=max_age, max_size=Inf)
#     end
#     push!(pop_ensemble, tk.k)
# end
# mean(pop_ensemble), exp((rho * g/(1 - g) + (b - d)) * max_age)


using PhyloInverse
using CairoMakie


truncN = 2^10

t, s = 1.0, 0.3
f = 0.9
b, d = 1.0, 0.4
rho, g = 0, 0.9
eta, alpha, beta = 0.1, 1.1, 2.0

ssd_complex =    logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta)[2:end]
ssd_matrix = logphis_exp(truncN, t, s, f, b, d, rho, g, eta, alpha, beta)[2:end]

fig = Figure();
ax = Axis(fig[1, 1]; xlabel="k", ylabel="log ϕₖ", xscale=log2);
stairs!(ax, 1:truncN-1, ssd_complex, step=:center, label="Complex integration");
stairs!(ax, 1:truncN-1, ssd_matrix, step=:center, label="Matrix exponentiation");
ylims!(ax, -30, 0)
axislegend(ax)
fig

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="k", ylabel="log ϕₖ", xscale=log2)
for f in 10 .^ LinRange(-3, 0, 5)
    ssd_complex = logphis(truncN, t, s, f, b, d, rho, g, eta, alpha, beta)[2:end]
    stairs!(ax, 1:truncN-1, ssd_complex, step=:center, label="f=$(round(f, digits=3))");
end
ylims!(ax, -30, 0)
axislegend(ax)
fig
