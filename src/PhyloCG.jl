module PhyloCG
using Roots: find_zero, Bisection
using Optim: optimize, NelderMead
using DifferentialEquations
using Phylo
using Random
using Distributions: Normal, Beta, Gamma, Truncated,
                     logpdf, MvNormal, MixtureModel,
                     Exponential, Uniform, Categorical,
                     Geometric
using StatsBase
using StatsFuns: logsumexp, logaddexp, logit,
                 logistic, log1mexp

using SpecialFunctions: loggamma, logabsgamma, polygamma
using HypergeometricFunctions: _₂F₁, _₂F₁maclaurin
using LinearAlgebra: diagind, diagm, I
using DataStructures: DefaultDict
using FFTW: irfft
using ComponentArrays: ComponentArray, labels, getaxes
using ProgressMeter
import MCMCDiagnosticTools: ess_rhat
using Makie: stairs!, barplot!, lines!, hist!, vlines!,
              Figure, Axis, xlims!, ylims!,
              with_theme, theme_minimal, wong_colors, Cycled
import Makie: plot
using Printf: @sprintf

include("hypa12f1.jl")
export hyp2f1a1, continued_hyp2f1a1

include("coarsegrain.jl")
export CGTree, truncate!, truncate,
       isvalid, maxmax, size

include("matrix_exp.jl")
export Lbdi, logphis_exp

include("simulation.jl")
export advance_gillespie_bdi, Pop

include("singularities.jl")
export bdih_singularity, bdihPhi_singularity, bdihPhi_optimal_radius
export _saddlepoint_cond, _saddlepoint_cond2

include("models.jl")
export _bdih!, _bdih, Ubdih, Phi, dPhi
export logphis, slicelogprob, cgtreelogprob, logdensity
export initparams
export _ea, _uv, _uvw, _eab

include("samplers.jl")
export AMWG, AM, LatentSlice, advance!, acceptancerate

include("chain.jl")
export Chain, advance_chain!,
       chainsamples, bestsample,
       ess_rhat, convergence,
       newmaxsubtree!, burn!, burn, nbks, nbsubtrees

include("randompartitions.jl")
export randompartitionAD5, conjugatepartition

include("gof.jl")
export GOFChain, Gstatistic, acceptancerate

include("plotting.jl")
export plot, plotssd!, plotssd, plotssds

end