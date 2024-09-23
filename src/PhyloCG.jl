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
using Makie: stairs!, barplot!, xlims!, ylims!, 
             with_theme, Figure, Axis, theme_minimal,
             wong_colors, Cycled, lines!, hist!
import Makie: plot

include("hypa12f1.jl")
export hyp2f1a1, continued_hyp2f1a1

include("coarsegrain.jl")
export CGTree, popbouquet!, pushbouquet!, isvalid

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
export AMWG, AM, LatentSlice, advance!

include("chain.jl")
export Chain, advance_chain!
export chainsamples, bestsample, ess_rhat, burn!, burn

include("randompartitions.jl")
export randompartitionAD5, conjugatepartition

include("gof.jl")
export GOFChain, Gstatistic

include("plotting.jl")
export plot, plotssd!, plotssd, plotssds

end