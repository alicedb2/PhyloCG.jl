module PhyloCG
using Roots: find_zero, Bisection
using Optim: optimize, NelderMead
using DifferentialEquations
using Random
using Distributions: Normal, Beta, Gamma, Truncated,
                     logpdf, MvNormal, MixtureModel,
                     Exponential, Uniform
using StatsBase
using StatsFuns: logsumexp, logaddexp, logit, logistic, log1mexp
using SpecialFunctions: loggamma, logabsgamma, polygamma
using HypergeometricFunctions: _₂F₁, _₂F₁maclaurin
using LinearAlgebra: diagind, diagm, I
using FFTW: irfft
using ComponentArrays: ComponentArray, labels, getaxes
using ProgressMeter
import MCMCDiagnosticTools: ess_rhat
using Makie
import Makie: plot

include("hypa12f1.jl")
export hyp2f1a1, continued_hyp2f1a1

include("matrix_exp.jl")
export Lbdi, logphis_exp

include("simulation.jl")
export advance_gillespie_bdi, Pop

include("singularities.jl")
export bdih_singularity, bdihPhi_singularity, bdihPhi_optimal_radius

include("models.jl")
export bdih, bdih!, Ubdih, Phi, _Ubdih
export logphis, slicelogprob, cgtreelogprob, logdensity
export initparams
export _ea, _uv, _uvw, _eab

include("samplers.jl")
export AMWG, AM, LatentSlice, advance!

include("chain.jl")
export Chain, advance_chain!
export chainsamples, bestsample, ess_rhat, burn!, burn

include("plotting.jl")
export plot, plotssd!, plotssd, plotssds

end