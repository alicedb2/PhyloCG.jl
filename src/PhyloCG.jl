module PhyloCG
using Roots: find_zero, Bisection
using Optim: optimize, NelderMead
using DifferentialEquations, LSODA
using Random
using Distributions: Normal, Beta, Gamma, Truncated, 
                     logpdf, MvNormal, MixtureModel, 
                     Exponential, Uniform
using StatsBase
using StatsFuns: logsumexp, logit, logistic
using SpecialFunctions: loggamma, logabsgamma, polygamma
using HypergeometricFunctions: _₂F₁
using LinearAlgebra: diagind, diagm, I
using FFTW: irfft
using ComponentArrays: ComponentArray, labels
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
export bdih, bdih!, Ubdih, Phi
export logphis, slicelogprob, cgtreelogprob, logdensity
export initparams
export _ea, _uv

include("samplers.jl")
export AMWG, AM, LatentSlice, advance!

include("chain.jl")
export Chain, advance_chain!
export chainsamples, bestsample, ess_rhat

include("plotting.jl")
export plot, plotssd!, plotssd, plotssds

end