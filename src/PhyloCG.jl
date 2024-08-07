module PhyloCG
using Roots: find_zero, Bisection
using DifferentialEquations, LSODA
using Random
using Distributions
using StatsBase
using StatsFuns: logsumexp, logit, logistic
using SpecialFunctions: loggamma, logabsgamma
using HypergeometricFunctions: _₂F₁
using LinearAlgebra: diagind, diagm
using FFTW: irfft
using ComponentArrays: ComponentArray

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
export logphis, slicelogprob, cgtreelogprob
# export BDIH, BDIH_gaussian

include("mcmc.jl")
export AMWGSampler
export SliceSampler

end