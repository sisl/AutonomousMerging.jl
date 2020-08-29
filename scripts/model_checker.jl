using Revise
using Random
using LinearAlgebra
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using IterTools
using AutomotiveDrivingModels
using AutomotivePOMDPs: polygon, overlap
using Parameters
using GridInterpolations
using StaticArrays
using AutoUrban
using AutomotiveVisualization 
using Printf
using Reel
using DiscreteValueIteration
using SparseArrays
using ProgressMeter
using JLD2
using FileIO


includet("environment.jl")
includet("mdp_definition.jl")
includet("helpers.jl")

mdp = MergingMDP()
@load "policies.jld2" policies
policy = policies[1] # [:wide, :cv, :ca, :cd]

function safe_actions(svec::Vector{Float64}, threshold::Float64=0.99)
    @assert length(svec) == length(first(states(mdp)))

    ## interpolate state 
    sitp, witp = interpolants(mdp.state_grid, svec)
    normalize!(witp, 1)
    vals = zeros(n_actions(mdp))
    for (i, si) in enumerate(sitp)
        w = witp[i]
        s = ind2x(mdp.state_grid, si)
        if isapprox(svec[4], AbsentPos)
            s[4] = AbsentPos
        end
        vals += w .* actionvalues(policy, s)
    end
    return findall(vals .> threshold)
end

println("Loaded model checker")
