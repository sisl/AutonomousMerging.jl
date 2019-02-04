using Revise
using Random
using POMDPs
using LinearAlgebra
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
using AutoViz 
using ProgressMeter
using BenchmarkTools
using Profile
using ProfileView

includet("environment.jl")
includet("mdp_definition.jl")

const NTESTS = 1000

mdp = MergingMDP()
statespace = collect(states(mdp));
actionspace = actions(mdp)
rng = MersenneTwister(1)
test_states = rand(rng, statespace, NTESTS)

function benchmark_transition(mdp::MergingMDP, test_states::Vector{MergingState}, actionspace::UnitRange{Int64})
    for s in test_states 
        for a in actionspace
            d = transition(mdp, s, a)
        end
    end
end

println("Starting transition benchmark")

@btime benchmark_transition($mdp, $test_states, $actionspace)

Profile.clear()
@profile benchmark_transition(mdp, test_states, actionspace)

ProfileView.view()