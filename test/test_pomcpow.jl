using Revise
using Random
using Printf
using LinearAlgebra
using Distributions
using StatsBase
using StaticArrays
using POMDPs
using AutomotiveVisualization
using AutomotiveDrivingModels
using AutomotivePOMDPs
using POMDPSimulators
using BeliefUpdaters
using ProgressMeter
using POMDPPolicies
using DeepQLearning 
using Flux 
using RLInterface
using POMDPModelTools
using TensorBoardLogger
using POMCPOW
includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("belief_updater.jl")
includet("belief_mdp.jl")
includet("overlays.jl")
includet("make_gif.jl");

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, min_cars=10,max_cars=14, traffic_speed = :mixed, driver_type=:random, 
                           observe_cooperation=true)
for i=2:mdp.max_cars+1
    mdp.driver_models[i] = CooperativeIDM()
end
pomdp = FullyObservablePOMDP(mdp)


solver = POMCPOWSolver(depth = 40,
                   exploration_constant = 1.0,
                   n_iterations = 100, 
                   k_state  = 2.0, 
                   alpha_state = 0.2, 
                   keep_tree = true,
                   enable_action_pw = false,
                   rng = rng, 
                    tree_in_info = true,
                  estimate_value = RolloutEstimator(RandomPolicy(mdp, rng=rng))
                   )

s0 = initialstate(pomdp, rng)

function initial_particles(s0::AugScene)
    driver_type_mask = [6, 9, 12, 15]
    ovec = convert_s(Vector{Float64})
