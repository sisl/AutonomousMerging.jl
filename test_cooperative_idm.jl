using Revise
using Random
using Printf
using StatsBase
using StaticArrays
using POMDPs
using AutoViz
using AutomotiveDrivingModels
using AutomotivePOMDPs
using POMDPSimulators
using BeliefUpdaters
using POMDPPolicies
using DeepQLearning 
using Flux 
using RLInterface
using POMDPModelTools
# using Interact
# using Blink
using MCTS
using Reel

includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("belief_updater.jl")
includet("rendering.jl")
includet("overlays.jl")
includet("make_gif.jl")

function test_state(s,v=0.0, acc=0.0,)
    ego = Vehicle(vehicle_state(35.0, merge_lane(mdp.env), 5.0, mdp.env.roadway), VehicleDef(), EGO_ID)
    veh1 = Vehicle(vehicle_state(s, main_lane(mdp.env), 4.9, mdp.env.roadway), VehicleDef(), EGO_ID + 1)
    scene = Scene()
    push!(scene, ego)
    push!(scene, veh1)
    return AugScene(scene, (acc=acc,))
end

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, dt=0.5)

mdp.driver_models[2] = CooperativeIDM(c=1.0)
set_desired_speed!(mdp.driver_models[2], 5.0)


s0 = test_state(85.0)


policy = FunctionPolicy(s->7)
for (k,m) in mdp.driver_models
    reset_hidden_state!(m)
end
hr = HistoryRecorder(rng = rng, max_steps=40)
hist = simulate(hr, mdp, policy, s0)

include("visualizer.jl");


mdp.driver_models[2].other_acc = s0.ego_info.acc
observe!(mdp.driver_models[2], s0.scene, mdp.env.roadway, 2)

make_gif(hist, mdp)


generate_s(mdp, s0, 7, rng)
