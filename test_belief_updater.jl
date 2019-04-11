using LinearAlgebra
using Distributions
using Revise
using Random
using Printf
using StatsBase
using StaticArrays
using POMDPs
using BeliefUpdaters
using AutoViz
using AutomotiveDrivingModels
using AutomotivePOMDPs
using POMDPSimulators
using POMDPPolicies
using DeepQLearning 
using Flux 
using RLInterface
using POMDPModelTools
using TensorBoardLogger
using Distributions
# using Interact
# using Blink
using MCTS
using Reel

# LOGGER = Logger("log1", overwrite=true)
includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("belief_updater.jl")
includet("overlays.jl")
includet("rendering.jl")
includet("make_gif.jl")

rng = MersenneTwister(1)

# mdp = GenerativeMergingMDP(n_cars_main=5, observe_cooperation=false)


function test_state(s,v=0.0, acc=0.0,)
    ego = Vehicle(vehicle_state(35.0, merge_lane(mdp.env), 5.0, mdp.env.roadway), VehicleDef(), EGO_ID)
    veh1 = Vehicle(vehicle_state(s, main_lane(mdp.env), 4.9, mdp.env.roadway), VehicleDef(), EGO_ID + 1)
    veh2 = Vehicle(vehicle_state(s-15, main_lane(mdp.env), 4.9, mdp.env.roadway), VehicleDef(), EGO_ID + 2)
    scene = Scene()
    push!(scene, ego)
    push!(scene, veh1)
    push!(scene, veh2)
    return AugScene(scene, (acc=acc,))
end

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, min_cars=10, max_cars=14, observe_cooperation=true)

mdp.driver_models[2] = CooperativeIDM(c=0.0)
set_desired_speed!(mdp.driver_models[2], 5.0)


s0 = test_state(85.0)

up = MergingUpdater(mdp)


s0 = initialstate(mdp, rng)
b0 = initialize_belief(up, s0)

super_render(mdp, s0, b0)
s = s0
b = b0
sp, r = generate_sr(mdp, s, 7, rng)
bp = update(up, b, 7, sp)
super_render(mdp, sp, bp)

s = sp
b = deepcopy(bp)
@show b.driver_types
a = 7
sp, r = generate_sr(mdp, s, a, rng)
bp = update(up, b, a, sp)
super_render(mdp, sp, bp)


pomdp = FullyObservablePOMDP(mdp)
up = MergingUpdater(mdp)
s0 = initialstate(mdp, rng)
b0 = initialize_belief(up, s0)
hr = HistoryRecorder(rng=rng, max_steps = 100)

hist= simulate(hr, pomdp, policy, up, b0, s0);


include("visualizer.jl")


includet("make_gif.jl")

make_gif(hist, mdp, hist.belief_hist)

function quick_evaluation(pomdp::POMDP, policy::Policy, rng::AbstractRNG, n_eval=1000)
    avg_r, avg_dr, c_rate, avg_steps, t_out = 0.0, 0.0, 0.0, 0.0, 0.0
    @showprogress for i=1:n_eval
        s0 = initialstate(pomdp, rng)
        up = MergingUpdater(pomdp.mdp)
        hr = HistoryRecorder(rng = rng, max_steps=100)
        hist = simulate(hr, pomdp, policy, up, b0, s0)
        avg_r += undiscounted_reward(hist)
        avg_dr += discounted_reward(hist)
        c_rate += undiscounted_reward(hist) <= pomdp.mdp.collision_cost
        t_out += n_steps(hist) >= hr.max_steps ? 1.0 : 0.0
        avg_steps += n_steps(hist)
    end
    avg_r /= n_eval 
    avg_dr /= n_eval 
    c_rate /= n_eval
    avg_steps /= n_eval
    t_out /= n_eval
    return avg_r, avg_dr, c_rate, avg_steps, t_out
end
policy = NNPolicy(mdp, policy.qnetwork, policy.action_map, policy.n_input_dims)
avg_r, avg_dr, c_rate, avg_steps, t_out = quick_evaluation(pomdp, policy, rng, 1000)

println("Collisions ", c_rate*100)
println("Avg steps ", avg_steps)
println("Time outs ", t_out*100)
println("avg disc reward ", avg_dr)
println("avg reward ", avg_r)




spbackup = deepcopy(sp)
sbackup = deepcopy(s)
bbackup = deepcopy(b)


neighbor = :fore 
o = sp

