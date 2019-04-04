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
using POMDPPolicies
using DeepQLearning 
using Flux 
using RLInterface
using POMDPModelTools
using TensorBoardLogger
# using Interact
# using Blink
using MCTS
using Reel

# LOGGER = Logger("log1", overwrite=true)
includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("overlays.jl")

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(n_cars_main=8, observe_cooperation=true,
                        default_driver_model = EgoDriver(LaneFollowingAccel(0.0)))
default_driver_model = CooperativeIDM()

s0 = initialstate(mdp, rng)

svec = convert_s(Vector{Float64}, s0, mdp)

input_dims = length(svec)


model = Chain(Dense(input_dims, 64, relu), Dense(64, 32, relu), Dense(32,32,relu), Dense(32, n_actions(mdp)))
solver = DeepQLearningSolver(qnetwork = model, 
                      max_steps = 1_000_000,
                      eps_fraction = 0.5,
                      eps_end = 0.01,
                      eval_freq = 10_000,
                      save_freq = 10_000,
                      target_update_freq = 10000,
                      batch_size = 32, 
                      learning_rate = 1e-3,
                      train_start = 1000,
                      log_freq = 1000,
                      num_ep_eval = 1000,
                      double_q = true,
                      dueling = false,
                      prioritized_replay = true,
                      verbose = true, 
                      rng = rng)
# solver.exploration_policy = masked_linear_epsilon_greedy(solver.max_steps, solver.eps_fraction, solver.eps_end)
# solver.evaluation_policy = masked_evaluation()

policy = solve(solver, mdp)

env = MDPEnvironment(mdp)
DeepQLearning.evaluation(solver.evaluation_policy, 
                policy, env,                                  
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)

# policy = MaskedPolicy(policy)
s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)

include("visualizer.jl");

## run many simulations
simlist = [Sim(mdp, policy,
rng=MersenneTwister(i), max_steps=200) for i=1:1000];

res = run_parallel(simlist) do sim, hist
    return [:steps=>n_steps(hist), :dreward=>discounted_reward(hist), :reward=>undiscounted_reward(hist)]
end

n_collisions = sum(res[:reward] .< 0.0)
avg_steps = mean(res[:steps])
avg_dreward = mean(res[:dreward])
avg_reward = mean(res[:reward])
collision_ind = findfirst(res[:reward] .< 0.0)
rng = MersenneTwister(collision_ind)

s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)

include("visualizer.jl");

frames = Frames(MIME("image/png"), fps=4);
for step in 1:n_steps(hist)
    s = hist.state_hist[step+1]
    a = hist.action_hist[step]
    f = AutoViz.render(s.scene, mdp.env.roadway, 
          SceneOverlay[
                       IDOverlay(),
                       MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
                       DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
                       MaskingOverlay(mdp=mdp)
                       #    NeighborsOverlay(EGO_ID),
                    #    CarFollowingStatsOverlay(EGO_ID), 
                        ],
          cam=CarFollowCamera(EGO_ID, 10.0), 
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
    push!(frames, f)
end

write("out.gif", frames)