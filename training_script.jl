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
using ArgParse
using BSON
using CSV
includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("overlays.jl")


s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        help = "specify the random seed"
        arg_type = Int
        default = 1
    "--training_steps"
        help = "specify the number of training steps"
        arg_type = Int
        default = 10_000
    "--eps_fraction"
        help = "Fraction of the training set to use to decay epsilon from 1.0 to eps_end"
        arg_type = Float64
        default = 0.5
    "--eps_end"
        help = "epsilon value at the end of the exploration phase"
        arg_type = Float64
        default = 0.01
    "--learning_rate"
        help = "learning rate for DQN"
        arg_type = Float64
        default = 1e-3
    "--target_update_freq"
        help = "target update frequency for DQN"
        default = 5000
    "--logdir"
        help = "Directory in which to save the model and log training data"
        arg_type = String
        default = "log"
    "--n_eval"
        help = "Number of episodes for evaluation"
        arg_type = Int64
        default = 1000
    "--n_cars"
        help = "Number of cars on the main road"
        arg_type = Int64
        default = 5
    "--cooperation"
        help = "whether the ego vehicle observe the cooperation level or not"
        action = :store_true
    "--recurrent"
        help = "whether to use an RNN in DQN"
        action = :store_true
end
parsed_args = parse_args(s)

seed = parsed_args["seed"]
Random.seed!(seed)
rng = MersenneTwister(seed)


mdp = GenerativeMergingMDP(n_cars_main=parsed_args["n_cars"], 
                           observe_cooperation=parsed_args["cooperation"])

s0 = initialstate(mdp, rng)

svec = convert_s(Vector{Float64}, s0, mdp)

input_dims = length(svec)

if parsed_args["recurrent"]
    model = Chain(LSTM(input_dims, 64), Dense(64, 32, relu), Dense(32, n_actions(mdp)))
else
    model = Chain(Dense(input_dims, 64, relu), Dense(64, 32, relu), Dense(32, n_actions(mdp)))
end
solver = DeepQLearningSolver(qnetwork = model, 
                      max_steps = parsed_args["training_steps"],
                      eps_fraction = parsed_args["eps_fraction"],
                      eps_end = parsed_args["eps_end"],
                      eval_freq = 10_000,
                      save_freq = 10_000,
                      target_update_freq = parsed_args["target_update_freq"],
                      batch_size = 32, 
                      learning_rate = parsed_args["learning_rate"],
                      train_start = 1000,
                      log_freq = 1000,
                      num_ep_eval = 1000,
                      double_q = true,
                      dueling = false,
                      prioritized_replay = true,
                      verbose = true, 
                      recurrence = parsed_args["recurrent"],
                      rng = rng,
                      logdir = parsed_args["logdir"])


policy = solve(solver, mdp)

simlist = [Sim(mdp, policy,
rng=MersenneTwister(i), max_steps=200) for i=1:parsed_args["n_eval"]];

res = run_parallel(simlist) do sim, hist
    return [:steps=>n_steps(hist), :dreward=>discounted_reward(hist), :reward=>undiscounted_reward(hist)]
end

n_collisions = sum(res[:reward] .< 0.0)
avg_steps = mean(res[:steps])
avg_dreward = mean(res[:dreward])
avg_reward = mean(res[:reward])

println("Collisions ", n_collisions)
println("Avg steps ", avg_steps)
println("avg disc reward ", avg_dreward)
println("avg reward ", avg_reward)

BSON.@save joinpath(parsed_args["logdir"], "policy_$(parsed_args["cooperation"]).bson") policy

BSON.@save "results.bson" n_collisions avg_steps  avg_dreward avg_reward