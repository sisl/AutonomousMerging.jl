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
using BeliefUpdaters
using TensorBoardLogger
using ArgParse
using BSON
using CSV
using LinearAlgebra
using ProgressMeter
includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("belief_updater.jl")
includet("overlays.jl")

BLAS.set_num_threads(8)

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
        default = 1e-4
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
    "--max_cars"
        help = "Number of cars on the main road"
        arg_type = Int64
        default = 12
    "--min_cars"
        help = "Number of cars on the main road"
        arg_type = Int64
        default = 0
    "--cooperation"
        help = "whether the ego vehicle observe the cooperation level or not"
        action = :store_true
    "--driver_type"
        help = "the distribution of driver types, choose among cooperative, aggressive or random"
        arg_type = String
        default = "random"
    "--traffic_speed"
        arg_type = String
        default = "mixed"
    "--recurrent"
        help = "whether to use an RNN in DQN"
        action = :store_true
    "--load"
        help = "load a pretrain policy"
        arg_type = Union{Nothing, String}
        default = nothing
    "--collision_cost"
        help = "collision cost in the mdp formulation"
        arg_type = Float64
        default = -1.0
end
parsed_args = parse_args(s)

seed = parsed_args["seed"]
Random.seed!(seed)
rng = MersenneTwister(seed)


mdp = GenerativeMergingMDP(random_n_cars = true,
                            collision_cost = parsed_args["collision_cost"],
                            max_cars = parsed_args["max_cars"],
                            min_cars = parsed_args["min_cars"],
                           driver_type = Symbol(parsed_args["driver_type"]),
                           traffic_speed = Symbol(parsed_args["traffic_speed"]),
                           observe_cooperation=parsed_args["cooperation"])

s0 = initialstate(mdp, rng)

svec = convert_s(Vector{Float64}, s0, mdp)

input_dims = length(svec)

if parsed_args["recurrent"]
    model = Chain(LSTM(input_dims, 64), Dense(64, 32, relu), Dense(32, n_actions(mdp)))
else
    model = Chain(Dense(input_dims, 64, relu), Dense(64, 32, relu), Dense(32, n_actions(mdp)))
end

if parsed_args["load"] != nothing
    BSON.@load parsed_args["load"] policy
    Flux.loadparams!(model, params(policy.qnetwork))
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


@time policy = solve(solver, mdp)

env = MDPEnvironment(mdp)
DeepQLearning.evaluation(solver.evaluation_policy, 
                policy, env,                                  
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)

function quick_evaluation(mdp::GenerativeMergingMDP, policy::Policy, rng::AbstractRNG, n_eval=1000)
    avg_r, avg_dr, c_rate, avg_steps, t_out = 0.0, 0.0, 0.0, 0.0, 0.0
    @showprogress for i=1:n_eval
        s0 = initialstate(mdp, rng)
        hr = HistoryRecorder(rng = rng, max_steps=100)
        hist = simulate(hr, mdp, policy, s0)
        avg_r += undiscounted_reward(hist)
        avg_dr += discounted_reward(hist)
        c_rate += undiscounted_reward(hist) <= mdp.collision_cost
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

avg_r, avg_dr, c_rate, avg_steps, t_out = quick_evaluation(mdp, policy, rng, 10000)

println("Collisions ", c_rate*100)
println("Avg steps ", avg_steps)
println("Time outs ", t_out*100)
println("avg disc reward ", avg_dr)
println("avg reward ", avg_r)

BSON.@save joinpath(parsed_args["logdir"], "policy_$(parsed_args["cooperation"]).bson") policy

BSON.@save "results.bson" c_rate avg_steps t_out avg_dr avg_r