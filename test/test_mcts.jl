using Distributed
using ProgressMeter
addprocs(46)
@everywhere begin
using Random
using Printf
using LinearAlgebra
using Distributions
using StatsBase
using ProgressMeter
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
using MCTS
include("environment.jl")
include("generative_mdp.jl")
include("masking.jl")
include("cooperative_IDM.jl")
include("belief_updater.jl")
include("belief_mdp.jl")

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, min_cars=10,max_cars=14, traffic_speed = :mixed, driver_type=:random, 
                           observe_cooperation=false)
for i=2:mdp.max_cars+1
    mdp.driver_models[i] = CooperativeIDM()
end


function POMDPs.reward(mdp::GenerativeMergingMDP, s::AugScene, a::Int64, sp::AugScene)
    ego = get_by_id(s.scene, EGO_ID)
    egop = get_by_id(sp.scene, EGO_ID)
    r = 0.0
    if ego.state.posG.x < egop.state.posG.x
      r += 0.1
    end
    if reachgoal(mdp, egop)
       r += mdp.goal_reward    
    elseif is_crash(sp.scene)
        r += mdp.collision_cost
    end
    if caused_hard_brake(mdp, sp.scene)
        r += mdp.hard_brake_cost
    end
    return r
end
# pomdp = FullyObservablePOMDP(mdp)
# up = MergingUpdater(mdp)
# bmdp = GenerativeBeliefMDP{typeof(pomdp), MergingUpdater, MergingBelief,Int64}(pomdp, up)

#s0 = initialstate(mdp, rng)

#BSON.@load "log12/policy_true.bson" policy
#rlpolicy = NNPolicy(mdp, policy.qnetwork, policy.action_map, policy.n_input_dims)

solver = DPWSolver(depth = 1,
                   exploration_constant = 1.0,
                   n_iterations = 10, 
                   k_state  = 2.0, 
                   alpha_state = 0.2, 
                   keep_tree = true,
                   enable_action_pw = false,
                   rng = rng, 
                    tree_in_info = true,
                  estimate_value = RolloutEstimator(RandomPolicy(mdp, rng=rng))
                   )


if mdp.observe_cooperation
  policy = solve(solver, mdp)
else
  @show keys(mdp.driver_models)
  mdp_noobs = deepcopy(mdp)
  @show keys(mdp_noobs.driver_models)
  #BSON.@load "rl13-backup/policy_false.bson" policy
  #rlpolicy = NNPolicy(mdp_noobs, policy.qnetwork, policy.action_map, policy.n_input_dims)
  #solver.estimate_value = RolloutEstimator(rlpolicy)
  for i=EGO_ID+1:EGO_ID+mdp_noobs.max_cars
    mdp_noobs.driver_models[i].c = 0.5
  end
  policy = solve(solver, mdp_noobs)
end



# s0 = initialstate(mdp, rng)
# hr = HistoryRecorder(rng = rng, max_steps=100)
# @time hist = simulate(hr, mdp, policy, s0);
# @show n_steps(hist)

# include("visualizer.jl");


# make_gif(hist, mdp)


# ## run many simulations

res = @showprogress pmap(x -> quick_evaluation(mdp, policy, x, 1), [MersenneTwister(i) for i=1:100])

function quick_evaluation(mdp::GenerativeMergingMDP, policy::Policy, rng::AbstractRNG, n_eval=1000)
    avg_r, avg_dr, c_rate, avg_steps, t_out = 0.0, 0.0, 0.0, 0.0, 0.0
    for i=1:n_eval
        s0 = initialstate(mdp, rng)
        hr = HistoryRecorder(rng = rng, max_steps=100)
        hist = simulate(hr, mdp, policy, s0)
        avg_r += undiscounted_reward(hist)
        avg_dr += discounted_reward(hist)
        c_rate += hist.reward_hist[end] < 0.0
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

end #@everywhere 

res = @showprogress pmap(x -> quick_evaluation(mdp, policy, MersenneTwister(x), 1), 1:1000)


c_rate = mean(x[3] for x in res)
avg_steps = mean(x[4] for x in res)
t_out = mean(x[5] for x in res)

# avg_r, avg_dr, c_rate, avg_steps, t_out = quick_evaluation(mdp, policy, rng, 100)

println("Collisions ", c_rate*100)
println("Avg steps ", avg_steps)
println("Time outs ", t_out*100)
# println("avg disc reward ", avg_dr)
# println("avg reward ", avg_r)


#simlist = [Sim(mdp, policy,
#rng=MersenneTwister(i), max_steps=100) for i=1:100];

#res = run_parallel(simlist) do sim, hist
#    return [:steps=>n_steps(hist), :dreward=>discounted_reward(hist), :reward=>undiscounted_reward(hist)]
#end

#n_collisions = sum(res[:reward] .< 0.0)
#avg_steps = mean(res[:steps])
#avg_dreward = mean(res[:dreward])
#avg_reward = mean(res[:reward])

#println("Collisions ", n_collisions)
#println("Avg steps ", avg_steps)
#println("avg disc reward ", avg_dreward)
#println("avg reward ", avg_reward)
