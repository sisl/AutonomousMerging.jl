using Distributed
addprocs(2)
@everywhere begin
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


includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("overlays.jl")
includet("make_gif.jl");

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(n_cars_main=8, observe_cooperation=true)

solver = DPWSolver(depth = 40,
                   exploration_constant = 1.0,
                   n_iterations = 100, 
                   k_state  = 2.0, 
                   alpha_state = 0.2, 
                   keep_tree = true,
                   enable_action_pw = false,
                   rng = rng, 
                    tree_in_info = true,
                  estimate_value = RolloutEstimator(FunctionPolicy(s->4))
                   )

s0 = initialstate(mdp, rng)

if mdp.observe_cooperation
  policy = solve(solver, mdp)
else
  mdp_noobs = deepcopy(mdp)
  for i=EGO_ID+1:EGO_ID+mdp_noobs.n_cars_main
    mdp_noobs.driver_models[i].c = 0.
  end
  policy = solve(solver, mdp_noobs)
end

end # @everywhere


s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
@time hist = simulate(hr, mdp, policy, s0);
@show n_steps(hist)

includet("visualizer.jl");


make_gif(hist, mdp)


## run many simulations
simlist = [Sim(mdp, policy,
rng=MersenneTwister(i), max_steps=100) for i=1:100];

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