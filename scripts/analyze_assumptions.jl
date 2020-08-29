## Solve MergingMDP for different human action spaces

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

@show n_states(mdp)

assumptions = [:wide, :cv, :ca, :cd]
action_spaces = [[-2.0, 0.0, 2.0], [0.0], [2.0], [-2.0]]
# solver = SparseValueIterationSolver(max_iterations=300, verbose=true)

# policies = ValueIterationPolicy[]

# for (i, human_actions) in enumerate(action_spaces)
#     mdp.human_actions = human_actions
#     push!(policies, solve(solver, mdp))
# end

# # mdp.human_actions = [-2.0]
# # policies[end] = solve(solver, mdp)

# @save "/scratch/boutonm/policies.jld2" policies

@load "policies.jld2" policies

## Display policies 
using Plots
using Interact
using Blink


function policy_plot(mdp::MergingMDP, policy::ValueIterationPolicy, ve::Float64=0.0, vo::Float64 = 0.0, assumption::Symbol=:wide)
    xs = mdp.ego_grid[1:end]
    ys = mdp.human_grid
    ae = mdp.acceleration_grid[3]
    z = map(((y,x),) -> value(policy, Float64[x, ve, ae, y, vo]), Iterators.product(ys, xs))
    za = map(((y,x),) -> action(policy, Float64[x, ve, ae, y, vo]), Iterators.product(ys, xs))
    p = [heatmap(xs, ys, z, title="Pr(success) ($assumption)", 
                     aspect_ratio=1, color=:plasma, clim=(0,1), size=(1400, 600), colorbar=false),
        heatmap(xs, ys, za, title="Action Map",  xlabel="Ego longitudinal position",
                     aspect_ratio=1,color=:viridis, clim=(1,7), size=(1400, 600), colorbar=false)
        ]
    scatter!(p[1], [20.0], [20.0], axis=:none, legend=:none, color=:red)
    scatter!(p[2], [20.0], [20.0], axis=:none, legend=:none, color=:red)
    return p
end


w1 = Window()
ui = @manipulate for ve = mdp.velocity_grid, vo = mdp.velocity_grid
    xs = mdp.ego_grid[1:end]
    ys = mdp.human_grid
    ae = mdp.acceleration_grid[3]
    plots = []
    for (i, policy) in enumerate(policies)
        push!.(Ref(plots), policy_plot(mdp, policy, ve, vo, assumptions[i]))
    end
    plot!(plots[1],  ylabel="Other longitudinal position")
    plot!(plots[end], colorbar=true, zticks=String["brake", "-0.2", "-0.1", "0.0", "0.1", "0.2", "release"])
    plot!(plots[end-1], colorbar=true)
    plot(plot([plots[i] for i=1:2:7]..., layout=(1, 4)),
         plot([plots[i] for i=2:2:8]..., layout=(1, 4)), layout=(2,1))
end
body!(w1, ui)


## Visualize policy running

policy = policies[3]
hr = HistoryRecorder(rng = rng, max_steps=200)
rng = MersenneTwister(1)
s0 = [mdp.ego_grid[1], mdp.velocity_grid[5], mdp.acceleration_grid[3], mdp.human_grid[1], mdp.velocity_grid[3]]
hist = simulate(hr, mdp, policy, s0)

w3 = Window()
ui = @manipulate for step in 1:n_steps(hist)
    s = state_hist(hist)[step]
    a = action_hist(hist)[step]
    v = MergingViz(mdp=mdp, s=s, a=a, action_values = actionvalues(policy, s))
    AutomotiveVisualization.render([v], cam=StaticCamera(VecE2(0.0, 0.0), 20.0))
end
body!(w3, ui)
