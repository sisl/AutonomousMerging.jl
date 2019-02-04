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
using AutoViz 
using Printf
using Reel
using DiscreteValueIteration
using SparseArrays
using ProgressMeter
using BSON: @save, @load
using FileIO


includet("environment.jl")
includet("mdp_definition.jl")
includet("helpers.jl")

mdp = MergingMDP()

@sprintf("Number of states: %i ", n_states(mdp))

rm("transition.bson")
trans_mat = DiscreteValueIteration.transition_matrix_a_s_sp(mdp)
@save "transition.bson" trans_mat 
rm("reward.bson")
reward_mat = DiscreteValueIteration.reward_s_a(mdp)
@save "reward.bson" reward_mat


solver = SparseValueIterationSolver(max_iterations=200, verbose=true)
# solver = ValueIterationSolver(max_iterations=30, verbose=true)
policy = solve(solver, mdp)

# post - process policy
for (i, s) in enumerate(states(mdp))
    if isterminal(mdp, s) && !collisioncheck(mdp, s)
        policy.util[i] = 1.0
        policy.qmat[i, :] .= 1.0
    end
end

@save "policy.bson" policy

@load "policy.bson" policy


## Evaluate policy 

rng = MersenneTwister(1)
randpol = RandomPolicy(mdp, rng=rng)
s0 = [mdp.ego_grid[1], mdp.velocity_grid[1], mdp.acceleration_grid[3], mdp.human_grid[1], mdp.velocity_grid[1]]


simlist = [Sim(mdp, policy, s0, metadata=Dict(:policy=>"VI policy"),
               rng=MersenneTwister(i), max_steps=200) for i=1:100]

res = run_parallel(simlist)

@sprintf(" Probability of success at the initial state: %1.4f ", value(policy, s0))
println(" Evaluation of $(res[:policy][1]) ")
@sprintf(" Average reward: %2.4f ", mean(res[:reward]))
@sprintf(" Maximum reward: %2.2f ", maximum(res[:reward]))



using Plots
using Interact
using Blink

w = Window()
hr = HistoryRecorder(rng = rng, max_steps=200)
hist = simulate(hr, mdp, policy, s0)
ui = @manipulate for step in 1:n_steps(hist)
    s = hist.state_hist[step]
    a = hist.action_hist[step]
    v = MergingViz(mdp=mdp, s=s, a=a, action_values = actionvalues(policy, s))
    AutoViz.render([v], cam=StaticCamera(VecE2(0.0, 0.0), 15.0))
end
body!(w, ui)


w = Window()
ui = @manipulate for ve = mdp.velocity_grid, vo = mdp.velocity_grid
    xs = mdp.ego_grid[1:13]
    ys = mdp.human_grid
    ae = mdp.acceleration_grid[3]
    z = map(((y,x),) -> value(policy, [x, ve, ae, y, vo]), Iterators.product(ys, xs))
    p = heatmap(xs, ys, z, aspect_ratio=1, xlabel="Ego longitudinal position", ylabel="Other longitudinal position")
end
body!(w, ui)



ci = findfirst(isequal(0.0), res[:reward])
rng = MersenneTwister(ci)
frames = Frames(MIME("image/png"), fps=4)
AutoViz.render([mdp.env.roadway, s0], cam = StaticCamera(VecE2(0.0,0.0), 15.0))
for step in stepthrough(mdp, policy, s0, "s,a,r,sp", rng=rng, max_steps=100)
    s, a, r, sp = step
    @show s
    @show a
    @show r
    info = @printf("EGO VEHICLE sp = %3.0f lane = %s reward = %1.1f", s[1], mdp.ego_lane.tag, r)
    push!(frames, AutoViz.render([mdp.env.roadway, (sp, a)], cam=StaticCamera(VecE2(0.0,0.0), 15.0)))
    print(" . \n" )
end
write("out.gif", frames)

hist = simulate(hr, mdp, policy, s0)

AutoViz.render([mdp.env.roadway, hist.state_hist[end]], cam = SceneFollowCamera())

undiscounted_reward(hist)


sf = SVector(900.0, 10.0, 0.0, 706.0, 16.0)

isterminal(mdp, sf)

hist.reward_hist[end]

isterminal(mdp, hist.state_hist[end])

frames = Frames(MIME("image/png"), fps=8)
# AutoViz.render([mdp.env.roadway, s0], cam = SceneFollowCamera())
for step in stepthrough(mdp, policy, s0, "s,a,r,sp", rng=rng, max_steps=100)
    s, a, r, sp = step
    @show s
    @show a
    info = @printf("EGO VEHICLE sp = %3.0f lane = %s reward = %1.1f", s[1], mdp.ego_lane.tag, r)
    push!(frames, AutoViz.render([mdp.env.roadway, sp], cam=SceneFollowCamera()))
    print(" . \n" )
end
write("out.gif", frames)

d = transition(mdp, s0, 2)
sp = rand(rng, d)

# DANGER
rm("transition.bson")
rm("reward.bson")



# rng = MersenneTwister(1)

# policy = RandomPolicy(mdp, rng=rng)

# s0 = SVector(700.0, 10.0, 0.0, 650.0, 16.0)
# a = 2
# @time d = transition(mdp, s0, 2);
# sp = rand(rng, d)
# @time r = reward(mdp, sp, a, sp)

# @code_warntype reward(mdp, sp, a, sp)

# hr = HistoryRecorder(max_steps=100, rng=rng)
# hist = simulate(hr, mdp, policy, s0)

# frames = Frames(MIME("image/png"), fps=4)
# for step in stepthrough(mdp, policy, s0, "s,a,r,sp", rng=rng, max_steps=100)
#     s, a, r, sp = step
#     @show s
#     @show a
#     info = @printf("EGO VEHICLE s = %3.0f lane = %s", s[1], mdp.ego_lane.tag)
#     push!(frames, AutoViz.render([mdp.env.roadway, s], cam=SceneFollowCamera()))
#     print(" . \n" )
# end
# write("out.gif", frames)

# AutoViz.render([mdp.env.roadway, hist.state_hist[2]], cam=SceneFollowCamera())
# Reel.set_output_type("gif")
# 
# s = initialstate(mdp, rng)


## Try solving the MDP 


# @show n_states(mdp)

# function testindexing(mdp::MergingMDP)
#     ss = states(mdp)
#     @showprogress for (i, s) in enumerate(ss)
#         si = stateindex(mdp, s)
#         @assert si == i "si = $si , i = $i, s = $s"
#         for a in actions(mdp)
#             d = transition(mdp, s, a)
#             for (sp, w) in weighted_iterator(d)
#                 try 
#                     spi = stateindex(mdp, sp)
#                 catch
#                     throw("Error s = $s, a = $a, sp=$sp")
#                 end
#             end
#         end            
#     end
# end

# function DiscreteValueIteration.transition_matrix_a_s_sp(mdp::MDP)
#     if isfile("transition.bson")
#         @load "transition.bson" trans_mat
#         @info "Loading transition from transition.bson"
#         return trans_mat
#     else
#         # Thanks to zach
#         na = n_actions(mdp)
#         ns = n_states(mdp)
#         transmat_row_A = [Float64[] for _ in 1:n_actions(mdp)]
#         transmat_col_A = [Float64[] for _ in 1:n_actions(mdp)]
#         transmat_data_A = [Float64[] for _ in 1:n_actions(mdp)]

#         @showprogress for s in states(mdp)
#             si = stateindex(mdp, s)
#             for a in actions(mdp, s)
#                 ai = actionindex(mdp, a)
#                 if !isterminal(mdp, s) # if terminal, the transition probabilities are all just zero
#                     td = transition(mdp, s, a)
#                     for (sp, p) in weighted_iterator(td)
#                         if p > 0.0
#                             spi = stateindex(mdp, sp)
#                             push!(transmat_row_A[ai], si)
#                             push!(transmat_col_A[ai], spi)
#                             push!(transmat_data_A[ai], p)
#                         end
#                     end
#                 end
#             end
#         end
#         transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], n_states(mdp), n_states(mdp)) for a in 1:n_actions(mdp)]
#         # Note: not valid for terminal states
#         # @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(n_states(mdp))) for a in 1:n_actions(mdp)) "Transition probabilities must sum to 1"
#         return transmats_A_S_S2
#     end
# end

# function DiscreteValueIteration.reward_s_a(mdp::MDP)
#     if isfile("reward.bson")
#         @load "reward.bson" reward_mat
#         @info "Loading reward from reward.bson"
#         return reward_mat
#     else
#         reward_S_A = fill(-Inf, (n_states(mdp), n_actions(mdp))) # set reward for all actions to -Inf unless they are in actions(mdp, s)
#         @showprogress for s in states(mdp)
#             for a in actions(mdp, s)
#                 td = transition(mdp, s, a)
#                 r = 0.0
#                 for (sp, p) in weighted_iterator(td)
#                     if p > 0.0
#                         r += p*reward(mdp, s, a, sp)
#                     end
#                 end
#                 reward_S_A[stateindex(mdp, s), actionindex(mdp, a)] = r
#             end
#         end
#         return reward_S_A
#     end
# end

# trans_mat = DiscreteValueIteration.transition_matrix_a_s_sp(mdp)
# reward_mat = DiscreteValueIteration.reward_s_a(mdp)

# @save "transition.bson" trans_mat 
# @save "reward.bson" reward_mat


# solver = SparseValueIterationSolver(max_iterations=100, verbose=true)

# policy = solve(solver, mdp)


# ## Visualize policy 
# rng = MersenneTwister(1)
# hr = HistoryRecorder(max_steps=100, rng=rng)
# s0 = SVector(700.0, 10.0, 0.0, 706.0, 16.0)
# hist = simulate(hr, mdp, policy, s0)

# frames = Frames(MIME("image/png"), fps=8)
# AutoViz.render([mdp.env.roadway, s0], cam = SceneFollowCamera())


# for step in stepthrough(mdp, policy, s0, "s,a,r,sp", rng=rng, max_steps=100)
#     s, a, r, sp = step
#     @show s
#     @show a
#     info = @printf("EGO VEHICLE sp = %3.0f lane = %s reward = %1.1f", s[1], mdp.ego_lane.tag, r)
#     push!(frames, AutoViz.render([mdp.env.roadway, sp], cam=SceneFollowCamera()))
#     print(" . \n" )
# end
# write("out.gif", frames)

# AutoViz.render([mdp.env.roadway, hist.state_hist[end-1]], cam=SceneFollowCamera())


# @show actionvalues(policy, hist.state_hist[end-1])


# ## Evaluate policy 

# function evaluate(mdp::MergingMDP, policy::Policy; n_ep::Int=100, max_steps::Int=100, rng=rng)
#     avg_r = 0.0
#     sim = RolloutSimulator(rng=rng, max_steps=max_steps)
#     r = broadcast(x->simulate(sim, mdp, policy))
        

# policy.util


# state_space = collect(states(mdp));

# s = rand(rng, state_space)
# AutoViz.render([mdp.env.roadway, s], cam = SceneFollowCamera())

# collisioncheck(mdp, s)

# ## Rendering 

# ego_posF = Frenet(roadway[LaneTag(MERGE_LANE_ID, 1)], 100.0)
# ego = Vehicle(VehicleState(ego_posF, roadway, 10.0), CARDEF, EGO_ID)
# scene = Scene()
# push!(scene, ego)

# roadway = mdp.env.roadway
# frames = Frames(MIME("image/png"), fps=4)
# for s = 1:10:1000.0
#     ego_posF = Frenet(roadway[LaneTag(MERGE_LANE_ID, 1)], s)
#     ego = Vehicle(VehicleState(ego_posF, roadway, 0.), CARDEF, EGO_ID)
#     scene = Scene()
#     push!(scene, ego)
#     info = @sprintf("EGO VEHICLE s = %3.0f lane = %s", ego.state.posF.s, ego.state.posF.roadind.tag)
#     to = TextOverlay(text=[info], pos=ego.state.posG, font_size = 20, incameraframe=true)
#     c = AutoViz.render(scene, roadway, [to])
#     push!(frames, c)
# end
# write("out.mp4", frames)


# ## Solving the MDP

# n_states(mdp)

