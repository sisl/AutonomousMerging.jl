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
# using Interact
# using Blink
using MCTS
using Reel

includet("environment.jl")
includet("generative_mdp.jl")
includet("masking.jl")
includet("cooperative_IDM.jl")
includet("overlays.jl")

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(n_cars_main=8, driver_type = :random, observe_cooperation = true, initial_ego_velocity=0.0)

policy = RandomPolicy(mdp, rng=rng)

policy = FunctionPolicy(s->4)

s0 = initialstate(mdp, rng)

scene = s0.scene
AutoViz.render(scene, mdp.env.roadway, [IDOverlay()], car_colors=get_car_type_colors(scene, mdp.driver_models), cam=StaticCamera(VecE2(-25.0, -10.0), 6.0))

hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)

include("visualizer.jl");

s = hist.state_hist[end-5]
s = s0
AutoViz.render(s.scene, mdp.env.roadway, 
          SceneOverlay[IDOverlay(),
                       MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
                    #    DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
                    #    DistToMergeOverlay(target_id=2, env=mdp.env),
                       MaskingOverlay(mdp=mdp),
                    #    NeighborsOverlay(EGO_ID),
                    #    CarFollowingStatsOverlay(EGO_ID), 
                        ],
        #   cam=CarFollowCamera(EGO_ID, 5.0),
        cam = StaticCamera(VecE2(-25.0, -10.0), 6.0), 
         car_colors=get_car_type_colors(s0.scene, mdp.driver_models))
        #   car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))

svec = convert_s(Vector{Float64}, s, mdp)
srec = convert_s(AugScene, svec, mdp)

AutoViz.render(srec.scene, mdp.env.roadway, 
          SceneOverlay[IDOverlay(),
                       MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
                       DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
                        DistToMergeOverlay(target_id=2, env=mdp.env),
                       MaskingOverlay(mdp=mdp),
                    #    NeighborsOverlay(EGO_ID),
                    #    CarFollowingStatsOverlay(EGO_ID), 
                        ],
        #   cam=CarFollowCamera(EGO_ID, 5.0), 
        cam = StaticCamera(VecE2(-25.0, -10.0), 6.0),
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))


s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
@time hist = simulate(hr, mdp, policy, s0)



scene = hist.state_hist[2].scene
get_neighbor_rear_along_left_lane(scene, 1, mdp.env.roadway, VehicleTargetPointRear(), VehicleTargetPointFront(), VehicleTargetPointRear())

AutoViz.render(s0.scene, mdp.env.roadway, cam=FitToContentCamera(0.0))
s = s0

sp = generate_s(mdp, s, 4, rng)
AutoViz.render(s.scene, mdp.env.roadway, cam=FitToContentCamera(0.0))
s = sp
w = Window()

ui = @manipulate for step in 1:n_steps(hist)
    s = hist.state_hist[step+1]
    a = hist.action_hist[step]
    AutoViz.render(s.scene, mdp.env.roadway, 
          SceneOverlay[IDOverlay()],
          cam=FitToContentCamera(0.0), 
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
end
body!(w, ui)


env = KMarkovEnvironment(FullyObservablePOMDP(mdp), k=4)

s0 = initialstate(mdp, rng)
input_dims = size(convert_s(Vector{Float32}, s0, mdp))
# input_dims = n_dims(env)
output_dims = n_actions(mdp)
model = Chain(x->flattenbatch(x), Dense(input_dims[1],32), Dense(32, 32, relu),  Dense(32, output_dims))

solver = DeepQLearningSolver(qnetwork = model, 
                      max_steps = 100_000,
                      eps_fraction = 0.5,
                      eps_end = 0.01,
                      eval_freq = 10_000,
                      save_freq = 10_000,
                      target_update_freq = 1000,
                      batch_size = 32, 
                      learning_rate = 1e-3,
                      train_start = 1000,
                      log_freq = 1000,
                      double_q = false,
                      dueling = false,
                      prioritized_replay = true,
                      verbose = true, 
                      rng = rng)

policy = solve(solver, mdp)


solver = DPWSolver(depth = 20,
                   exploration_constant = 1.0,
                   n_iterations = 1000, 
                   k_state  = 2.0, 
                   alpha_state = 0.2, 
                   keep_tree = true,
                   enable_action_pw = false,
                   rng = rng, 
                    tree_in_info = true,
                  estimate_value = RolloutEstimator(FunctionPolicy(s->4))
                   )

policy = solve(solver, mdp)

s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)

frames = Frames(MIME("image/png"), fps=4)
for step in 1:n_steps(hist)
    s = hist.state_hist[step+1]
    a = hist.action_hist[step]
    f = AutoViz.render(s, mdp.env.roadway, 
          SceneOverlay[IDOverlay()],
          cam=FitToContentCamera(0.0), 
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
    push!(frames, f)
end

write("out.gif", frames)

using Reel
using D3Trees

@time a, info = action_info(policy, s0)
t = D3Tree(info[:tree]);
inchrome(t)

D3Tree(policy, s0)



@save "merging_policy_0.jld2" policy

s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)
undiscounted_reward(hist)
ui = @manipulate for step in 1:n_steps(hist)
    s = hist.state_hist[step+1]
    a = hist.action_hist[step]
    AutoViz.render(s, mdp.env.roadway, 
          SceneOverlay[IDOverlay()],
          cam=FitToContentCamera(0.0), 
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
end
body!(w, ui)

## run many simulations
simlist = [Sim(mdp, policy,
               rng=MersenneTwister(i), max_steps=200) for i=1:100]

res = run_parallel(simlist)


## Draft 

s0 = initialstate(mdp, rng)

convert_s(Vector{Float64}, s0, mdp)

render(s0, mdp.env.roadway, SceneOverlay[IDOverlay()],
           cam=FitToContentCamera(0.0), 
           car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))



veh1 = get_by_id(s0, 6)
veh2 = get_by_id(s0, 5)

rel_posf = get_frenet_relative_position(veh1, veh2, mdp.env.roadway)

if rel_posf < 0.0
    rel_posf += get_end(get_lane(veh1)) - veh1.state.posF.s
end

rel_posf = get_frenet_relative_position(veh2, veh1, mdp.env.roadway)


sp = generate_s(mdp, sp, 1, rng)

for veh in s0
    println(veh.state.posF)
end

for veh in sp
    println(veh.state.posF)
end
