using Revise
using AutonomousMerging
using AutomotiveDrivingModels
using Random
using POMDPs
using POMDPSimulators
using DeepQLearning
using Flux
using BSON
using Reel
using RLInterface
using AutomotiveVisualization
AutomotiveVisualization.set_color_theme(OFFICETHEME)
include("scripts/make_gif.jl")

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, min_cars=16,max_cars=16, traffic_speed = :mixed, driver_type=:binary, initial_ego_velocity=5.0,
                           observe_cooperation=true);
for i=2:mdp.max_cars+1
    mdp.driver_models[i] = CooperativeIDM();
end
# pomdp = FullyObservablePOMDP(mdp)
# up = MergingUpdater(mdp)
# bmdp = GenerativeBeliefMDP{typeof(pomdp), MergingUpdater, MergingBelief,Int64}(pomdp, up)

s0 = initialstate(mdp, rng)

svec = convert_s(Vector{Float64}, s0, mdp)

input_dims = length(svec)

model = Chain(Dense(input_dims, 64, relu), Dense(64, 32, relu), Dense(32, n_actions(mdp)))
# do not run if you just want to load a policy
# solver = DeepQLearningSolver(qnetwork = model, 
#                       max_steps = 1_000_000,
#                       eps_fraction = 0.5,
#                       eps_end = 0.01,
#                       eval_freq = 10_000,
#                       save_freq = 10_000,
#                       target_update_freq = 5000,
#                       batch_size = 32, 
#                       learning_rate = 1e-4,
#                       train_start = 10000,
#                       log_freq = 1000,
#                       num_ep_eval = 1000,
#                       double_q = true,
#                       dueling = false,
#                       prioritized_replay = true,
#                       verbose = true, 
#                       rng = rng)

# policy = solve(solver, mdp)

# BSON.@save "policy_true.bson" policy

# BSON.@load joinpath(solver.logdir,"policy_true.bson") policy
BSON.@load "log15/qnetwork.bson" qnetwork
Flux.loadparams!(model, qnetwork)
policy = NNPolicy(mdp, model, collect(actions(mdp)), 1);

env = MDPEnvironment(mdp)
DeepQLearning.evaluation(solver.evaluation_policy, 
                policy, env,                                  
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)

# policy = MaskedPolicy(policy)

for i=1:10
    s0 = initialstate(mdp, rng);
    hr = HistoryRecorder(rng = rng, max_steps=100);
    hist = simulate(hr, mdp, policy, s0);
    @show undiscounted_reward(hist)
    if undiscounted_reward(hist) > 0.0
        make_gif(hist, mdp, nothing, "out$i.gif")
    end
end


include("visualizer.jl");

i = 36
s = state_hist(hist)[i]
c = AutomotiveVisualization.render(s.scene, mdp.env.roadway,  cam = StaticCamera(VecE2(-25.0, -10.0), 10.0),
                [MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env)],
                car_colors=get_car_type_colors(s0.scene, mdp.driver_models))

write_to_png(c, "snap-bis$i.png")

i = 36
collect(action_hist(hist))[i], state_hist(hist)[i].ego_info.acc, get_by_id(state_hist(hist)[i].scene, 1).state.v

make_gif(hist, mdp)

## run many simulations

function quick_evaluation(mdp::MDP, policy::Policy, rng::AbstractRNG, n_eval=1000)
    avg_r, avg_dr, c_rate, avg_steps, t_out = 0.0, 0.0, 0.0, 0.0, 0.0
    @showprogress for i=1:n_eval
        s0 = initialstate(mdp, rng)
        hr = HistoryRecorder(rng = rng, max_steps=100)
        hist = simulate(hr, mdp, policy, s0)
        avg_r += undiscounted_reward(hist)
        avg_dr += discounted_reward(hist)
        c_rate += undiscounted_reward(hist) <= -1
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

avg_r, avg_dr, c_rate, avg_steps, t_out = quick_evaluation(bmdp, policy, rng, 100)

println("Collisions ", c_rate*100)
println("Avg steps ", avg_steps)
println("Time outs ", t_out*100)
println("avg disc reward ", avg_dr)
println("avg reward ", avg_r)

simlist = [Sim(mdp, policy, max_steps=200, rng=MersenneTwister(i)) for i=1:1000];

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
n_steps(hist)
undiscounted_reward(hist)

include("visualizer.jl");

frames = Frames(MIME("image/png"), fps=4);
for step in 1:n_steps(hist)
    s = state_hist(hist)[step+1]
    a = collect(action_hist(hist))[step]
    f = AutomotiveVisualization.render(s.scene, mdp.env.roadway, 
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


## Env version 

obs = reset!(env)
act = action(policy, op)
vals = actionvalues(policy, op)
op, rew, done, info = step!(env, act)

s = env.state

action(policy, s)
actionvalues(policy, s)

using ProgressMeter
avg_r = 0.0
@showprogress for i=1:1000
    global avg_r
    s0 = initialstate(mdp, rng)
    hr = HistoryRecorder(rng = rng, max_steps=100)
    hist = simulate(hr, mdp, policy, s0)
    avg_r += undiscounted_reward(hist)
end



avg_r/1000
