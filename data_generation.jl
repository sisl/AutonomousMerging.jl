using Revise
using Random
using AutomotiveDrivingModels
using AutoViz
using AutomotivePOMDPs
using AutomotiveSensors
using POMDPs
using POMDPSimulators
using POMDPPolicies
using StaticArrays
using ProgressMeter
using Parameters
using Printf
using JLD2
using Flux
using Flux: mse, batchseq, throttle, @epochs, crossentropy
using Base.Iterators: partition
includet("environment.jl")
includet("cooperative_IDM.jl")
includet("generative_mdp.jl")


const Obs = Vector{Float64}
const Traj = Vector{Vector{Float64}}

function generate_trajectory(mdp::GenerativeMergingMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG)
    s0 = initialstate(mdp, rng)
    svec = convert_s(Vector{Float64}, s0, mdp)
    hr = HistoryRecorder(max_steps=max_steps, rng=rng)
    hist = simulate(hr, mdp, policy, s0)
    # extract data from the history 
    X = Vector{SVector{length(svec), Float64}}(undef, max_steps+1)
    for i=1:max_steps+1
        X[i] = zeros(length(svec))
    end
    fill!(X, zeros(length(svec)))
    X[1:n_steps(hist)+1] = convert_s.(Ref(Vector{Float64}), hist.state_hist, Ref(mdp))
    # build labels 
    labels_dims = mdp.n_cars_main
    driver_types = Float64[mdp.driver_models[i].c for i=2:mdp.n_cars_main+1]
    Y = [driver_types for i=1:(max_steps+1)]
    return X, Y
end

function collect_set(mdp::GenerativeMergingMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG, n_set)
    X_batch = Vector{Traj}(undef, n_set)
    Y_batch = Vector{Traj}(undef, n_set)
    @showprogress for i=1:n_set
        X_batch[i], Y_batch[i] = generate_trajectory(mdp, policy, max_steps, rng)
    end
    return X_batch, Y_batch
end

rng = MersenneTwister(1)


mdp = GenerativeMergingMDP(n_cars_main = 1, observe_cooperation = false)

# @load "mixed_policy.jld2" policy

policy = RandomPolicy(mdp, rng=rng)

max_steps = 40
n_train = 1000
n_val = 100

X_train, Y_train = collect_set(mdp, policy, max_steps, rng, n_train);
X_val, Y_val = collect_set(mdp, policy, max_steps, rng, n_val);


s0 = initialstate(mdp, rng)
svec = convert_s(Vector{Float64}, s0, mdp)
input_dims = length(svec)
output_dims = mdp.n_cars_main
model = Chain(LSTM(input_dims, 64), Dense(64, 32, relu), Dense(32, output_dims, σ))
model = Chain(Dense(input_dims, output_dims, σ)) # logistic regression

function loss(xs, ys)
    l = sum(crossentropy.(model.(xs), ys))
    Flux.truncate!(model)
    Flux.reset!(model)
    return l
end

function accuracy(x, y)
   mean( (model(x) .> 0.5) .== y)
end

function sequence_accuracy(x, y)
    acc = mean(accuracy.(x, y))
    Flux.truncate!(model)
    Flux.reset!(model)
    return acc
end


batch_size = 32
xv = batchseq(X_val)
yv = batchseq(Y_val)
# xt = batchseq.(partition(X_train, batch_size))
# yt = batchseq.(partition(Y_train, batch_size))
xt = batchseq(X_train)
yt = batchseq(Y_train)

evalcb = () -> @printf("eval loss %2.2f | train loss %2.2f | eval acc %2.2f | train acc %2.2f \n ", loss(xv, yv), loss(xt, yt), sequence_accuracy(xv, yv), sequence_accuracy(xt, yt))

opt = Descent(1e-2)
@epochs 1000 Flux.train!(loss, params(model), [(xt, yt)], opt, cb=evalcb)

loss(xt, yt)

sequence_accuracy(xt, yt)

for data in zip(xt, yt)
    @show loss(data...)
end

loss(xt, yt)

(model(xv[1]) .> 0.5) .== yv[1]

sequence_accuracy(xv, yv)

model.(xv)

@time loss(xv, yv)
# loss(X_val, Y_val)

size(X_train)
size(X_train[1])
size(X_train[1][1])

size(batchseq(X_train))
size(batchseq(X_train)[1])
size(batchseq(X_train)[1][1])



endind = findfirst(isequal(zeros(input_dims)), X_train[1])
state_hist = convert_s.(Ref(AugScene), X_train[1][1:endind], Ref(mdp))

using Blink
using Interact
ui = @manipulate for s=1:length(state_hist)
    AutoViz.render(state_hist[s].scene, mdp.env.roadway,  
                   cam = StaticCamera(VecE2(-25.0, -10.0), 6.0))
end


w = Window()
body!(w, ui);

X_train[1][1]

s0 = initialstate(mdp, rng)
AutoViz.render(s0.scene, mdp.env.roadway, cam = StaticCamera(VecE2(-25.0, -10.0), 6.0))