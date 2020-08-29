using Revise
using Random
using AutomotiveDrivingModels
using AutomotiveVisualization
using AutomotivePOMDPs
using AutomotiveSensors
using POMDPs
using POMDPSimulators
using POMDPPolicies
using StatsBase
using StaticArrays
using ProgressMeter
using Parameters
using Printf
using JLD2
using Flux
using Flux: mse, batchseq, throttle, @epochs
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
    X = Vector{MVector{length(svec), Float64}}(undef, max_steps+1)
    for i=1:max_steps+1
        X[i] = zeros(length(svec))
    end
    fill!(X, zeros(length(svec)))
    X[1:n_steps(hist)+1] = convert_s.(Ref(Vector{Float64}), state_hist(hist), Ref(mdp))
    # build labels
    Y = Vector{Vector{Float64}}(undef, max_steps + 1) 
    for (i,x) in enumerate(X)
        Y[i] = [x[6], x[9], x[12], x[15]]
        x[6] = 0.5
        x[9] = 0.5
        x[12] = 0.5
        x[15] = 0.5
    end
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


mdp = GenerativeMergingMDP(random_n_cars = true, driver_type = :random, observe_cooperation = true)

# @load "mixed_policy.jld2" policy

policy = RandomPolicy(mdp, rng=rng)

max_steps = 40
n_train = 1000
n_val = 100

X_train, Y_train = collect_set(mdp, policy, max_steps, rng, n_train);
X_val, Y_val = collect_set(mdp, policy, max_steps, rng, n_val);


input_dims = length(X_train[1][1])
output_dims = length(Y_train[1][1])
model = Chain(LSTM(input_dims, 64), Dense(64, 32, relu), Dense(32, output_dims, σ))
model = Chain(Dense(input_dims, output_dims, σ)) # logistic regression

function loss(xs, ys)
    # l = sum(Flux.crossentropy.(model.(xs), ys))
    l = mean(Flux.mse.(model.(xs), ys))
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

opt = ADAM(1e-2)
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
    AutomotiveVisualization.render(state_hist[s].scene, mdp.env.roadway,  
                   cam = StaticCamera(VecE2(-25.0, -10.0), 6.0))
end


w = Window()
body!(w, ui);

X_train[1][1]

s0 = initialstate(mdp, rng)
AutomotiveVisualization.render(s0.scene, mdp.env.roadway, cam = StaticCamera(VecE2(-25.0, -10.0), 6.0))
