function POMDPs.transition(mdp::GenerativeMergingMDP, s::AugScene, a::Int64)
    # Pr(sp |s, a)
    rng = MersenneTwister(1)
    spred = generate_s(mdp, s, a, rng)
    s_vec = extract_features(mdp, spred)
    sig_p = diagm(0 => [1.0 for i=1:length(s_vec)])
    return  d = MultivariateNormal(s_vec, sig_p)
end

function POMDPs.observation(mdp::GenerativeMergingMDP, a::Int64, s)
    sig_o = diagm(0 => [0.5 for i=1:length(s)])
    return MultivariateNormal(s, sig_o)
end

struct MergingBelief
    o::AugScene
    driver_types::Dict{Int64, Float64}
    # fore::NamedTuple{(:id, :prob),Tuple{Union{Nothing, Int64},Float64}}
    # merge::NamedTuple{(:id, :prob),Tuple{Union{Nothing, Int64},Float64}}
    # fore_main::NamedTuple{(:id, :prob),Tuple{Union{Nothing, Int64},Float64}}
    # rear_main::NamedTuple{(:id, :prob),Tuple{Union{Nothing, Int64},Float64}}
end

function POMDPs.convert_s(t::Type{V}, b::MergingBelief, mdp::GenerativeMergingMDP) where V<:AbstractArray
    ovec = convert_s(t, b.o, mdp)
    fore, merge, fore_main, rear_main = get_neighbors(mdp.env, b.o.scene, EGO_ID)
    if fore.ind != nothing 
        fore_id = b.o.scene[fore.ind].id
        ovec[6] = b.driver_types[fore_id] > 0.5
    end
    if merge.ind != nothing 
        merge_id = b.o.scene[merge.ind].id
        ovec[9] = b.driver_types[merge_id] > 0.5
    end
    if fore_main.ind != nothing 
        fore_main_id = b.o.scene[fore_main.ind].id
        ovec[12] = b.driver_types[fore_main_id] > 0.5
    end
    if rear_main.ind != nothing 
        rear_main_id = b.o.scene[rear_main.ind].id
        ovec[15] = b.driver_types[rear_main_id] > 0.5
    end
    return ovec
end

function belief_weight(t::Type{V}, b::MergingBelief, mdp::GenerativeMergingMDP, state) where V <: AbstractArray
    ovec = convert_s(t, b.o, mdp)
    fore, merge, fore_main, rear_main = get_neighbors(mdp.env, b.o.scene, EGO_ID)
    weight = 1.0
    if fore.ind != nothing 
        fore_id = b.o.scene[fore.ind].id
        ovec[6] = state[1] #b.driver_types[fore_id] > 0.5
        weight *= b.driver_types[fore_id]*state[1] + (1 - b.driver_types[fore_id])*(1 - state[1])
    end
    if merge.ind != nothing 
        merge_id = b.o.scene[merge.ind].id
        ovec[9] = state[2] #b.driver_types[merge_id] > 0.5
        weight *= b.driver_types[merge_id]*state[2] + (1 - b.driver_types[merge_id])*(1 - state[2])
    end
    if fore_main.ind != nothing 
        fore_main_id = b.o.scene[fore_main.ind].id
        ovec[12] = state[3] #b.driver_types[fore_main_id] > 0.5
        weight *= b.driver_types[fore_main_id]*state[3] + (1 - b.driver_types[fore_main_id])*(1 - state[3])
    end
    if rear_main.ind != nothing 
        rear_main_id = b.o.scene[rear_main.ind].id
        ovec[15] = state[4] #b.driver_types[rear_main_id] > 0.5
        weight *= b.driver_types[rear_main_id]*state[4] + (1 - b.driver_types[rear_main_id])*(1 - state[4])
    end
    return ovec, weight
end

const driver_type_states =  collect(Iterators.product([[0,1] for i=1:4]...))

# function POMDPs.action(policy::NNPolicy{GenerativeMergingMDP}, b::MergingBelief)
#     vals = zeros(Float32,n_actions(policy.problem))
#     for state in driver_type_states
#         ovec, weight = belief_weight(Vector{Float32}, b, policy.problem, state)
#         vals += weight*actionvalues(policy, ovec)
#     end
#     @show vals
#     return argmax(vals[:])
# end

struct MergingUpdater <: Updater
    # must set observe_cooperation to false 
    # must fill in all the driver models and make a deepcopy
    mdp::GenerativeMergingMDP

    function MergingUpdater(mdp::GenerativeMergingMDP)
        mdp_ = deepcopy(mdp)
        mdp_.observe_cooperation = false 
        for i=1:mdp.max_cars 
            mdp_.driver_models[i+1] = CooperativeIDM()
        end
        return new(mdp_)
    end
end

function POMDPs.update(up::MergingUpdater, b_old::MergingBelief, a::Int64, o::AugScene)
    # b_neigh = update_neighbors(up.mdp, b_old, o)
    driver_types = deepcopy(b_old.driver_types)
    for i=2:up.mdp.max_cars+1
        update_proba!(up.mdp, b_old, driver_types, a, o, i)
    end
    return MergingBelief(o, driver_types)
    # bfore = update_proba(up.mdp, b_neigh, a, o, :fore)
    # bmerge = update_proba(up.mdp, b_neigh, a, o, :merge)
    # bfore_main = update_proba(up.mdp, b_neigh, a, o, :fore_main)
    # brear_main = update_proba(up.mdp, b_neigh, a, o, :rear_main)
    # return MergingBelief(o, bfore, bmerge, bfore_main, brear_main)
end

function update_neighbors(mdp::GenerativeMergingMDP, b::MergingBelief, o::AugScene)
    fore, merge, fore_main, rear_main = get_neighbors(mdp.env, o.scene, EGO_ID)
    bfore = b.fore 
    bmerge = b.merge 
    bforemain = b.fore_main
    brearmain = b.rear_main
    current_neighbors = [bfore.id, bmerge.id, bforemain.id, brearmain.id]
    if fore.ind == nothing
        bfore = (id=nothing, prob=0.5)
    elseif o.scene[fore.ind].id != b.fore.id
        bfore = (id=o.scene[fore.ind].id, prob=0.5)
    end
    if merge.ind == nothing 
        bmerge = (id=nothing, prob=0.5)
    elseif o.scene[merge.ind].id != b.merge.id
        bmerge = (id=o.scene[merge.ind].id, prob=0.5)
    end
    if fore_main.ind == nothing 
        bforemain = (id=nothing, prob=0.5)
    elseif o.scene[fore_main.ind].id != b.fore_main.id 
        bforemain = (id=o.scene[fore_main.ind].id, prob=0.5)
    end
    if rear_main.ind == nothing 
        bforerear = (id=nothing, prob=0.5)
    elseif o.scene[rear_main.ind].id != b.rear_main.id 
        brearmain = (id=o.scene[rear_main.ind].id, prob=0.5)
    end
    return MergingBelief(b.o, bfore, bmerge, bforemain, brearmain)
end

function update_proba!(mdp::GenerativeMergingMDP, b::MergingBelief, driver_types::Dict, a::Int64, o::AugScene, id::Int64)
    sp_vec = extract_features(mdp, o)
    probs = zeros(2)
    sp_vec = extract_features(mdp, o)
    old_prob = driver_types[id]
    if maximum(old_prob) ≈ 1.0
        maximum(old_prob)
        return driver_types
    end
    probs = zeros(2)
    for c in [0, 1]
        mdp.driver_models[id].c = c
        v_des = 5.0
        set_desired_speed!(mdp.driver_models[id], v_des)
        d = transition(mdp, b.o, a)
        probs[Int(c + 1)] += pdf(d, sp_vec)*(c*old_prob + (1 - c)*(1 - old_prob))
    end
    # @printf("probs = %s id=%d \n", probs, id)
    if sum(probs) ≈ 0.0
        probs = [0.5, 0.5]
    end
    normalize!(probs, 1)
    driver_types[id] = probs[2]
    return driver_types
end

# function update_proba(mdp::GenerativeMergingMDP, b::MergingBelief, a::Int64, o::AugScene, neighbor::Symbol)
#     @show neigh = getfield(b, neighbor)
#     if neigh.id == nothing 
#         return (id=nothing, prob=0.5)
#     else
#         @show sp_vec = extract_features(mdp, o)
#         probs = zeros(2)
#         for c in [0, 1]
#             for v_des in [5.0, 10.0, 15.0]
#                 mdp.driver_models[neigh.id].c = c
#                 set_desired_speed!(mdp.driver_models[neigh.id], v_des)
#                 d = transition(mdp, b.o, a)
#                 @show mean(d)
#                 @show probs[Int(c + 1)] += pdf(d, sp_vec)*(c*neigh.prob + (1 - c)*neigh.prob)
#             end
#         end
#         @show normalize!(probs, 1)
#         return (id=neigh.id, prob=probs[2])
#     end
# end

function BeliefUpdaters.initialize_belief(up::MergingUpdater, s0::AugScene)
    driver_types = Dict{Int64, Float64}()
    for i=2:up.mdp.max_cars+1
        driver_types[i] = 0.5
    end
    b0 = MergingBelief(s0, driver_types)
    # b0 = MergingBelief(s0, (id=nothing, prob=0.5), (id=nothing, prob=0.5), (id=nothing, prob=0.5), (id=nothing, prob=0.5))
    # b0 = update_neighbors(up.mdp, b0, s0)
    return b0
end