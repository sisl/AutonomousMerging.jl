# cannot go over the speed limit 
# cannot go backward
function speed_limit_actions(mdp::GenerativeMergingMDP, s::AugScene)
    valid_acts = Int64[]
    sizehint!(valid_acts, 7)
    ego = get_by_id(s.scene, EGO_ID)
    acc = s.ego_info.acc
    for a in actions(mdp)
        next_v = ego.state.v + action_map(mdp, acc, a).a*mdp.dt   
        if 0 <= next_v <= mdp.env.main_lane_vmax
            push!(valid_acts, a)
        end
    end
    return valid_acts     
end

# jerk constraints
function acceleration_limit_actions(mdp::GenerativeMergingMDP, s::AugScene)
    acc = s.ego_info.acc
    valid_acts = Int64[]
    # if we were hard braking we must release the brake
    # if isapprox(acc, mdp.max_deceleration)
    #     for a in actions(mdp)
    #         if action_map(mdp, 0.0, a).a >= 0.0
    #             push!(valid_acts, a)
    #         end
    #     end
    #     return valid_acts
    # else
    for a in actions(mdp)
        next_acc = action_map(mdp, acc, a).a 
        if abs(next_acc) <= mdp.comfortable_acceleration
            push!(valid_acts, a)
        end
    end
    return valid_acts
    # end
end

function collision_lookahead(mdp::GenerativeMergingMDP, s::AugScene)
    valid_acts = Int64[]
    action_list = LatLonAccel[LatLonAccel(0.,0.0) for veh in s.scene if veh.id != EGO_ID]
    for a in actions(mdp)
        sp = deepcopy(s.scene)
        acc = action_map(mdp, s.ego_info.acc, a)
        tick!(sp, mdp.env.roadway, [acc, action_list...], mdp.dt)
        if !icollision_checker(sp, EGO_ID)
            push!(valid_acts, a)
        end
    end
    return valid_acts
end

function idm_limit_actions(mdp::GenerativeMergingMDP, s::AugScene)
    scene = s.scene
    ego_ind = findfirst(EGO_ID, scene)
    ego = scene[ego_ind]
    acc = s.ego_info.acc
    fore_res = get_front_neighbor(mdp.env, s.scene, EGO_ID)
    if fore_res.ind === nothing 
        return actions(mdp)
    end
    v_ego = scene[ego_ind].state.v
    v_oth = scene[fore_res.ind].state.v
    headway = fore_res.Δs
    track_longitudinal!(mdp.ego_idm, v_ego, v_oth, headway)
    des_acc = mdp.ego_idm.a
    valid_acts = Int64[]
    for a in actions(mdp)
        next_acc = action_map(mdp, acc, a)
        if next_acc.a <= des_acc 
            push!(valid_acts, a)
        end
    end
    return valid_acts
end

function safe_actions(mdp::GenerativeMergingMDP, s::AugScene)
    speed = speed_limit_actions(mdp, s)
    # jerk = acceleration_limit_actions(mdp, s)
    collision_free = collision_lookahead(mdp, s)
    available_acts = intersect(speed, collision_free)
    # available_acts = intersect(speed, jerk, collision_free)
    # available_acts = intersect(jerk, speed, collision_free)
    # ego = get_by_id(s.scene, EGO_ID)
    # if get_lane(mdp.env.roadway, ego) == main_lane(mdp.env)
    #     idm = idm_limit_actions(mdp, s)
    #     available_acts = intersect(speed, jerk, idm, collision_free)
    # end
    # push!(available_acts, HARD_BRAKE)
    # if isempty(available_acts)
        # return Int64[HARD_BRAKE]
    # end
    sort!(available_acts)
    return available_acts
end

function safe_actions(mdp::MDP{S,A}, o::AbstractArray) where {S,A}
    s = POMDPs.convert_s(S, o, mdp)
    return safe_actions(mdp, s)
end

function best_action(acts::Vector{A}, val::AbstractArray{T}, problem::M) where {A, T <: Real, M <: Union{POMDP, MDP}}
    best_a = acts[1]
    best_ai = actionindex(problem, best_a)
    best_val = val[best_ai]
    for a in acts
        ai = actionindex(problem, a)
        if val[ai] > best_val
            best_val = val[ai]
            best_ai = ai
            best_a = a
        end
    end
    return best_a::A
end

function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64)
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        acts = safe_actions(policy.problem, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
        val = actionvalues(policy, obs) #change this
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end


function masked_evaluation() 
    function masked_evaluation_policy(policy::AbstractNNPolicy, env::MDPEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
        avg_r = 0 
        violations = 0
        avg_steps = 0
        for i=1:n_eval
            done = false 
            r_tot = 0.0
            step = 0
            obs = reset!(env)
            DeepQLearning.resetstate!(policy)
            while !done && step <= max_episode_length
                acts = safe_actions(env.problem, obs)
                val = actionvalues(policy, obs)
                act = best_action(acts, val, env.problem)
                obs, rew, done, info = step!(env, act)
                r_tot += rew 
                step += 1
            end
            # println("Episode reward: $r_tot")
            r_tot < 0. ? violations += 1 : nothing
            avg_r += r_tot 
            avg_steps += step
        end
        # log_value(LOGGER, "eval_reward", avg_r/n_eval)
        # log_value(LOGGER, "collisions", violations/n_eval*100)
        # log_value(LOGGER, "avg_steps", avg_steps/n_eval)
        if verbose
            @printf("Evaluation ... Avg Reward %1.2f | Violations (%%) %2.2f | Avg Steps %3.2f", avg_r/n_eval, violations/n_eval*100, avg_steps/n_eval)
        end
        avg_r /= n_eval
        return  avg_r
    end
    return masked_evaluation_policy
end

struct MaskedPolicy{P<:AbstractNNPolicy} <: AbstractNNPolicy
    policy::P
end

function POMDPs.action(p::MaskedPolicy, s)
    acts = safe_actions(p.policy.problem, s)
    val = actionvalues(p.policy, s)
    act = best_action(acts, val, p.policy.problem)
    @assert in(act, acts)
    return act 
end

struct MaskedRandomPolicy <: Policy
    problem::GenerativeMergingMDP
    rng::MersenneTwister
end

function POMDPs.action(p::MaskedRandomPolicy, s)
    acts = safe_actions(p.problem, s)
    return rand(p.rng, acts)
end
