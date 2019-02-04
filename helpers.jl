function DiscreteValueIteration.transition_matrix_a_s_sp(mdp::MDP)
    if isfile("transition.bson")
        @load "transition.bson" trans_mat
        @info "Loading transition from transition.bson"
        return trans_mat
    else
        # Thanks to zach
        na = n_actions(mdp)
        ns = n_states(mdp)
        transmat_row_A = [Float64[] for _ in 1:n_actions(mdp)]
        transmat_col_A = [Float64[] for _ in 1:n_actions(mdp)]
        transmat_data_A = [Float64[] for _ in 1:n_actions(mdp)]

        @showprogress for s in states(mdp)
            si = stateindex(mdp, s)
            for a in actions(mdp, s)
                ai = actionindex(mdp, a)
                if !isterminal(mdp, s) # if terminal, the transition probabilities are all just zero
                    td = transition(mdp, s, a)
                    for (sp, p) in weighted_iterator(td)
                        if p > 0.0
                            spi = stateindex(mdp, sp)
                            push!(transmat_row_A[ai], si)
                            push!(transmat_col_A[ai], spi)
                            push!(transmat_data_A[ai], p)
                        end
                    end
                end
            end
        end
        transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], n_states(mdp), n_states(mdp)) for a in 1:n_actions(mdp)]
        # Note: not valid for terminal states
        # @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(n_states(mdp))) for a in 1:n_actions(mdp)) "Transition probabilities must sum to 1"
        return transmats_A_S_S2
    end
end

function DiscreteValueIteration.reward_s_a(mdp::MDP)
    if isfile("reward.bson")
        @load "reward.bson" reward_mat
        @info "Loading reward from reward.bson"
        return reward_mat
    else
        reward_S_A = fill(-Inf, (n_states(mdp), n_actions(mdp))) # set reward for all actions to -Inf unless they are in actions(mdp, s)
        @showprogress for s in states(mdp)
            if isterminal(mdp, s)
                reward_S_A[stateindex(mdp, s), :] .= 0.0
            else
                for a in actions(mdp, s)
                    td = transition(mdp, s, a)
                    r = 0.0
                    for (sp, p) in weighted_iterator(td)
                        if p > 0.0
                            r += p*reward(mdp, s, a, sp)
                        end
                    end
                    reward_S_A[stateindex(mdp, s), actionindex(mdp, a)] = r
                end
            end
        end
        return reward_S_A
    end
end
