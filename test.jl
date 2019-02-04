using Revise 
using POMDPTesting
include("mdp_definition.jl")

function testindexing(mdp::MergingMDP)
    ss = states(mdp)
    @showprogress for (i, s) in enumerate(ss)
        si = stateindex(mdp, s)
        @assert si == i "si = $si , i = $i, s = $s"
        for a in actions(mdp)
            d = transition(mdp, s, a)
            for (sp, w) in weighted_iterator(d)
                try 
                    spi = stateindex(mdp, sp)
                catch
                    throw("Error s = $s, a = $a, sp=$sp")
                end
            end
        end            
    end
end


testindexing(mdp)

trans_prob_consistency_check(mdp)

function POMDPTesting.trans_prob_consistency_check(pomdp::Union{MDP, POMDP})
    # initalize space
    sspace = states(pomdp)
    aspace = actions(pomdp)
    # iterate through all s-a pairs
    for s in sspace
        for a in aspace
            tran = transition(pomdp, s, a)
            p = 0.0
            for sp in sspace
                p += pdf(tran, sp)
            end
            @assert isapprox(p, 1.0) "Transition probability does not sum to unity for state: ", s, " action: ", a, " sums to", p
        end
    end
end