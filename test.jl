using Revise 
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