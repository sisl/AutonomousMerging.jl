BT = GenerativeBeliefMDP{FullyObservablePOMDP{AugScene,Int64},MergingUpdater,MergingBelief,Int64}
function POMDPs.initialstate(bmdp::BT)
    ImplicitDistribution() do rng
    s0 = initialstate(bmdp.pomdp, rng)
    b0 = initialize_belief(bmdp.updater, s0)
    return b0 
    end
end

function POMDPs.convert_s(t::Type{V}, b::MergingBelief, bmdp::BT) where V<:AbstractArray
    return convert_s(t, b, bmdp.pomdp.mdp)
end

function POMDPs.actionindex(bmdp::BT, a::Int64)
    return actionindex(bmdp.pomdp.mdp, a)
end

function Base.rand(rng::AbstractRNG, b::MergingBelief)
    return b.o
end

POMDPs.isterminal(bmdp::BT, b::MergingBelief) = isterminal(bmdp.pomdp.mdp, b.o)
