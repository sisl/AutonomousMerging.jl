## MDP definition for the merging problem

const MergingState = Vector{Float64} #SVector{5, Float64} # se, ve, ae, so, vo

@with_kw mutable struct MergingMDP <: MDP{MergingState, Int64}
    env::MergingEnvironment = MergingEnvironment()
    cardef::VehicleDef = CARDEF
    dt::Float64 = 0.2
    min_acc::Float64 = -4.0
    pos_res::Float64 = 2.0*M2L
    vel_res::Float64 = 2.0
    acc_res::Float64 = 1.0
    jerk_levels::SVector{5, Float64} = SVector(-0.2, -0.1, 0, 0.1, 0.2)
    human_actions::Vector{Float64} = [0.0]# [-2.0, 0.0, 2.0]
    ego_lane::Lane = env.roadway[LaneTag(MERGE_LANE_ID, 1)]
    other_lane::Lane = env.roadway[LaneTag(MAIN_LANE_ID, 1)]
    ego_grid::LinRange{Float64} = LinRange(0.0, env.merge_lane_length + env.after_merge_length, 40)
    human_grid::LinRange{Float64} = LinRange(0.0, env.merge_lane_length + env.after_merge_length, 40)
    velocity_grid::LinRange{Float64} = LinRange(0.0, env.main_lane_vmax, 10)
    acceleration_grid::LinRange{Float64} = LinRange(-4.0, 2, 4)
    state_grid::RectangleGrid{5} = RectangleGrid(ego_grid, velocity_grid, acceleration_grid, human_grid, velocity_grid)
    goal_reward::Float64 = 1.0
    collision_cost::Float64 = 0.0
    discount_factor::Float64 = 1.0
end

const AbsentPos = -1.0

POMDPs.discount(mdp::MergingMDP) = mdp.discount_factor
POMDPs.actions(mdp::MergingMDP) = 1:7
POMDPs.n_actions(mdp::MergingMDP) = 7
POMDPs.actionindex(mdp::MergingMDP, a::Int64) = a


function POMDPs.states(mdp::MergingMDP) 
    prod = Iterators.product(mdp.ego_grid, 
                             mdp.velocity_grid,
                             mdp.acceleration_grid,
                             vcat(mdp.human_grid, AbsentPos),
                             mdp.velocity_grid)
    vecs = imap(x->Float64[x...], prod)
end
POMDPs.n_states(mdp::MergingMDP) = reduce(*, length.([mdp.ego_grid, mdp.velocity_grid, mdp.acceleration_grid, vcat(mdp.human_grid, AbsentPos), mdp.velocity_grid]))

function POMDPs.stateindex(mdp::MergingMDP, s::MergingState)
    se, ve, ae, so, vo = s
    sei = findfirst(x->isapprox(x, se), mdp.ego_grid)
    vei = findfirst(x->isapprox(x, ve), mdp.velocity_grid)
    aei = findfirst(x->isapprox(x, ae), mdp.acceleration_grid)
    soi = findfirst(x->isapprox(x, so), mdp.human_grid)
    if so == AbsentPos
        soi = length(mdp.human_grid) + 1
    end
    voi = findfirst(x->isapprox(x, vo), mdp.velocity_grid)
    return LinearIndices((length(mdp.ego_grid), length(mdp.velocity_grid), length(mdp.acceleration_grid), 
                          length(mdp.human_grid)+1, length(mdp.velocity_grid)))[sei, vei, aei, soi, voi]
end

function POMDPs.reward(mdp::MergingMDP, s::MergingState, a::Int64, sp::MergingState)
    if first(sp) >= last(mdp.ego_grid)
        return mdp.goal_reward
    end
    if collisioncheck(mdp, sp)
        return mdp.collision_cost
    end
    return 0.0
end

POMDPs.isterminal(mdp::MergingMDP, s::MergingState) = first(s) >= last(mdp.ego_grid) || collisioncheck(mdp, s)

function POMDPs.transition(mdp::MergingMDP, s::MergingState, a::Int64)
    if first(s) >= last(mdp.ego_grid)
        return SparseCat([s], [1.0])
    end
    se, ve, ae = ego_transition(mdp, s, a)
    sitps = MergingState[]
    witps = Float64[]
    if s[4] == AbsentPos
        so = AbsentPos
        vo = first(mdp.velocity_grid)
        sitp, witp = interpolants(mdp.state_grid, SVector(se, ve, ae, so, vo))
        sitp = ind2x.(Ref(mdp.state_grid), sitp)
        for s in sitp
            s[4] = AbsentPos
        end
        sitps = vcat(sitps, sitp)
        witps = vcat(witps, witp)
    else
        for ah in mdp.human_actions
            so, vo = other_transition(mdp, s, ah)
            sitp, witp = interpolants(mdp.state_grid, SVector(se, ve, ae, so, vo))
            sitp = ind2x.(Ref(mdp.state_grid), sitp)
            if so == AbsentPos
                for s in sitp
                    s[4] = AbsentPos
                end
            end
            sitps = vcat(sitps, sitp)
            witps = vcat(witps, witp)
        end
    end
    sitps, witps = remove_doublons(sitps, witps)
    normalize!(witps, 1)
    return SparseCat(sitps, witps)
end

function remove_doublons(s::Vector{S}, w::Vector{Float64}) where S
    ns = S[]
    ws = Float64[]
    for (i, s) in enumerate(s)
        si = findfirst(isequal(s), ns)
        if si != nothing 
            ws[si] += w[i]
        else 
            push!(ns, s)
            push!(ws, w[i])
        end
    end
    return ns, ws
end

function ego_transition(mdp::MergingMDP, s::MergingState, a::Int64)
    se, v, acc, so, vo = s 
    acc_ = acc
    if a == 1 # full brake
        acc_ = mdp.min_acc
    elseif a == 7 # release throttle
        acc_ = 0.0
    else 
        acc_ += mdp.jerk_levels[a - 1]
    end
    v_ = v + acc_ * mdp.dt 
    s_ = se + v_*mdp.dt
    return s_, v_, acc_
end

function other_transition(mdp::MergingMDP, s::MergingState, ah::Float64)
    se, v, acc, so, vo = s  
    vo_ = vo + ah* mdp.dt 
    so_ = so + vo_*mdp.dt
    if so_ >= last(mdp.human_grid)
        so_ = AbsentPos
    end
    return so_, vo_
end

## helpers
function collisioncheck(mdp::MergingMDP, s::MergingState)
    ego_posG = get_posG(Frenet(mdp.ego_lane, s[1]), mdp.env.roadway)
    other_posG = get_posG(Frenet(mdp.other_lane, s[4]), mdp.env.roadway)
    ego_p = polygon(ego_posG, mdp.cardef)
    other_p = polygon(other_posG, mdp.cardef)
    return overlap(ego_p, other_p)
end

function getegostate(mdp::MergingMDP, s::MergingState)
    ego_posF = Frenet(mdp.ego_lane, s[1])
    return VehicleState(ego_posF, mdp.env.roadway, s[2])
end

function getotherstate(mdp::MergingMDP, s::MergingState)
    other_posF = Frenet(mdp.other_lane, s[4])
    return VehicleState(other_posF, mdp.env.roadway, s[5])
end


## rendering 

@with_kw struct MergingViz
    mdp::MergingMDP = MergingMDP()
    s::MergingState =  [mdp.ego_grid[1], mdp.velocity_grid[1], mdp.acceleration_grid[1], mdp.human_grid[1], mdp.velocity_grid[1]]
    a::Union{Nothing, Int64} = nothing
    action_values::Union{Nothing, Vector{Float64}} = nothing
end

function AutoViz.render!(rm::RenderModel, v::MergingViz)
    render!(rm , mdp.env.roadway)
    se, ve, acc, so, vo = v.s
    text_ego = @sprintf("v: %2.2f", ve)
    act = v.a
    if act != nothing
        as = "a = "
        if act == 1
            as *= "full brake"
        elseif act == 7
            as *= "release"
        else
            as *=  @sprintf("%1.2f", mdp.jerk_levels[act - 1])
        end
        text_ego *= ", " * as
    end
    av = v.action_values 
    if av != nothing 
        text_ego *= "Probas " * reduce(*, @sprintf("%1.2f, ", pa) for pa in av)
    end
    ego_posG = get_posG(Frenet(mdp.ego_lane, se), mdp.env.roadway)
    ego_acar = ArrowCar(ego_posG.x, ego_posG.y, ego_posG.θ,
                        color=COLOR_CAR_EGO, text=text_ego,
                        length=mdp.cardef.length, width=mdp.cardef.width)
    other_posG = get_posG(Frenet(mdp.other_lane, so), mdp.env.roadway)
    if so == AbsentPos
        other_color = RGB(0.5,0.5,0.5)
    else
        other_color = COLOR_CAR_OTHER
    end
    other_acar = ArrowCar(other_posG.x, other_posG.y, other_posG.θ,
                          color=other_color, text="v: $vo",
                          length=mdp.cardef.length, width=mdp.cardef.width)
    render!(rm, ego_acar)
    render!(rm, other_acar)
end

function AutoViz.getcenter(s::MergingState)
    se, v, acc, so, vo = s  
    ego_posG = get_posG(Frenet(mdp.ego_lane, se), mdp.env.roadway)
    other_posG = get_posG(Frenet(mdp.other_lane, so), mdp.env.roadway)
    return 0.5*SVector(ego_posG.x + other_posG.x, ego_posG.y + other_posG.y)
end
