## MDP definition for the merging problem

const MergingState = SVector{5, Float64} # se, ve, ae, so, vo

@with_kw struct MergingMDP <: MDP{MergingState, Int64}
    env::MergingEnvironment = MergingEnvironment()
    cardef::VehicleDef = CARDEF
    dt::Float64 = 0.2
    min_acc::Float64 = 0.0
    pos_res::Float64 = 2.0*M2L
    vel_res::Float64 = 2.0
    acc_res::Float64 = 1.0
    jerk_levels::SVector{5, Float64} = SVector(-0.2, -0.1, 0, 0.1, 0.2)
    human_actions::SVector{3, Float64} =  SVector(-2.0, 0.0, 2.0)
    ego_lane::Lane = env.roadway[LaneTag(MERGE_LANE_ID, 1)]
    other_lane::Lane = env.roadway[LaneTag(MAIN_LANE_ID, 1)]
    ego_grid::RectangleGrid{1} = RectangleGrid(700.0:pos_res:900)
    human_grid::RectangleGrid{1} = RectangleGrid(650.0:pos_res:850) 
    velocity_grid::RectangleGrid{1} = RectangleGrid(10.0:vel_res:20.0)
    acceleration_grid::RectangleGrid{1} = RectangleGrid(-4.0:acc_res:2)
    goal_reward::Float64 = 1.0
    collision_cost::Float64 = 0.0
    discount_factor::Float64 = 1.0
end

POMDPs.discount(mdp::MergingMDP) = mdp.discount_factor

POMDPs.actions(mdp::MergingMDP) = 1:7
POMDPs.n_actions(mdp::MergingMDP) = 7
POMDPs.actionindex(mdp::MergingMDP, a::Int64) = a


function POMDPs.states(mdp::MergingMDP) 
    prod = Iterators.product(mdp.ego_grid.cutPoints[1], 
                             mdp.velocity_grid.cutPoints[1],
                             mdp.acceleration_grid.cutPoints[1],
                             mdp.human_grid.cutPoints[1],
                             mdp.velocity_grid.cutPoints[1])
    vecs = imap(x->MergingState(x...), prod)
end
POMDPs.n_states(mdp::MergingMDP) = reduce(*, length.([mdp.ego_grid, mdp.velocity_grid, mdp.acceleration_grid, mdp.human_grid, mdp.velocity_grid]))

function POMDPs.stateindex(mdp::MergingMDP, s::SVector{5, Float64})
    se, ve, ae, so, vo = s
    sei = findfirst(x->isapprox(x, se), mdp.ego_grid.cutPoints[1])
    vei = findfirst(x->isapprox(x, ve), mdp.velocity_grid.cutPoints[1])
    aei = findfirst(x->isapprox(x, ae), mdp.acceleration_grid.cutPoints[1])
    soi = findfirst(x->isapprox(x, so), mdp.human_grid.cutPoints[1])
    voi = findfirst(x->isapprox(x, vo), mdp.velocity_grid.cutPoints[1])
    return LinearIndices((length(mdp.ego_grid), length(mdp.velocity_grid), length(mdp.acceleration_grid), 
                          length(mdp.human_grid), length(mdp.velocity_grid)))[sei, vei, aei, soi, voi]
end


function POMDPs.reward(mdp::MergingMDP, s::MergingState, a::Int64, sp::MergingState)
    if sp[1] >= mdp.ego_grid.cutPoints[1][end]
        return mdp.goal_reward
    end
    if collisioncheck(mdp, sp)
        return mdp.collision_cost
    end
    return 0.0
end

POMDPs.isterminal(mdp::MergingMDP, s::MergingState) = s[1] >= mdp.ego_grid.cutPoints[1][end] || collisioncheck(mdp, s)

function POMDPs.transition(mdp::MergingMDP, s::MergingState, a::Int64)
    sitp, switp, vitp, vwitp, acc_itp, acc_witp = ego_transition(mdp, s, a)
    so, sow, vo, vow = other_transition(mdp, s)
    states = map(x -> SVector(x...), Iterators.product(sitp, vitp, acc_itp, so, vo))
    weights = map(x->reduce(*,x), Iterators.product(switp, vwitp, acc_witp, sow, vow))
    tot_weights = sum(weights)
    weights = map(x->reduce(*,x)/tot_weights, Iterators.product(switp, vwitp, acc_witp, sow, vow) )
    return SparseCat(states, weights)
end

function ego_transition(mdp::MergingMDP, s::MergingState, a::Int64)
    se, v, acc, so, vo = s 

    # ego vehicle update
    acc_ = acc
    if a == 1
        acc_ = mdp.min_acc
    elseif a == 7
        acc_ = 0.
    else 
        acc_ += mdp.jerk_levels[a - 1]
    end
    v_ = v + acc_ * mdp.dt 
    s_ = se + v_*mdp.dt
    sitp, switp = interpolants(mdp.ego_grid, SVector(s_))
    sitp = broadcast(x-> ind2x(mdp.ego_grid, x)[1], sitp)
    vitp, vwitp = interpolants(mdp.velocity_grid, SVector(v_))
    vitp = broadcast(x-> ind2x(mdp.velocity_grid, x)[1], vitp)
    acc_itp, acc_witp = interpolants(mdp.acceleration_grid, SVector(acc_))
    acc_itp = broadcast(x -> ind2x(mdp.acceleration_grid, x)[1], acc_itp)
    return sitp, switp, vitp, vwitp, acc_itp, acc_witp
end

function other_transition(mdp::MergingMDP, s::MergingState)
    so = Float64[]
    sow = Float64[]
    vo = Float64[]
    vow = Float64[]
    for ah in mdp.human_actions 
        so_itp, so_witp, vo_itp, vo_witp = other_transition(mdp, s, ah)
        so = vcat(so, so_itp)
        sow = vcat(sow, so_witp)
        vo = vcat(vo, vo_itp)
        vow = vcat(vow, vo_witp)
    end
    return so, sow, vo, vow
end

function other_transition(mdp::MergingMDP, s::MergingState, ah::Float64)
    se, v, acc, so, vo = s  
    vo_ = vo + ah* mdp.dt 
    so_ = so + vo_*mdp.dt
    if so_ > maximum(mdp.human_grid.cutPoints[1])
        so_ = minimum(mdp.human_grid.cutPoints[1])
    end
    so_itp, so_witp = interpolants(mdp.human_grid, SVector(so_))
    so_itp = broadcast(x-> ind2x(mdp.human_grid, x)[1], so_itp)
    vo_itp, vo_witp = interpolants(mdp.velocity_grid, SVector(vo_))
    vo_itp = broadcast(x-> ind2x(mdp.velocity_grid, x)[1], vo_itp)
    prob = 1/length(mdp.human_actions)
    return so_itp, so_witp.*prob, vo_itp, vo_witp.*prob
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

function AutoViz.render!(rm::RenderModel, s::MergingState)
    se, v, acc, so, vo = s  
    ego_posG = get_posG(Frenet(mdp.ego_lane, se), mdp.env.roadway)
    ego_acar = ArrowCar(ego_posG.x, ego_posG.y, ego_posG.θ,
                        color=COLOR_CAR_EGO, text="v: $v",
                        length=mdp.cardef.length, width=mdp.cardef.width)
    other_posG = get_posG(Frenet(mdp.other_lane, so), mdp.env.roadway)
    other_acar = ArrowCar(other_posG.x, other_posG.y, other_posG.θ,
                          color=COLOR_CAR_OTHER, text="v: $vo",
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
