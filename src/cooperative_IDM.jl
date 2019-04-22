"""
    CooperativeIDM <: DriverModel{LaneFollowingAccel}
The cooperative IDM (c-IDM) is a rule based driver model for merging scenarios. 
It controls the longitudinal actions of vehicles on the main lane. 
A cooperation level `c` controls how the vehicles reacts to the merging vehicle. 
When `c=0` the vehicle ignores the merging vehicle. When `c=1` the vehicle considers the merging 
vehicle as its front car when TTC_ego > TTC_mergin_vehicle. When it is not considering the merging vehicle, 
the car follows the IntelligentDriverModel.

# Fields 
    - `env::MergingEnvironment = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/6)` the merging environment
    - `idm::IntelligentDriverModel = IntelligentDriverModel(v_des = env.main_lane_vmax, d_cmf = 2.0, d_max=2.0, T = 1.5, s_min = 2.0, a_max = 2.0)` the default IDM
    - `c::Float64 = 0.0` the cooperation level
    - `fov::Float64 = 20.0` [m] A field of view, the merging vehicle is not considered if it is further than `fov`
"""
@with_kw mutable struct CooperativeIDM <: DriverModel{LaneFollowingAccel}
    env::MergingEnvironment = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/6)
    idm::IntelligentDriverModel = IntelligentDriverModel(v_des = env.main_lane_vmax, 
                                                         d_cmf = 2.0, 
                                                         d_max=2.0,
                                                         T = 1.5,
                                                         s_min = 2.0,
                                                         a_max = 2.0)
    c::Float64 = 0.0 # cooperation level
    fov::Float64 = 20.0 # when to consider merge car
    # internals
    a::Float64 = 0.0
    a_merge::Float64 = 0.0
    a_idm::Float64 = 0.0
    other_acc::Float64 = 0.0
    s_des::Float64 = idm.s_min
    dist_at_merge::Float64 = 0.0
    ego_ttm::Float64 = 0.0
    veh_ttm::Float64 = 0.0
    front_car::Bool = false
    consider_merge::Bool = false
end

Base.rand(model::CooperativeIDM) = LaneFollowingAccel(model.a)
function AutomotiveDrivingModels.reset_hidden_state!(model::CooperativeIDM)
    reset_hidden_state!(model.idm)
    model.a = 0.0
    model.a_merge = 0.0
    model.a_idm = 0.0
    model.other_acc = 0.0
    model.s_des = model.idm.s_min
    model.dist_at_merge = 0.0
end

function AutomotiveDrivingModels.set_desired_speed!(model::CooperativeIDM, vdes::Float64)
    set_desired_speed!(model.idm, vdes)
end

function AutomotiveDrivingModels.observe!(model::CooperativeIDM, scene::Scene, roadway::Roadway, egoid::Int64)
    ego_ind = findfirst(egoid, scene)
    # @printf("OBSERVE COOPERATIVE IDM \n")
    ego = scene[ego_ind]
    fore = get_neighbor_fore_along_lane(scene, ego_ind, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    if fore.ind == nothing # uses the first vehicle on the lain as neighbor

        vehmin, vehind = findfirst_lane(scene, main_lane(model.env))
        headway = get_end(main_lane(model.env)) - ego.state.posF.s + vehmin.state.posF.s
        track_longitudinal!(model.idm, ego.state.v, vehmin.state.v, headway)
        # @printf("No neighbor, ID: %d | neighbor %d | headway %2.1f \n", egoid, scene[vehind].id, headway)
    else
        # @printf("ID: %d | neighbor %d \n", egoid, scene[fore.ind].id)
        observe!(model.idm, scene, roadway, egoid)
    end
    a_idm = model.idm.a
    model.a_idm = a_idm 
    veh = find_merge_vehicle(model.env, scene)
    if veh == nothing || veh.state.posF.s < model.fov
        # println("No merge vehicle")
        model.a = model.a_idm
    else
        model.other_acc = 0.0
        model.a = 0.0
        model.a
        ego_ttm = time_to_merge(model.env, ego, model.a)
        veh_ttm = time_to_merge(model.env, veh, model.other_acc)
        model.ego_ttm = ego_ttm
        model.veh_ttm = veh_ttm
        if ( ego_ttm < 0.0 || ego_ttm < veh_ttm || veh_ttm == Inf)
            # println("Ego TTM < Merge TTM, ignoring")
            ego_ttm < veh_ttm
            model.a = model.a_idm
            model.consider_merge = false
            model.front_car = false
        else
            model.consider_merge = true 
            if veh_ttm < model.c*ego_ttm 
                model.front_car = true
                headway = distance_projection(model.env, veh) - distance_projection(model.env, ego)
                headway -= veh.def.length 
                v_oth = veh.state.v
                v_ego = ego.state.v
                # @show "tracking front car"
                track_longitudinal!(model.idm, v_ego, v_oth, headway)
                model.a_merge = model.idm.a
                model.a = min(model.a_merge, model.a_idm)
            else 
                model.a = model.a_idm 
                model.front_car = false
            end
        end
    end
    return model
end

"""
    findfirst_lane(scene::Scene, lane::Lane)
find the first vehicle on the lane (in terms of longitudinal position)
"""
function findfirst_lane(scene::Scene, lane::Lane)
    s_min = Inf
    vehmin = nothing
    vehind = nothing 
    for (i, veh) in enumerate(scene)
        if veh.state.posF.roadind.tag == lane.tag && veh.state.posF.s < s_min
            s_min = veh.state.posF.s
            vehmin = veh
            vehind = i
        end
    end
    return vehmin, vehind
end
