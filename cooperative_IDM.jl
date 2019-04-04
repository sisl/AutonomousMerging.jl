@with_kw mutable struct CooperativeIDM <: DriverModel{LaneFollowingAccel}
    env::MergingEnvironment = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/6)
    ttm_threshold::Float64 = 2.5 # threshold on the time to merge 
    idm::IntelligentDriverModel = IntelligentDriverModel(v_des = env.main_lane_vmax, d_cmf = 2.0, d_max=3.0)
    c::Float64 = 0.0 # cooperation level
    a_min::Float64 = -2.0 # minimum acceleration
    # internals
    a::Float64 = 0.0
    a_merge::Float64 = 0.0
    a_idm::Float64 = 0.0
    other_acc::Float64 = 0.0
    s_des::Float64 = 0.0
    dist_at_merge::Float64 = 0.0
    ego_ttm::Float64 = 0.0
    veh_ttm::Float64 = 0.0
    front_car::Bool = false
end

Base.rand(model::CooperativeIDM) = LaneFollowingAccel(model.a)
function AutomotiveDrivingModels.reset_hidden_state!(model::CooperativeIDM)
    reset_hidden_state!(model.idm)
    model.a = 0.0
    model.a_merge = 0.0
    model.a_idm = 0.0
    model.other_acc = 0.0
    model.s_des = 0.0
    model.dist_at_merge = 0.0
end

function AutomotiveDrivingModels.observe!(model::CooperativeIDM, scene::Scene, roadway::Roadway, egoid::Int64)
    observe!(model.idm, scene, roadway, egoid)
    a_idm = model.idm.a
    model.a_idm = a_idm 
    veh = find_merge_vehicle(model.env, scene)
    if veh == nothing 
        # println("No merge vehicle")
        model.a = model.a_idm
    else
        ego = get_by_id(scene, egoid)
        model.a
        ego_ttm = time_to_merge(model.env, ego, model.a)
        veh_ttm = time_to_merge(model.env, veh, model.other_acc)
        model.ego_ttm = ego_ttm
        model.veh_ttm = veh_ttm
        if ego_ttm < 0.0 || ego_ttm == Inf || ego_ttm <= veh_ttm || veh_ttm == Inf 
            # println("Ego TTM < Merge TTM, ignoring")
            model.a = model.a_idm
            model.front_car = false
        else
            # println("Ego TTM >= Merge TTM, predicting")
            vehp = constant_acceleration_prediction(model.env, veh, model.other_acc, veh_ttm)
            egop = constant_acceleration_prediction(model.env, ego, model.a, veh_ttm)
            # vehp.state
            # egop.state
            dist_at_merge = distance_projection(model.env, vehp) - distance_projection(model.env, egop)
            model.dist_at_merge = dist_at_merge
            # t_coll = collision_time(model.env, egop, vehp, model.a_merge, model.a_min)
            # d_brake = t_coll == nothing ? 0 : braking_distance(egop.state.v, t_coll, model.a_min)
            v_oth = vehp.state.v
            v_ego = egop.state.v 
            Δv = v_oth - v_ego
            headway = dist_at_merge
            s_des = model.idm.s_min + v_ego*model.idm.T - v_ego*Δv / (2*sqrt(model.idm.a_max*model.idm.d_cmf))
            # @show t_coll
            # @show d_brake
            # @show dist_at_merge
            # @show s_des
            model.s_des = s_des
            if dist_at_merge > s_des
                # println("predicted distance at merge > s_des, ignoring")
                model.a = model.a_idm
                model.front_car = false
            # elseif dist_at_merge <= d_brake
            # elseif dist_at_merge <= s_des + (1 - model.c)*(d_brake - s_des)
            elseif dist_at_merge < model.c*s_des
                # if d_brake <= dist_at_merge
                    # println("critical distance predicted")
                # end
                # consider veh as front car and apply IDM 
                model.front_car = true
                # println("consider merge car as front car")
                headway = distance_projection(model.env, veh) - distance_projection(model.env, ego) 
                v_oth = veh.state.v
                v_ego = ego.state.v
                track_longitudinal!(model.idm, v_ego, v_oth, headway)
                model.a_merge = model.idm.a
                model.a = min(model.a_merge, model.a_idm)
            end
        end
    end
    return model
end


# function AutomotiveDrivingModels.observe!(model::CooperativeIDM, scene::Scene, roadway::Roadway, egoid::Int)
#     observe!(model.idm, scene, roadway, egoid)
#     a_idm = model.idm.a
#     model.a_idm = a_idm
#     model.a = a_idm
#     veh = find_merge_vehicle(model.env, scene)
#     if veh != nothing 
#         ego = get_by_id(scene, egoid)
#         @show ego_ttm = time_to_merge(model.env, ego)
#         @show ttm = time_to_merge(model.env, veh)
#         a_merge = a_idm
#         if ttm < model.ttm_threshold
#             v_ego =  ego.state.v
#             headway = -dist_to_merge(model.env, ego)
#             v_oth = veh.state.v
#             if headway > 0.
#                 track_longitudinal!(model.idm, v_ego, v_oth, headway)
#                 @show a_merge = model.idm.a
#                 model.a = min(a_idm, a_merge)
#             end
#             model.a_merge = a_merge
#         end
#     end    
#     return model
# end


