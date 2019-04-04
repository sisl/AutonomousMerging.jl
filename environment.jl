
# Specification from Yeping's code
# path1 = Path('mainLane', [0,50], [800,50], 0, [1000,50], 2, maxV1 = 20, maxV2 = 20)
# path2 = Path('mergeLane', [0,150], [180,150], [800,50], [1000,50], 3, maxV1 = 15, maxV2 = 20)
# path = [path1, path2]

# '''
# roadType: road type is either 'mainLane' or 'mergeLane'
# startPt:  [x, y] start point of the road
# midPt1:   [x, y] fist break point of the road
# midPt2:   [x, y] second break point of the road 
# endPt:    [x, y] end point of the road
# numSeg:   >=2, number of road segment which needs to be consistent with the (point number - 1)
# width :   lane width 
# maxV1:    maximum velocity allowed for the lane before the merging point
# maxV2: 	  maximum velocity allowed for the lane after the merging point
# '''

using AutomotiveDrivingModels
using Parameters

const M2L = 4
const MAIN_LANE_ID = 1
const MERGE_LANE_ID = 2
const AFTER_MERGE_LANE_ID = 3
const CARDEF = VehicleDef()
const ACC_MIN = -2.0
const EGO_ID = 1

@with_kw struct MergingEnvironment
    lane_width::Float64 = 3.0
    main_lane_vmax::Float64 = 15.0
    merge_lane_vmax::Float64 = 10.0
    main_lane_length::Float64 = 100.0
    main_lane_angle::Float64 = float(pi)/4
    merge_lane_angle::Float64 = float(pi)/4
    merge_lane_length::Float64 = 50.0
    after_merge_length::Float64 = 50.0

    # internals
    roadway::Roadway = generate_merging_roadway(lane_width, 
                                                main_lane_vmax, 
                                                merge_lane_vmax, 
                                                main_lane_length,
                                                merge_lane_length,
                                                after_merge_length, 
                                                main_lane_angle, 
                                                merge_lane_angle) 
    merge_point::VecSE2{Float64} = VecSE2(0.0, 0.0, 0.0)
    merge_proj::RoadProjection = proj(merge_point, roadway)
    merge_index::RoadIndex = RoadIndex(merge_proj.curveproj.ind, merge_proj.tag)
end


function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

function generate_merging_roadway(lane_width::Float64 = 3.0, 
                                   main_lane_vmax::Float64 = 20.0,
                                   merge_lane_vmax::Float64 = 15.0,
                                   main_lane_length::Float64 = 20.0, 
                                   merge_lane_length::Float64 = 20.0,
                                   after_merge_length::Float64 = 20.0,
                                   main_lane_angle::Float64 = float(pi)/4, 
                                   merge_lane_angle::Float64 = float(pi)/4) 
    # init empty roadway 
    roadway = Roadway()
    n_pts = 2 # sample points for the roadway, only two needed each time, since all segments are straight
    main_tag = LaneTag(MAIN_LANE_ID, 1)
    merge_tag = LaneTag(MERGE_LANE_ID, 1)
    # after_merge_tag = LaneTag(AFTER_MERGE_LANE_ID, 1)

    # define curves
    merge_point = VecE2(0.0, 0.0) 
    main_lane_startpt = merge_point + polar(main_lane_length, -float(pi) - main_lane_angle)
    main_curve = gen_straight_curve(main_lane_startpt, merge_point, n_pts)
    merge_index = curveindex_end(main_curve)
    append_to_curve!(main_curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts)[2:end])
    merge_lane_startpt = merge_point + polar(merge_lane_length, float(pi) + merge_lane_angle)
    merge_curve = gen_straight_curve(merge_lane_startpt, merge_point, n_pts)


    # define lanes with connections 
    main_lane = Lane(main_tag, main_curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax))
    merge_lane = Lane(merge_tag, merge_curve, width = lane_width,speed_limit=SpeedLimit(0.,merge_lane_vmax),
                        next=RoadIndex(merge_index, main_tag))

    # after_merge_curve =  gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts)
    # after_merge_lane = Lane(after_merge_tag, after_merge_curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax))

    # # define lanes with connections 
    # main_lane = Lane(main_tag, main_curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax),
    #                     next=RoadIndex(CURVEINDEX_START, after_merge_tag))
    # merge_lane = Lane(merge_tag, merge_curve, width = lane_width,speed_limit=SpeedLimit(0.,merge_lane_vmax),
    #                     next=RoadIndex(CURVEINDEX_START, after_merge_tag))
    # after_merge_lane = Lane(after_merge_tag, after_merge_curve, width=lane_width, speed_limit=SpeedLimit(0., main_lane_vmax))

    # add segments to roadway 
    push!(roadway.segments, RoadSegment(MAIN_LANE_ID, [main_lane]))
    push!(roadway.segments, RoadSegment(MERGE_LANE_ID, [merge_lane]))
    # push!(roadway.segments, RoadSegment(AFTER_MERGE_LANE_ID, [after_merge_lane]))

    # append_to_curve!(curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts))
    # push!(main_lane.exits, LaneConnection(true, curveindex_end(main_lane.curve), 
    #                                  RoadIndex(CURVEINDEX_START, main_lane.tag)))
    # push!(main_lane.entrances, LaneConnection(false, CURVEINDEX_START, 
    #                                  RoadIndex(curveindex_end(main_lane.curve), main_lane.tag)))
    # push!(roadway.segments, RoadSegment(main_lane.tag.segment, [main_lane]))


    # define merge lane 
    
    # append_to_curve!(curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts))
    # merge_lane = Lane(LaneTag(MERGE_LANE_ID, 1), curve, width = lane_width, speed_limit=SpeedLimit(0.,merge_lane_vmax))
    # push!(merge_lane.exits, LaneConnection(true, curveindex_end(main_lane.curve), 
    #                                  RoadIndex(CURVEINDEX_START, main_lane.tag)))
    # push!(roadway.segments, RoadSegment(merge_lane.tag.segment, [merge_lane]))

    # define after merge
    # after_merge_curve =  gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts)
    # after_merge_lane = Lane(LaneTag(AFTER_MERGE_LANE_ID, 1), curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax))
    return roadway
end

## convenience 

main_lane(env::MergingEnvironment) = env.roadway[LaneTag(MAIN_LANE_ID, 1)]
merge_lane(env::MergingEnvironment) = env.roadway[LaneTag(MERGE_LANE_ID, 1)]

## features 

function get_front_neighbor(env::MergingEnvironment, scene::Scene, egoid::Int64)
    ego_ind = findfirst(egoid, scene)
    ego = scene[ego_ind]
    # merge neighbor
    ego_lane = get_lane(env.roadway, ego)
    if ego_lane == main_lane(env)
        fore_res = get_neighbor_fore_along_lane(scene, ego_ind, env.roadway)
    else
        lane = main_lane(env)
        # s_base = lane[env.merge_proj.curveproj.ind, env.roadway].s
        s_base = env.main_lane_length
        fore_res = get_neighbor_fore_along_lane(scene, env.roadway, lane.tag, s_base, index_to_ignore=ego_ind)
        # fore_res = NeighborLongitudinalResult(fore_res.ind, fore_res.Δs - dist_to_merge(env, ego))
    end
end

function get_neighbors(env::MergingEnvironment, scene::Scene, egoid::Int64)
    ego_ind = findfirst(egoid, scene)
    ego = scene[ego_ind]

    #front neighbor 
    front = get_front_neighbor(env, scene, egoid)

    # merge neighbor
    ego_lane = get_lane(env.roadway, ego)
    if ego_lane == main_lane(env)
        fore_res = get_neighbor_rear_along_lane(scene, ego_ind, env.roadway)
    else
        lane = main_lane(env)
        # s_base = lane[env.merge_proj.curveproj.ind, env.roadway].s
        s_base = env.main_lane_length
        fore_res = get_neighbor_rear_along_lane(scene, env.roadway, lane.tag, s_base, index_to_ignore=ego_ind)
        # fore_res = NeighborLongitudinalResult(fore_res.ind, fore_res.Δs + dist_to_merge(env, ego))
    end
    # two closest car in main lane
    proj_lane = main_lane(env)
    main_lane_proj = proj(ego.state.posG, proj_lane, env.roadway)
    s_main = proj_lane[main_lane_proj.curveproj.ind, env.roadway].s
    fore_main = get_neighbor_fore_along_lane(scene, env.roadway, proj_lane.tag, s_main, index_to_ignore=ego_ind)
    rear_main = get_neighbor_rear_along_lane(scene, env.roadway, proj_lane.tag, s_main, index_to_ignore=ego_ind)
    return front, fore_res, fore_main, rear_main
end

function dist_to_merge(env::MergingEnvironment, veh::Vehicle)
    lane = get_lane(env.roadway, veh)
    if lane == main_lane(env)
        frenet_merge = get_frenet_relative_position(veh.state.posG, env.merge_index, env.roadway)
        dist = frenet_merge.Δs
    else
        dist = veh.state.posF.s - get_end(lane) 
    end
    return dist
end

function time_to_merge(env::MergingEnvironment, veh::Vehicle, a::Float64 = 0.0)
    d = -dist_to_merge(env, veh)
    v = veh.state.v
    t = Inf
    if isapprox(a, 0) 
        t =  d/veh.state.v 
    else
        delta = v^2 + 2.0*a*d
        if delta < 0.0
            t = Inf
        else
            t = (-v + sqrt(delta)) / a 
        end
        if t < 0.0
            t = Inf
        end
    end
    return t
end

function find_merge_vehicle(env::MergingEnvironment, scene::Scene)
    for veh in scene 
        lane = get_lane(env.roadway, veh)
        if lane == merge_lane(env)
            return veh
        end
    end
    return nothing
end

function constant_acceleration_prediction(env::MergingEnvironment, 
                                          veh::Vehicle,
                                          acc::Float64,
                                          time::Float64)
        act = LaneFollowingAccel(acc)
        vehp = propagate(veh, act, env.roadway, time)
        return Vehicle(vehp, veh.def, veh.id)
end

function distance_projection(env::MergingEnvironment, veh::Vehicle)
    if get_lane(env.roadway, veh) == main_lane(env)
        return veh.state.posF.s 
    else
        dm = -dist_to_merge(env, veh)
        return env.roadway[env.merge_index].s - dm
    end
end

function collision_time(env::MergingEnvironment, 
                        veh::Vehicle, 
                        mergeveh::Vehicle, 
                        acc_merge::Float64, 
                        acc_min::Float64)
    rel_vel = mergeveh.state.v - veh.state.v
    rel_pos = distance_projection(env, mergeveh) - distance_projection(env, veh)
    rel_acc = acc_merge - acc_min 
    delta = rel_vel^2 - 2*rel_acc*rel_pos
    if delta < 0.0
        return nothing 
    elseif rel_acc != 0.0
        t_coll = (-rel_vel + sqrt(delta))/rel_acc
        return t_coll
    elseif rel_vel != 0.0 
        t_coll = - rel_pos / rel_vel
        return t_coll
    else
        t_coll = nothing
        return t_coll
    end
end

function braking_distance(v::Float64, t_coll::Float64, acc::Float64)
    brake_dist = v*t_coll + 0.5*acc*t_coll^2
    return brake_dist
end