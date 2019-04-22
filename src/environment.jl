using AutomotiveDrivingModels
using Parameters

const MAIN_LANE_ID = 1
const MERGE_LANE_ID = 2

"""
    MergingEnvironment

A road network with a main lane and a merging lane. The geometry can be modified by 
passing the parameters as keyword arguments in the constructor

# Parameters
- `lane_width::Float64 = 3.0`
- `main_lane_vmax::Float64 = 15.0`
- `merge_lane_vmax::Float64 = 10.0`
- `main_lane_length::Float64 = 100.0`
- `main_lane_angle::Float64 = float(pi)/4`
- `merge_lane_angle::Float64 = float(pi)/4`
- `merge_lane_length::Float64 = 50.0`
- `after_merge_length::Float64 = 50.0`

# Internals 

- `roadway::Roadway{Float64}` contains all the road segment and lane information
- `merge_point::VecSE2{Float64}` coordinate of the merge point in cartesian frame (0.0, 0.0, 0.0) by default
- `merge_proj::RoadProjection{Int64, Float64}` projection of the merge point on the roadway 
- `merge_index::RoadIndex`

"""
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
    roadway::Roadway{Float64} = generate_merging_roadway(lane_width, 
                                                main_lane_vmax, 
                                                merge_lane_vmax, 
                                                main_lane_length,
                                                merge_lane_length,
                                                after_merge_length, 
                                                main_lane_angle, 
                                                merge_lane_angle) 
    merge_point::VecSE2{Float64} = VecSE2(0.0, 0.0, 0.0)
    merge_proj::RoadProjection{Int64, Float64} = proj(merge_point, roadway)
    merge_index::RoadIndex{Int64, Float64} = RoadIndex(merge_proj.curveproj.ind, merge_proj.tag)
end


function append_to_curve!(target::Curve{T}, newstuff::Curve{T}) where T <: Real
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt{T}(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

"""
    generate_merging_roadway(lane_width::Float64 = 3.0, main_lane_vmax::Float64 = 20.0, merge_lane_vmax::Float64 = 15.0, main_lane_length::Float64 = 20.0, merge_lane_length::Float64 = 20.0, after_merge_length::Float64 = 20.0, main_lane_angle::Float64 = float(pi)/4, merge_lane_angle::Float64 = float(pi)/4)

Generate a `Roadway` object representing a merging scenario. 
The merge point is defined at (0., 0.) by default.
"""
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

    # add segments to roadway 
    push!(roadway.segments, RoadSegment(MAIN_LANE_ID, [main_lane]))
    push!(roadway.segments, RoadSegment(MERGE_LANE_ID, [merge_lane]))
  
    return roadway
end

## convenience 
"""
    main_lane(env::MergingEnvironment)
returns the main lane of the merging scenario
"""
main_lane(env::MergingEnvironment) = env.roadway[LaneTag(MAIN_LANE_ID, 1)]

"""
    merge_lane(env::MergingEnvironment)
returns the merging lane of the merging scenario
"""
merge_lane(env::MergingEnvironment) = env.roadway[LaneTag(MERGE_LANE_ID, 1)]