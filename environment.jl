
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
const CARDEF = VehicleDef()

const EGO_ID = 1

@with_kw struct MergingEnvironment
    lane_width::Float64 = 3.0
    main_lane_vmax::Float64 = 25.0
    merge_lane_vmax::Float64 = 25.0
    main_lane_length::Float64 = 20.0
    merge_lane_length::Float64 = 20.0
    after_merge_length::Float64 = 20.0
    roadway::Roadway = generate_merging_roadway(lane_width, main_lane_vmax, merge_lane_vmax, after_merge_length) 
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
                                   after_merge_length::Float64 = 20.0)
    # init empty roadway 
    roadway = Roadway()
    n_pts = 2 # sample points for the roadway, only two needed each time, since all segments are straight

    merge_point = VecE2(0.0, 0.0)
    # define main lane 
    main_lane_startpt = merge_point + polar(main_lane_length, 3*float(pi)/4)
    curve = gen_straight_curve(main_lane_startpt, merge_point, n_pts)
    append_to_curve!(curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts))
    main_lane = Lane(LaneTag(MAIN_LANE_ID, 1), curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax))
    push!(roadway.segments, RoadSegment(main_lane.tag.segment, [main_lane]))


    # define merge lane 
    merge_lane_startpt = merge_point + polar(merge_lane_length, -3*float(pi)/4)
    curve = gen_straight_curve(merge_lane_startpt, merge_point, n_pts)
    append_to_curve!(curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts))
    merge_lane = Lane(LaneTag(MERGE_LANE_ID, 1), curve, width = lane_width, speed_limit=SpeedLimit(0.,merge_lane_vmax))
    push!(roadway.segments, RoadSegment(merge_lane.tag.segment, [merge_lane]))

    return roadway
end

