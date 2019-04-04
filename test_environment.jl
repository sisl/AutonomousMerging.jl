using Revise
using AutomotiveDrivingModels
using Parameters
using AutoViz 

includet("environment.jl")

env = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/12)

AutoViz.render(Scene(),env.roadway, [LaneOverlay(main_lane(env), colorant"blue")])

struct LaneOverlay <: SceneOverlay
    lane::Lane
    color::Colorant
end
function AutoViz.render!(rendermodel::RenderModel, overlay::LaneOverlay, scene::Scene, roadway::Roadway)
    render!(rendermodel, overlay.lane, roadway, color_asphalt=overlay.color)
    return rendermodel
end

# test modulo roadway 

state1 = VehicleState(VecSE2(45.0, 0.0, 0.0), env.roadway, 10.0)
veh1 = Vehicle(state1, VehicleDef(), 1)

s = Scene()
push!(s, veh1)

AutoViz.render(s, env.roadway, cam=FitToContentCamera(0.0))


veh1 = Vehicle(propagate(veh1, LaneFollowingAccel(0.0), env.roadway, 1.0), VehicleDef(), 1)

s = Scene()
push!(s, veh1)
AutoViz.render(s, env.roadway, cam=FitToContentCamera(0.0))



env.roadway