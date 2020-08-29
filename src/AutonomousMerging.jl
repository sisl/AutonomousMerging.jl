module AutonomousMerging

using Random
using Printf
using LinearAlgebra
using Distributions
using Parameters
using StatsBase
using POMDPs
using POMDPModelTools
using BeliefUpdaters
using StaticArrays
using GridInterpolations
using AutomotiveSimulator
using AutomotiveVisualization

const EGO_ID = 1

export MergingEnvironment,
       generate_merging_roadway,
       main_lane,
       merge_lane,
       EGO_ID,
       MAIN_LANE_ID,
       MERGE_LANE_ID

include("environment.jl")

export get_front_neighbor,
       get_neighbors,
       dist_to_merge,
       time_to_merge,
       find_merge_vehicle,
       constant_acceleration_prediction,
       distance_projection,
       collision_time,
       braking_distance

include("features.jl")

export CooperativeIDM

include("cooperative_IDM.jl")

export MergingMDP

include("mdp_definition.jl")

export GenerativeMergingMDP,
       AugScene,
       extract_features,
       normalize_features!,
       unnormalize_features!,
       
       #helpers
       initial_merge_car_state,
       reset_main_car_state,
       reachgoal,
       caused_hard_brake,
       action_map,
       vehicle_state

include("generative_mdp.jl")

export MergingBelief,
       MergingUpdater

include("belief_updater.jl")
include("belief_mdp.jl")

export MergingNeighborsOverlay,
       DistToMergeOverlay,
       MaskingOverlay,
       CooperativeIDMOverlay,
       BeliefOverlay,
       get_car_type_colors,
       super_render

include("overlays.jl")
include("rendering.jl")

end
