var documenterSearchIndex = {"docs": [

{
    "location": "cooperative_idm/#",
    "page": "Cooperative IDM",
    "title": "Cooperative IDM",
    "category": "page",
    "text": ""
},

{
    "location": "cooperative_idm/#AutonomousMerging.CooperativeIDM",
    "page": "Cooperative IDM",
    "title": "AutonomousMerging.CooperativeIDM",
    "category": "type",
    "text": "CooperativeIDM <: DriverModel{LaneFollowingAccel}\n\nThe cooperative IDM (c-IDM) is a rule based driver model for merging scenarios.  It controls the longitudinal actions of vehicles on the main lane.  A cooperation level c controls how the vehicles reacts to the merging vehicle.  When c=0 the vehicle ignores the merging vehicle. When c=1 the vehicle considers the merging  vehicle as its front car when TTCego > TTCmergin_vehicle. When it is not considering the merging vehicle,  the car follows the IntelligentDriverModel.\n\nFields\n\n- `env::MergingEnvironment = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/6)` the merging environment\n- `idm::IntelligentDriverModel = IntelligentDriverModel(v_des = env.main_lane_vmax, d_cmf = 2.0, d_max=2.0, T = 1.5, s_min = 2.0, a_max = 2.0)` the default IDM\n- `c::Float64 = 0.0` the cooperation level\n- `fov::Float64 = 20.0` [m] A field of view, the merging vehicle is not considered if it is further than `fov`\n\n\n\n\n\n"
},

{
    "location": "cooperative_idm/#Cooperative-IDM-1",
    "page": "Cooperative IDM",
    "title": "Cooperative IDM",
    "category": "section",
    "text": "    CooperativeIDM"
},

{
    "location": "generative_merging_mdp/#",
    "page": "Generative Merging MDP",
    "title": "Generative Merging MDP",
    "category": "page",
    "text": ""
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.GenerativeMergingMDP",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.GenerativeMergingMDP",
    "category": "type",
    "text": "GenerativeMergingMDP\n\nA simulation environment for a highway merging scenario. Implemented using POMDPs.jl \n\nParameters\n\n- `env::MergingEnvironment = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/7)`\n- `n_cars_main::Int64 = 1`\n- `n_cars_merge::Int64 = 1`\n- `n_agents::Int64 = n_cars_main + n_cars_merge`\n- `max_cars::Int64 = 16`\n- `min_cars::Int64 = 0`\n- `car_def::VehicleDef = VehicleDef()`\n- `dt::Float64 = 0.5 # time step`\n- `jerk_levels::SVector{5, Float64} = SVector(-1, -0.5, 0, 0.5, 1.0)`\n- `accel_levels::SVector{6, Float64} = SVector(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0)`\n- `max_deceleration::Float64 = -4.0`\n- `max_acceleration::Float64 = 3.5`\n- `comfortable_acceleration::Float64 = 2.0`\n- `discount_factor::Float64 = 0.95`\n- `ego_idm::IntelligentDriverModel = IntelligentDriverModel(σ=0.0, v_des=env.main_lane_vmax)`\n- `default_driver_model::DriverModel{LaneFollowingAccel} = IntelligentDriverModel(v_des=env.main_lane_vmax)`\n- `observe_cooperation::Bool = false`\n- `observe_speed::Bool = true`\n- `traffic_speed::Symbol = :mixed`\n- `random_n_cars::Bool = false`\n- `driver_type::Symbol = :random`\n- `max_burn_in::Int64 = 20`\n- `min_burn_in::Int64 = 10`\n- `initial_ego_velocity::Float64 = 10.0`\n- `initial_velocity::Float64 = 5.0`\n- `initial_velocity_std::Float64 = 1.0`\n- `main_lane_slots::LinRange{Float64} = LinRange(0.0, env.main_lane_length + env.after_merge_length, max_cars)`\n- `collision_cost::Float64 = -1.0`\n- `goal_reward::Float64 = 1.0`\n- `hard_brake_cost::Float64 = 0.0`\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.AugScene",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.AugScene",
    "category": "type",
    "text": "AugScene\n\nDriving scene augmented with information about the ego vehicle\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.MergingBelief",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.MergingBelief",
    "category": "type",
    "text": "MergingBelief\n\nA type to represent belief state. It consists of the current observation o and the estimated driver types represented by a dictionnary mapping ID to cooperation levels.\n\nfields\n\no::AugScene\ndriver_types::Dict{Int64, Float64}\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.MergingUpdater",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.MergingUpdater",
    "category": "type",
    "text": "MergingUpdater <: Updater\n\nA belief updater for MergingBelief. It sets o to the current observation and updates  the belief on the drivers cooperation level using Bayes\' rule\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.extract_features",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.extract_features",
    "category": "function",
    "text": "extract_features(mdp::GenerativeMergingMDP, s::AugScene)\n\nextract a feature vector from AugScene\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.normalize_features!",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.normalize_features!",
    "category": "function",
    "text": "normalize_features!(mdp::GenerativeMergingMDP, features::Vector{Float64})\n\nnormalize a feature vector extracted from a scene\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.unnormalize_features!",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.unnormalize_features!",
    "category": "function",
    "text": " unnormalize_features!(mdp::GenerativeMergingMDP, features::Vector{Float64})\n\nrescale feature vector\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.initial_merge_car_state",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.initial_merge_car_state",
    "category": "function",
    "text": "initial_merge_car_state(mdp::GenerativeMergingMDP, rng::AbstractRNG, id::Int64)\n\nreturns a Entity, at the initial state of the merging car.\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.reset_main_car_state",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.reset_main_car_state",
    "category": "function",
    "text": "reset_main_car_state(mdp::GenerativeMergingMDP, veh::Entity)\n\ninitialize a car at the beginning of the main lane\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.reachgoal",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.reachgoal",
    "category": "function",
    "text": "reachgoal(mdp::GenerativeMergingMDP, ego::Entity)\n\nreturn true if ego reached the goal position\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.caused_hard_brake",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.caused_hard_brake",
    "category": "function",
    "text": "caused_hard_brake(mdp::GenerativeMergingMDP, scene::Scene)\n\nreturns true if the ego vehicle caused its rear neighbor to hard brake\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.action_map",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.action_map",
    "category": "function",
    "text": "action_map(mdp::GenerativeMergingMDP, acc::Float64, a::Int64)\n\nmaps integer to a LaneFollowingAccel\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#AutonomousMerging.vehicle_state",
    "page": "Generative Merging MDP",
    "title": "AutonomousMerging.vehicle_state",
    "category": "function",
    "text": "vehicle_state(s::Float64, lane::Lane, v::Float64, roadway::Roadway)\n\nconvenient constructor for VehicleState\n\n\n\n\n\n"
},

{
    "location": "generative_merging_mdp/#Generative-Merging-MDP-1",
    "page": "Generative Merging MDP",
    "title": "Generative Merging MDP",
    "category": "section",
    "text": "    GenerativeMergingMDP\r\n    AugScene\r\n    MergingBelief\r\n    MergingUpdater\r\n    extract_features\r\n    normalize_features!\r\n    unnormalize_features!\r\n    initial_merge_car_state\r\n    reset_main_car_state\r\n    reachgoal\r\n    caused_hard_brake\r\n    action_map\r\n    vehicle_state"
},

{
    "location": "#",
    "page": "About",
    "title": "About",
    "category": "page",
    "text": ""
},

{
    "location": "#About-1",
    "page": "About",
    "title": "About",
    "category": "section",
    "text": "This is the documentation for AutonomousMerging.jl. The environment is defined by MergingEnvironment. This package exports two MDP types:     - MergingMDP, a discrete MDP implemented using the explicit interface of POMDPs.jl with only two traffic participants     - GenerativeMDP, a continuous state MDP with a traffic flow implemented using the generative interface of POMDPs.jlFor more information on the explicit vs generative definition of MDPs read:http://juliapomdp.github.io/POMDPs.jl/latest/explicit/\nhttp://juliapomdp.github.io/POMDPs.jl/latest/generative/"
},

{
    "location": "merging_environment/#",
    "page": "Merging Environment",
    "title": "Merging Environment",
    "category": "page",
    "text": ""
},

{
    "location": "merging_environment/#Merging-Environment-1",
    "page": "Merging Environment",
    "title": "Merging Environment",
    "category": "section",
    "text": ""
},

{
    "location": "merging_environment/#AutonomousMerging.MergingEnvironment",
    "page": "Merging Environment",
    "title": "AutonomousMerging.MergingEnvironment",
    "category": "type",
    "text": "MergingEnvironment\n\nA road network with a main lane and a merging lane. The geometry can be modified by  passing the parameters as keyword arguments in the constructor\n\nParameters\n\nlane_width::Float64 = 3.0\nmain_lane_vmax::Float64 = 15.0\nmerge_lane_vmax::Float64 = 10.0\nmain_lane_length::Float64 = 100.0\nmain_lane_angle::Float64 = float(pi)/4\nmerge_lane_angle::Float64 = float(pi)/4\nmerge_lane_length::Float64 = 50.0\nafter_merge_length::Float64 = 50.0\n\nInternals\n\nroadway::Roadway{Float64} contains all the road segment and lane information\nmerge_point::VecSE2{Float64} coordinate of the merge point in cartesian frame (0.0, 0.0, 0.0) by default\nmerge_proj::RoadProjection{Int64, Float64} projection of the merge point on the roadway \nmerge_index::RoadIndex\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.generate_merging_roadway",
    "page": "Merging Environment",
    "title": "AutonomousMerging.generate_merging_roadway",
    "category": "function",
    "text": "generate_merging_roadway(lane_width::Float64 = 3.0, main_lane_vmax::Float64 = 20.0, merge_lane_vmax::Float64 = 15.0, main_lane_length::Float64 = 20.0, merge_lane_length::Float64 = 20.0, after_merge_length::Float64 = 20.0, main_lane_angle::Float64 = float(pi)/4, merge_lane_angle::Float64 = float(pi)/4)\n\nGenerate a Roadway object representing a merging scenario.  The merge point is defined at (0., 0.) by default.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.main_lane",
    "page": "Merging Environment",
    "title": "AutonomousMerging.main_lane",
    "category": "function",
    "text": "main_lane(env::MergingEnvironment)\n\nreturns the main lane of the merging scenario\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.merge_lane",
    "page": "Merging Environment",
    "title": "AutonomousMerging.merge_lane",
    "category": "function",
    "text": "merge_lane(env::MergingEnvironment)\n\nreturns the merging lane of the merging scenario\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#Environment-1",
    "page": "Merging Environment",
    "title": "Environment",
    "category": "section",
    "text": "    MergingEnvironment\r\n    generate_merging_roadway\r\n    main_lane\r\n    merge_lane"
},

{
    "location": "merging_environment/#AutonomousMerging.get_front_neighbor",
    "page": "Merging Environment",
    "title": "AutonomousMerging.get_front_neighbor",
    "category": "function",
    "text": "get_front_neighbor(env::MergingEnvironment, scene::Scene, egoid::Int64)\n\nreturns the front neighbor of egoid in its lane. It returns an object of type NeighborLongitudinalResult\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.get_neighbors",
    "page": "Merging Environment",
    "title": "AutonomousMerging.get_neighbors",
    "category": "function",
    "text": "get_neighbors(env::MergingEnvironment, scene::Scene, egoid::Int64)\n\nreturns the following neighbors id and relative distance (if they exist)      - the front neighbor of vehicle egoid     - the vehicle right behind the merge point (if egoid is on the main lane)     - the front neighbor of the projection of egoid on the main lane      - the rear neighbor of the projection of egoid on the merge lane \n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.dist_to_merge",
    "page": "Merging Environment",
    "title": "AutonomousMerging.dist_to_merge",
    "category": "function",
    "text": "dist_to_merge(env::MergingEnvironment, veh::Entity)\n\nreturns the distance to the merge point.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.time_to_merge",
    "page": "Merging Environment",
    "title": "AutonomousMerging.time_to_merge",
    "category": "function",
    "text": "time_to_merge(env::MergingEnvironment, veh::Entity, a::Float64 = 0.0)\n\nreturn the time to reach the merge point using constant acceleration prediction.  If the acceleration, a is not specified, it performs a constant velocity prediction.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.find_merge_vehicle",
    "page": "Merging Environment",
    "title": "AutonomousMerging.find_merge_vehicle",
    "category": "function",
    "text": "find_merge_vehicle(env::MergingEnvironment, scene::Scene)\n\nreturns the id of the merging vehicle if there is a vehicle on the merge lane.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.constant_acceleration_prediction",
    "page": "Merging Environment",
    "title": "AutonomousMerging.constant_acceleration_prediction",
    "category": "function",
    "text": "constant_acceleration_prediction(env::MergingEnvironment, veh::Entity, acc::Float64, time::Float64, v_des::Float64)\n\nreturns the state of vehicle veh after time time using a constant acceleration prediction. \n\ninputs\n\nenv::MergingEnvironment the environment \nveh::Entity the initial state of the vehicle\nacc::Float64 the current acceleration of the vehicle \ntime::Float64 the prediction horizon \nv_des::Float64 the desired speed of the vehicle (assumes that the vehicle will not exceed that speed)\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.distance_projection",
    "page": "Merging Environment",
    "title": "AutonomousMerging.distance_projection",
    "category": "function",
    "text": "distance_projection(env::MergingEnvironment, veh::Entity)\n\nPerforms a projection of veh onto the main lane. It returns the longitudinal position of the projection of veh on the main lane.  The projection is computing by conserving the distance to the merge point.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.collision_time",
    "page": "Merging Environment",
    "title": "AutonomousMerging.collision_time",
    "category": "function",
    "text": "collision_time(env::MergingEnvironment, veh::Entity, mergeveh::Entity, acc_merge::Float64, acc_min::Float64)\n\ncompute the collision time between two vehicles assuming constant acceleration.\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#AutonomousMerging.braking_distance",
    "page": "Merging Environment",
    "title": "AutonomousMerging.braking_distance",
    "category": "function",
    "text": "braking_distance(v::Float64, t_coll::Float64, acc::Float64)\n\ncomputes the distance to reach a velocity of 0. at constant acceleration acc in time t_coll with initial velocity v\n\n\n\n\n\n"
},

{
    "location": "merging_environment/#Features-and-Helper-functions-1",
    "page": "Merging Environment",
    "title": "Features and Helper functions",
    "category": "section",
    "text": "    get_front_neighbor\r\n    get_neighbors\r\n    dist_to_merge\r\n    time_to_merge\r\n    find_merge_vehicle\r\n    constant_acceleration_prediction\r\n    distance_projection\r\n    collision_time\r\n    braking_distance"
},

{
    "location": "merging_mdp/#",
    "page": "Merging MDP",
    "title": "Merging MDP",
    "category": "page",
    "text": ""
},

{
    "location": "merging_mdp/#AutonomousMerging.MergingMDP",
    "page": "Merging MDP",
    "title": "AutonomousMerging.MergingMDP",
    "category": "type",
    "text": "MergingMDP <: MDP{MergingState, Int64}\n\nA discrete state and action MDP representing a merging scenario with only two agents:  the ego vehicle on the merge lane and another traffic participant on the main lane.\n\n\n\n\n\n"
},

{
    "location": "merging_mdp/#Merging-MDP-1",
    "page": "Merging MDP",
    "title": "Merging MDP",
    "category": "section",
    "text": "    MergingMDP"
},

{
    "location": "rendering/#",
    "page": "Rendering",
    "title": "Rendering",
    "category": "page",
    "text": ""
},

{
    "location": "rendering/#Rendering-1",
    "page": "Rendering",
    "title": "Rendering",
    "category": "section",
    "text": "For more information on how to use overlays and other rendering features see AutomotiveVisualization.jl.    MergingNeighborsOverlay\r\n    DistToMergeOverlay\r\n    MaskingOverlay\r\n    CooperativeIDMOverlay\r\n    BeliefOverlay\r\n    get_car_type_colors\r\n    super_render"
},

]}
