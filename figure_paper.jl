using Revise
using Random
using Printf
using StatsBase
using StaticArrays
using Distributions
using LinearAlgebra
using POMDPs
using AutoViz
using AutomotiveDrivingModels
using AutomotivePOMDPs
using POMDPSimulators
using BeliefUpdaters
using POMDPPolicies
using RLInterface
using POMDPModelTools

# LOGGER = Logger("log1", overwrite=true)
includet("environment.jl")
includet("generative_mdp.jl")
# includet("masking.jl")
includet("cooperative_IDM.jl")
includet("belief_updater.jl")
includet("belief_mdp.jl")
includet("overlays.jl")
includet("make_gif.jl")

rng = MersenneTwister(1)

mdp = GenerativeMergingMDP(random_n_cars=true, min_cars=12,max_cars=14, traffic_speed = :mixed, driver_type=:random, 
                           observe_cooperation=true)

s0 = initialstate(mdp, rng)

deleteat!(s0.scene,1)
vehstate = vehicle_state(110.0, main_lane(mdp.env), 10.0, mdp.env.roadway)
push!(s0.scene, Vehicle(vehstate, VehicleDef(), EGO_ID))

c = AutoViz.render(s0.scene, mdp.env.roadway, 
            SceneOverlay[
                #  DistToMergeOverlay(target_id=EGO_ID, env=mdp.env, line_width=0.3),
                #  MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env)
            ],
            cam = StaticCamera(VecE2(2.0, -5.0), 22.0),
            car_colors=get_car_type_colors(s0.scene, mdp.driver_models),
            canvas_width=1000,
            canvas_height=400)

using Cairo

write_to_png(c, "features2.png")

c = AutoViz.render([ArrowCar([0.,0.], color=RGBA(0.8, 0.8, 0.8, 1))])

write_to_png(c, "ego_shade2.png")