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

deleteat!(s0.scene, 1)
vehstate = vehicle_state(30.0, merge_lane(mdp.env), 10.0, mdp.env.roadway)
push!(s0.scene, Vehicle(vehstate, VehicleDef(), EGO_ID))

c = AutoViz.render(s0.scene, mdp.env.roadway, 
            SceneOverlay[
                 DistToMergeOverlay(target_id=EGO_ID, env=mdp.env, line_width=0.3)
            ],
            cam = StaticCamera(VecE2(-20.0, -5.0), 18.0),
            car_colors=get_car_type_colors(s0.scene, mdp.driver_models),
            canvas_width=1300,
            canvas_height=300)

using Cairo

write_to_png(c, "scenario.png")

c = AutoViz.render([ArrowCar([0.,0.], color=RGBA(0.0, 1., 0., 1.))])