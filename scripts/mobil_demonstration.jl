using Revise
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveVisualization
using AutonomousMerging
using Distributions
using Random

action_type(d::DriverModel{A}) where A = A

function AutomotiveDrivingModels.propagate(veh::Entity{VehicleState,D,I}, action::LatLonAccel, roadway::Roadway, ΔT::Float64, 
                                           nobackup::Bool,
                                           wraparound::Bool) where {D,I}

    a_lat = action.a_lat
    a_lon = action.a_lon

     v = veh.state.v
     ϕ = veh.state.posF.ϕ
    ds = v*cos(ϕ)
     t = veh.state.posF.t
    dt = v*sin(ϕ)

    ΔT² = ΔT*ΔT
    Δs = ds*ΔT + 0.5*a_lon*ΔT²
    Δt = dt*ΔT + 0.5*a_lat*ΔT²

    ds₂ = ds + a_lon*ΔT
    nobackup ? ds₂ = max(0, ds₂) : nothing
    dt₂ = dt + a_lat*ΔT
    speed₂ = sqrt(dt₂*dt₂ + ds₂*ds₂)
    v₂ = sqrt(dt₂*dt₂ + ds₂*ds₂) # v is the magnitude of the velocity vector
    ϕ₂ = atan(dt₂, ds₂)

    if wraparound && veh.posF.s + Δs > roadway[veh.posF.roadind.tag].curve[end].s
        lane = roadway[veh.posF.roadind.tag]
        ds_to_end =  lane.curve[end].s - veh.posF.s
        ds_beginning = Δs - ds_to_end
        curveind = lane.curve[1]
        roadind = RoadIndex{I,T}(curveind, lane.exits[1].target.tag)
        roadind = move_along(roadind, roadway, ds_beginning)
    end
    roadind = move_along(veh.state.posF.roadind, roadway, Δs)
    footpoint = roadway[roadind]
    posG = VecE2{Float64}(footpoint.pos.x,footpoint.pos.y) + polar(t + Δt, footpoint.pos.θ + π/2)

    posG = VecSE2{Float64}(posG.x, posG.y, footpoint.pos.θ + ϕ₂)

    veh = VehicleState(posG, roadway, v₂)
    return veh
end

function wrap_around(roadway::Roadway, veh::Entity)
    lane = get_lane(roadway, veh)
    s_end = get_end(lane)
    s = veh.state.posF.s
    if s >= s_end 
        posF = Frenet(lane, 0.0)
        vehstate = VehicleState(posF, roadway, veh.state.v)
        return Entity(vehstate,veh.def, veh.id)
    end
    return veh
end

function AutomotiveDrivingModels.simulate!(scene::Scene, roadway::Roadway, models::Dict{Int64, D}, dt::Float64) where {A, D<:DriverModel}
    s = deepcopy(scene)
    acts = Vector{action_type(models[1])}(undef, length(s))
    get_actions!(acts, s, roadway, models)
    tick!(s, roadway, acts, dt, true)
    for (i, veh) in enumerate(s)
        s[i] = wrap_around(roadway, s[i])
    end
    return s
end

function AutomotiveDrivingModels.simulate!(s0::Scene, roadway::Roadway, models::Dict{Int64, D}, nsteps::Int64, dt::Float64) where D <: DriverModel
    scenes = Vector{Scene}(undef, nsteps+1)
    scenes[1] = s0
    for t=1:nsteps
        scenes[t+1] = simulate!(scenes[t], roadway, models, dt)
    end
    return scenes
end

function initial_scene(n_vehicles, roadway, model;
                       timestep::Float64 = 0.1,
                       n_lanes::Int64 = 3,
                       burn_in::Int64 = 20,
                       init_velocity_dist = Normal(10.0, 3.0),
                       rng::AbstractRNG = Random.GLOBAL_RNG)
    # populate roadway 
    scene = Scene()
    models = Dict{Int64, typeof(model)}()
    for i=1:n_vehicles
        v = rand(rng, init_velocity_dist)
        lane_tag = LaneTag(1, rand(rng, 1:n_lanes))
        lane = roadway[lane_tag]
        s = rand(rng, 0.0:lane.curve[end].s)
        posF = Frenet(lane, s)
        vehstate = VehicleState(posF, roadway, v)
        veh = Entity(vehstate, VehicleDef(), i)
        models[i] = deepcopy(model)
        set_desired_speed!(models[i], v)
        push!(scene, veh)
    end

    scenes = simulate!(scene, roadway, models, burn_in, timestep)
    return scenes[end], models, scenes
end

##### SIMULATION PARAMETERS #########

roadway = gen_straight_roadway(3, 200.0)


time_step = 0.1

n_vehicles = 20

idm = IntelligentDriverModel(v_des=10.0,
                               d_cmf = 2.0, 
                                d_max=2.0,
                                T = 1.5,
                                s_min = 2.0,
                                a_max = 2.0)
model = Tim2DDriver(time_step, 
                    mlon = idm,
                    mlat = ProportionalLaneTracker(),
                    mlane = MOBIL(time_step, mlon=idm, politeness=0.35, safe_decel = 2.0))

s0, models, _ = initial_scene(n_vehicles, roadway, model, burn_in=100)

n_steps = 300

@time scenes = simulate!(s0, roadway, models, n_steps, 0.1)


using Blink
using Interact

w = Window()
ui = @manipulate for s=1:n_steps
    render([roadway, scenes[s]])
end
body!(w, ui)


