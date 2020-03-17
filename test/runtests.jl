using AutonomousMerging 
using Random
using AutomotiveDrivingModels
using POMDPs
using POMDPPolicies
using POMDPSimulators
using Test 

function test_state(mdp::GenerativeMergingMDP, s,v=0.0, acc=0.0)
    ego = Vehicle(vehicle_state(35.0, merge_lane(mdp.env), 5.0, mdp.env.roadway), VehicleDef(), EGO_ID)
    veh1 = Vehicle(vehicle_state(s, main_lane(mdp.env), 4.9, mdp.env.roadway), VehicleDef(), EGO_ID + 1)
    scene = Scene()
    push!(scene, ego)
    push!(scene, veh1)
    return AugScene(scene, (acc=acc,))
end

@testset "Environment" begin 
    env = MergingEnvironment(main_lane_angle = 0.0, merge_lane_angle = pi/12)
    main = main_lane(env)
    merge = merge_lane(env)
    @test main == env.roadway[LaneTag(MAIN_LANE_ID, 1)]
    @test main.curve[end].s == env.main_lane_length + env.after_merge_length
    @test merge.curve[end].s == env.merge_lane_length
end

@testset "GenerativeMDP" begin 

    rng = MersenneTwister(1)
    mdp = GenerativeMergingMDP(max_cars=12, 
                               min_cars=10,
                               driver_type = :random, 
                               observe_cooperation = true, 
                               initial_ego_velocity=0.0)
    s0 = initialstate(mdp, rng)
    policy = RandomPolicy(mdp, rng=rng)
    
    hr = HistoryRecorder(rng = rng, max_steps=100)
    hist = POMDPSimulators.simulate(hr, mdp, policy, s0)

    s = s0
    svec = convert_s(Vector{Float64}, s, mdp)
    srec = convert_s(AugScene, svec, mdp)
    @test get_by_id(srec.scene, EGO_ID).state.posG ≈ get_by_id(s.scene, EGO_ID).state.posG
    
    s = state_hist(hist)[end]
    svec = convert_s(Vector{Float64}, s, mdp)
    srec = convert_s(AugScene, svec, mdp)
    @test get_by_id(srec.scene, EGO_ID).state.posG ≈ get_by_id(s.scene, EGO_ID).state.posG
end

@testset "CooperativeIDM" begin 
    rng = MersenneTwister(1)

    mdp = GenerativeMergingMDP(random_n_cars=true, dt=0.5)

    mdp.driver_models[2] = CooperativeIDM(c=1.0)
    set_desired_speed!(mdp.driver_models[2], 5.0)
    

    s0 = test_state(mdp, 85.0)
    policy = FunctionPolicy(s->7)
    hr = HistoryRecorder(rng = rng, max_steps=40)
    hist = POMDPSimulators.simulate(hr, mdp, policy, s0)

end
