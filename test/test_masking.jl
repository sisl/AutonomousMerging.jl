using Test
function test_state(v, acc, s=101.0)
    ego = Entity(vehicle_state(s, main_lane(mdp.env), v, mdp.env.roadway), VehicleDef(), EGO_ID)
    veh1 = Entity(vehicle_state(70.0, main_lane(mdp.env), 20.0, mdp.env.roadway), VehicleDef(), EGO_ID + 1)
    veh2 = Entity(vehicle_state(150.0, main_lane(mdp.env), 20.0, mdp.env.roadway), VehicleDef(), EGO_ID + 2)
    scene = Scene()
    push!.(Ref(scene), [ego, veh1, veh2])

    return AugScene(scene, (acc=acc,))
end

AutomotiveVisualization.render(s.scene, mdp.env.roadway, [IDOverlay()], cam=CarFollowCamera(1, 10.0))

s = test_state(20.0, 0.0)
@test speed_limit_actions(mdp, s) == actions(mdp)
s = test_state(25.0, 0.0)
@test speed_limit_actions(mdp, s) == [1,2,3,4,7]
s = test_state(26.0, 0.0)
@test speed_limit_actions(mdp, s) == [1]

# test acceleration limit 
s = test_state(20.0, 0.0)
@test acceleration_limit_actions(mdp, s) == [2,3,4,5,6,7]
s = test_state(20.0, 2.0)
acceleration_limit_actions(mdp, s) == [2,3,4,7]
s = test_state(20.0, -2.0)
acceleration_limit_actions(mdp, s) == [4,5,6,7]

# test idm_limit_actions
s = test_state(20.0, 0.0)
idm_limit_actions(mdp, s) == [1,2,3,4,5,6,7]
s = test_state(20.0, 0.0, 120.0)
idm_limit_actions(mdp, s) == [1]
s = test_state(20.0, 0.0, 110.0)
idm_limit_actions(mdp, s) == [1,2,3,4,7]

# test joint mask 
s = test_state(20.0, 0.0)
safe_actions(mdp, s) == [2,3,4,5,6,7]
s = test_state(25.0, 0.0, 110.0)
safe_actions(mdp, s) == [1]
s = test_state(20.0, 0.0, 110.0)
safe_actions(mdp, s) == [2,3,4,7]
