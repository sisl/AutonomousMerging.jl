"""
    py_initial_state(mdp::GenerativeMergingMDP)
sample an initial state, returns the observation vector associated to the initial state.
This function is intended to be called by python pyjulia.
Note that it is using the GLOBAL RNG to sample the initial state.
"""
function py_initial_state(mdp::GenerativeMergingMDP, seed::Int64=1)
    s0 = initialstate(mdp, MersenneTwister(seed))
    return scene_to_vec(mdp, s0)
end


"""
    py_generate_s(mdp::GenerativeMergingMDP, o::Vector{Float64}, acc::Float64)
It performs one simulation step.
This funciton is intended to be called by python pyjulia.
"""
function py_generate_s(mdp::GenerativeMergingMDP, o::Vector{Float64}, acc::Float64)
    s = vec_to_scene(mdp, o)
    scene = deepcopy(s.scene)
    ego_acc = LaneFollowingAccel(acc)
    models = init_driver_models(mdp, s.ego_info.acc)
    models[EGO_ID].a = ego_acc
    # mdp.driver_models[EGO_ID].a = ego_acc
    acts = Vector{LaneFollowingAccel}(undef, length(scene))

    # call driver models 
    for i=EGO_ID+1:EGO_ID+mdp.n_cars_main
        models[i].other_acc = s.ego_info.acc
    end
    get_actions!(acts, scene, mdp.env.roadway, models)

    # update scene 
    tick!(scene, mdp.env.roadway, acts, mdp.dt, true)

    # clamp speed 
    for (i, veh) in enumerate(scene)
        # scene[i] = clamp_speed(mdp.env, veh)
        scene[i] = wrap_around(mdp.env, scene[i]) 
    end
    sp = AugScene(scene, (acc=acc,))
    return scene_to_vec(mdp, sp)
end

function init_driver_models(mdp::GenerativeMergingMDP, acc::Float64)
    models = deepcopy(mdp.driver_models)
    # call driver models 
    for i=EGO_ID+1:EGO_ID+mdp.n_cars_main
        models[i].other_acc = acc
    end
    return models
end


function POMDPs.isterminal(mdp::GenerativeMergingMDP, o::Vector{Float64})
    s = vec_to_scene(mdp, o)
    return isterminal(mdp, s)
end

"""
    scene_to_vec(mdp::GenerativeMergingMDP, s::AugScene)

convert a scene object to a feature vector

# components:
- ego distance to merge point (start negative, 0.0 is the merge point)
- ego speed
- ego acceleration
for each other car 
    - car position on main lane
    - car speed
- collision indicator
"""
function scene_to_vec(mdp::GenerativeMergingMDP, s::AugScene)
    n_features = 2*mdp.n_cars_main + 4
    features = zeros(n_features)
    ego = get_by_id(s.scene, EGO_ID)
    features[1] = dist_to_merge(mdp.env, ego)
    features[2] = ego.state.v
    features[3] = s.ego_info.acc
    @assert s.scene[1].id == EGO_ID
    for i=2:length(s.scene)
        veh = s.scene[i]
        features[2*i] = veh.state.posF.s
        features[2*i+1] = veh.state.v
    end
    features[end] = is_crash(s.scene)
    return features 
end

"""
    vec_to_scene(mdp::GenerativeMergingMDP, features::Vector{Float64})

converts a feature vector to a AugScene object
"""
function vec_to_scene(mdp::GenerativeMergingMDP, features::Vector{Float64})
    scene = Scene() 
    if features[1] < 0.0
        lane_ego = merge_lane(mdp.env)
        s_ego = get_end(lane_ego) + features[1]
    else
        lane_ego = main_lane(mdp.env)
        s_ego = mdp.env.roadway[mdp.env.merge_index].s + features[1]
    end
    
    v_ego = features[2]
    acc_ego = features[3]
    ego = Vehicle(vehicle_state(s_ego, lane_ego, v_ego, mdp.env.roadway), VehicleDef(), EGO_ID)
    push!(scene, ego)
    for i=2:mdp.n_cars_main+1
        veh = Vehicle(vehicle_state(features[2*i], main_lane(mdp.env), features[2*i+1], mdp.env.roadway),
                      VehicleDef(), i)
        push!(scene, veh)
    end
    return AugScene(scene, (acc=acc_ego,))
end

## Rendering

"""
    AutoViz.render(mdp::GenerativeMergingMDP, features::Vector{Float64}, filename=nothing)
render a scene as a CairoSurface. If filename is passed in, write the surface to a png file.
"""
function AutoViz.render(mdp::GenerativeMergingMDP, features::Vector{Float64}, filename=nothing)
    s = vec_to_scene(mdp, features)
    c = super_render(mdp, s)
    if filename != nothing 
        write_to_png(c, filename)
    end
    return c
end

"""
    make_gif(mdp::GenerativeMergingMDP, feature_list::Vector{Vector{Float64}}, filename="out.gif", fps=6)
create "filename.gif" that visualizes the list of scenes.
"""
function make_gif(mdp::GenerativeMergingMDP, feature_list::Vector{Vector{Float64}}, filename="out.gif", fps=6)
    frames = Frames(MIME("image/png"), fps=fps);
    for step in 1:length(feature_list)
        s = vec_to_scene(mdp, feature_list[step])
        c = super_render(mdp, s)
        push!(frames, c)
    end
    write(filename, frames)
end