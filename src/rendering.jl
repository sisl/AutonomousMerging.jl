function super_render(mdp, s, b=nothing)
    # model = deepcopy(mdp.driver_models[2])
    car_colors=get_car_type_colors(s.scene, mdp.driver_models)
    renderables = [
        mdp.env.roadway,
        (EntityRectangle(entity=x, color=car_colors[x.id]) for x in s.scene)...,
        (VelocityArrow(entity=x) for x in s.scene)...,
        IDOverlay(scene=s.scene, color=colorant"white", font_size=25, x_off=-2, y_off=1),
        MergingNeighborsOverlay(scene=s.scene, target_id=EGO_ID, env=mdp.env),
        DistToMergeOverlay(scene=s.scene, target_id=EGO_ID, env=mdp.env),
        # DistToMergeOverlay(target_id=10, env=mdp.env),
        #    DistToMergeOverlay(target_id=3, env=mdp.env),
        #    DistToMergeOverlay(target_id=6, env=mdp.env),
        # CooperativeIDMOverlay(targetid=10, model=mdp.driver_models[10], textparams=TextParams(size = 20, x = 100, y_start=300)),
        # MaskingOverlay(mdp=mdp, acc=s.ego_info.acc),
        # TextOverlay(text=[@sprintf("Acc %2.2f m/s^2", act), 
        #                     @sprintf("Action %d", a), 
        #                     @sprintf("Time %2.1f", mdp.dt*step)], 
        #             pos=VecE2(300, 20), font_size=20),
        #    CooperativeIDMOverlay(targetid=3, model=model, textparams=TextParams(size = 20, x = 700, y_start=250))
        #    NeighborsOverlay(EGO_ID),
        #    CarFollowingStatsOverlay(EGO_ID), 
            ]
    if b!=nothing 
        push!(renderables, BeliefOverlay(b))
    end

    AutomotiveVisualization.render(renderables,
            #   cam=SceneFollowCamera(10.0), 
            # cam = CarFollowCamera(1, 5.0),s
            # cam = FitToContentCamera(),
            # cam = StaticCamera(VecE2(-25.0, -10.0), 6.0),
            # camera = StaticCamera(position=VecE2(-25.0, -10.0), zoom=15.0),
            #   car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO)
            
            )
end
