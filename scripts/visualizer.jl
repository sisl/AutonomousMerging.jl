
using Blink
using Interact

model = deepcopy(mdp.driver_models[2])
ui = @manipulate for step in 1:n_steps(hist)+1
    s = state_hist(hist)[step]
    # b = hist.belief_hist[step]
    b = nothing
    acc = s.ego_info.acc
    if step < n_steps(hist)+1
        a = collect(action_hist(hist))[step]
        act = action_map(mdp, acc, a).a
    else
        a = -1
        act = NaN
    end
    if step == 1
        reset_hidden_state!(model)
        # @show model.a
    end
    if b!== nothing
        super_render(mdp, s, b)
    else
        super_render(mdp, s)
    end
    # model.other_acc = s.ego_info.acc
    # AutomotiveVisualization.render(s.scene, mdp.env.roadway, 
    #       SceneOverlay[
    #                    IDOverlay(color=colorant"white",font_size=20),
    #                    MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
    #                    DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
    #                    DistToMergeOverlay(target_id=2, env=mdp.env),
    #                 #    DistToMergeOverlay(target_id=3, env=mdp.env),
    #                 #    DistToMergeOverlay(target_id=6, env=mdp.env),
    #                    CooperativeIDMOverlay(targetid=2, model=model, textparams=TextParams(size = 20, x = 100, y_start=300)),
    #                    MaskingOverlay(mdp=mdp, acc=s.ego_info.acc),
    #                    TextOverlay(text=[@sprintf("Acc %2.2f m/s^2", act), 
    #                                      @sprintf("Action %d", a), 
    #                                      @sprintf("Time %2.1f", mdp.dt*step)], 
    #                                pos=VecE2(300, 20), font_size=20),
    #                 #    CooperativeIDMOverlay(targetid=3, model=model, textparams=TextParams(size = 20, x = 700, y_start=250))
    #                    #    NeighborsOverlay(EGO_ID),
    #                 #    CarFollowingStatsOverlay(EGO_ID), 
    #                     ],
    #     #   cam=SceneFollowCamera(10.0), 
    #     # cam = CarFollowCamera(1, 5.0),
    #     # cam = FitToContentCamera(0.0),
    #     cam = StaticCamera(VecE2(-25.0, -10.0), 6.0),
    #     #   car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO)
    #     car_colors=get_car_type_colors(s0.scene, mdp.driver_models)
    #     )
end

w = Window()
body!(w, ui);
