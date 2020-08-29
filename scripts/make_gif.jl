function make_gif(hist, mdp, belief_hist=nothing)
    frames = Frames(MIME("image/png"), fps=6);
    for step in 1:n_steps(hist)+1
        # b = nothing
        # if belief
        #     b = hist.belief_hist[step +1]
        # end
        s = state_hist(hist)[step]
        # a = collect(action_hist(hist))[step]
        # acc = s.ego_info.acc
        if step < n_steps(hist)+1
            # a = collect(action_hist(hist))[step]
            # act = action_map(mdp, acc, a).a
        else
            a = -1
            act = NaN
        end
        if belief_hist !== nothing 
            f = super_render(mdp, s, belief_hist[step])
        else
            f = super_render(mdp, s)
        end
        # f = AutomotiveVisualization.render(s.scene, mdp.env.roadway, 
        #     SceneOverlay[
        #                 IDOverlay(),
        #                 MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
        #                 DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
        #                 DistToMergeOverlay(target_id=2, env=mdp.env),
        #                 CooperativeIDMOverlay(targetid=2, model=model, textparams=TextParams(size = 20, x = 100, y_start=300)),
        #                 TextOverlay(text=[@sprintf("Acc %2.2f m/s^2", act), 
        #                                     @sprintf("Action %d", a), 
        #                                     @sprintf("Time %2.1f", mdp.dt*step)], 
        #                             pos=VecE2(300, 20), font_size=20)
        #                     #    NeighborsOverlay(EGO_ID),
        #                 #    CarFollowingStatsOverlay(EGO_ID), 
        #                     ],
        #         car_colors=get_car_type_colors(s0.scene, mdp.driver_models), 
        #         #  cam=FitToContentCamera(0.0))
        #         cam=StaticCamera(VecE2(-25.0, -10.0), 6.0))
            #   cam=CarFollowCamera(EGO_ID, 10.0), 
            #   car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
        push!(frames, f)
    end

    write("out.gif", frames)
end
