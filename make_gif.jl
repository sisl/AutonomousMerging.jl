using Reel

function make_gif(hist, mdp)
    frames = Frames(MIME("image/png"), fps=6);
    for step in 1:n_steps(hist)
        s = hist.state_hist[step+1]
        a = hist.action_hist[step]
        acc = s.ego_info.acc
        if step < n_steps(hist)+1
            a = hist.action_hist[step]
            act = action_map(mdp, acc, a).a
        else
            a = -1
            act = NaN
        end
        f = AutoViz.render(s.scene, mdp.env.roadway, 
            SceneOverlay[
                        IDOverlay(),
                        MergingNeighborsOverlay(target_id=EGO_ID, env=mdp.env),
                        DistToMergeOverlay(target_id=EGO_ID, env=mdp.env),
                        DistToMergeOverlay(target_id=2, env=mdp.env),
                        TextOverlay(text=[@sprintf("Acc %2.2f m/s^2", act), 
                                            @sprintf("Action %d", a), 
                                            @sprintf("Time %2.1f", mdp.dt*step)], 
                                    pos=VecE2(300, 20), font_size=20)
                            #    NeighborsOverlay(EGO_ID),
                        #    CarFollowingStatsOverlay(EGO_ID), 
                            ],
                car_colors=get_car_type_colors(s0.scene, mdp.driver_models), 
                #  cam=FitToContentCamera(0.0))
                cam=StaticCamera(VecE2(-25.0, -10.0), 6.0))
            #   cam=CarFollowCamera(EGO_ID, 10.0), 
            #   car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
        push!(frames, f)
    end

    write("out.gif", frames)
end