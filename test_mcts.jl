

solver = DPWSolver(depth = 20,
                   exploration_constant = 1.0,
                   n_iterations = 1000, 
                   k_state  = 2.0, 
                   alpha_state = 0.2, 
                   keep_tree = true,
                   enable_action_pw = false,
                   rng = rng, 
                    tree_in_info = true,
                  estimate_value = RolloutEstimator(FunctionPolicy(s->4))
                   )

policy = solve(solver, mdp)

s0 = initialstate(mdp, rng)
hr = HistoryRecorder(rng = rng, max_steps=100)
hist = simulate(hr, mdp, policy, s0)

frames = Frames(MIME("image/png"), fps=4)
for step in 1:n_steps(hist)
    s = hist.state_hist[step+1]
    a = hist.action_hist[step]
    f = AutoViz.render(s, mdp.env.roadway, 
          SceneOverlay[IDOverlay()],
          cam=FitToContentCamera(0.0), 
          car_colors = Dict{Int64, Colorant}(1 => COLOR_CAR_EGO))
    push!(frames, f)
end

write("out.gif", frames)