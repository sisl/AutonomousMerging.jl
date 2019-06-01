using AutonomousMerging
using AutonomousMerging: py_generate_s, py_initial_state, make_gif
using POMDPs
using Test

@testset "python interface" begin
    mdp = GenerativeMergingMDP(max_cars = 3, dt=0.5)

    s = AutonomousMerging.py_initial_state(mdp)
    feature_list = [s]
    while !isterminal(mdp, s)
        a = 1.5 # constant jerk
        sp = py_generate_s(mdp, s, a)
        s = sp
        push!(feature_list, sp)
    end

    make_gif(mdp, feature_list)
end