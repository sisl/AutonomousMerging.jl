# About 

This is the documentation for [AutonomousMerging.jl](https://github.com/sisl/AutonomousMerging.jl). 

The environment is defined by `MergingEnvironment`. This package exports two MDP types:
    - `MergingMDP`, a discrete MDP implemented using the explicit interface of POMDPs.jl with only two traffic participants
    - `GenerativeMDP`, a continuous state MDP with a traffic flow implemented using the generative interface of POMDPs.jl

For more information on the explicit vs generative definition of MDPs read:
- http://juliapomdp.github.io/POMDPs.jl/latest/explicit/
- http://juliapomdp.github.io/POMDPs.jl/latest/generative/

```contents
```