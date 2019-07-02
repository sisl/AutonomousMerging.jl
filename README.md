# AutonomousMerging.jl

[![Build Status](https://travis-ci.org/sisl/AutonomousMerging.jl.svg?branch=master)](https://travis-ci.org/sisl/AutonomousMerging.jl)
[![Coverage Status](https://coveralls.io/repos/github/sisl/AutonomousMerging.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/AutonomousMerging.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://sisl.github.io/AutonomousMerging.jl/latest)

Maintainer: Maxime Bouton (boutonm@stanford.edu)

This code base implements an autonomous merging scenario using [POMDPs.jl](https:://github.com/JuliaPOMDP/POMDPs.jl)
and [AutomotiveDrivingModels.jl](https://github.com/sisl/AutomotiveDrivingModels.jl) described in the paper:
M. Bouton, A. Nakhaei, K. Fujimura, M. J. Kochenderfer, "Cooperation-Aware Reinforcement Learning for Merging in Dense Traffic," in *IEEE Conference on Intelligent Transportation Systems (ITSC)*, 2019. [ArXiv](https://arxiv.org/abs/1906.11021)

<img src="demo.gif" width="700"/>

## Installation

To install this julia package, add the sisl registry:
```julia
] registry add https://github.com/sisl/Registry
```
Then add this package using its url:
```julia
] add https://github.com/sisl/AutonomousMerging.jl
```

## Folder structure

- `src`: the source code containing the MDP definition, the C-IDM model definition, some feature extraction helpers,
  as well as some rendering helpers
- `test`: the tests run by Travis are defined in `runtests.jl`. The other file are interactive tests for debugging using visualizations.
- `scripts`: contains training, simulation, and analysis scripts


## Documentation

All the objects exported by the package have docstring that can be consulted using the julia command `?`.
For further documentation we refer the user to [POMDPs.jl](https:://github.com/JuliaPOMDP/POMDPs.jl)
and [AutomotiveDrivingModels.jl](https://github.com/sisl/AutomotiveDrivingModels.jl)
