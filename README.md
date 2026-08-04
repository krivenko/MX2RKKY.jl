# MX2RKKY.jl

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://krivenko.github.io/MX2RKKY.jl)
[![CI](https://github.com/krivenko/MX2RKKY.jl/actions/workflows/CI.yml/badge.svg)](
https://github.com/krivenko/MX2RKKY.jl/actions/workflows/CI.yml)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](
https://github.com/JuliaTesting/Aqua.jl)

Copyright (C) 2024-2026 Igor Krivenko

Classical Monte Carlo simulations of the triangular-lattice XY antiferromagnet with
a chiral RKKY interaction, as realized by magnetic adatoms on a transition metal
dichalcogenide (MX2) monolayer.

This code was used to produce numerical data presented in

> Valley-controlled chiral magnetism in transition metal dichalcogenide monolayers,
> I. S. Krivenko, V. N. Mantsevich, D. S. Smirnov,
> arXiv preprint [arXiv:2608.02260](https://arxiv.org/abs/2608.02260)

## Installation

In Julia REPL:
```julia
] add https://github.com/krivenko/MX2RKKY.jl
```

## Simulation scripts

Julia scripts used to run simulations on the [NHR@ZIB](https://nhr.zib.de/en/) "Lise" system
can be found in [scripts/NHR@ZIB-Lise](scripts/NHR@ZIB-Lise).