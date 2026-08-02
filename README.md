# Monte-Carlo simulation of RKKY-induced chirality of two-dimensional antiferromagnet

[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](
https://github.com/JuliaTesting/Aqua.jl)

Copyright (C) 2024-2026 Igor Krivenko

This repository contains a Julia implementation of the Monte Carlo algorithm
used to study magnetic order in an XY-model on a triangular lattice with a long-range
chiral interaction between local magnetic moments.

This code was used to produce numerical data presented in

> Valley-controlled chiral magnetism in transition metal dichalcogenide monolayers
> I. S. Krivenko, V. N. Mantsevich, D. S. Smirnov
> arXiv preprint [arXiv:2608.00000](https://arxiv.org/abs/2608.00000)

## Installation

In Julia REPL:
```julia
] add https://github.com/krivenko/MX2RKKY.jl
```