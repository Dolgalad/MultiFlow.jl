# MultiFlows.jl
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dolgalad.github.io/MultiFlows.jl/dev/)
[![Documentation](https://github.com/Dolgalad/MultiFlows.jl/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/Dolgalad/MultiFlows.jl/actions/workflows/documentation.yml)

## Overview
_MultiFlows.jl_ is a Julia package for solving Multi-Commodity Flow problems. The package implements a collection of solver methods : 

- direct solver implementation (using _JuMP.jl_)
- column generation frameworks
- machine learning based solvers

## Basic usage
Load an Multi-Commodity Flow instance from a directory `dirname`. The directory should contain two files : a `link.csv` file containing graph edge data and a `service.csv` file containing demand data.

```julia
julia> using MultiFlows

julia> pb = load(dirname)
MCF(nv = 7, ne = 10, nk = 3)
	Demand{Int64, Int64}(1, 7, 5)
	Demand{Int64, Int64}(2, 6, 5)
	Demand{Int64, Int64}(3, 7, 5)

```

This package offers a collection of solvers for these problems.

```julia
julia> sol, ss = solve_compact(pb); # solve compact formulation

julia> sol, ss = solve_column_generation(pb) # solve by column generation

julia> sol
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 4, 7])
	Demand k = 2
		1.0 on VertexPath{Int64}([2, 4, 6])
	Demand k = 3
		1.0 on VertexPath{Int64}([3, 4, 6, 7])

```

## Documentation

The full documentation is available at [GitHub Pages](https://dolgalad.github.io/MultiFlows.jl/dev/). 
