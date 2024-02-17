# Solving Compact Formulation

```@eval
push!(LOAD_PATH, "../../..")
using Graphs, MultiFlows, Compose, GraphPlot
g = grid((3,3))
loc_x, loc_y = spring_layout(g)
pb = MCF(g, ones(ne(g)), ones(ne(g)), [Demand(1,9,1.0), Demand(1,6,1.0)])
draw(PNG("grid3x3_problem.png", 16cm, 16cm), mcfplot(pb, loc_x, loc_y))

# default config
sol, ss = solve_compact(pb)
draw(PNG("grid3x3_solution.png", 16cm, 16cm), mcfsolplot(sol, pb, loc_x, loc_y))

# max acceptance
add_demand!(pb, Demand(1,7,1.0))
draw(PNG("grid3x3_problem_1.png", 16cm, 16cm), mcfplot(pb, loc_x, loc_y))
sol, ss = solve_compact(pb, max_acceptance=true)
draw(PNG("grid3x3_solution_ma.png", 16cm, 16cm), mcfsolplot(sol, pb, loc_x, loc_y))
```


Solving Compact Formulation with state of the art LP solver. Interfacing with solver is done using the _JuMP.jl_ package.

First define a simple MCF problem : 
```julia
julia> using Graphs, MultiFlows

julia> g = grid((3,3))
{9, 12} undirected simple Int64 graph

julia> pb = MCF(gr, ones(ne(g)), ones(ne(g)), [Demand(1,9,1.0), Demand(1,6,1.0)])
MCF(nv = 9, ne = 24, nk = 2)
	Demand{Int64, Float64}(1, 9, 1.0)
	Demand{Int64, Float64}(1, 6, 1.0)

julia> solve_compact(pb)
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 2, 5, 8, 9])
	Demand k = 2
		1.0 on VertexPath{Int64}([1, 4, 5, 6])

```

Problem                    |  Solution
:-------------------------:|:-------------------------:
![](grid3x3_problem.png)  |  ![](grid3x3_solution.png)

Adding a demand `1 -> 7` renders the problem infeasible.
```julia
julia> add_demand!(pb, Demand(1, 7, 1.0))
3-element Vector{Demand{Int64, Float64}}:
 Demand{Int64, Float64}(1, 9, 1.0)
 Demand{Int64, Float64}(1, 6, 1.0)
 Demand{Int64, Float64}(1, 7, 1.0)

julia> solve_compact(pb)
ERROR: Infeasible problem
[...]
```

## Max-acceptance
We can solve the Max-acceptance variant of the problem in which a demand may not be routed and incurs a penalty `M = sum(costs(pb))`.

```julia
julia> solve_compact(pb)
MCFSolution
	Demand k = 1
	Demand k = 2
		1.0 on VertexPath{Int64}([1, 2, 3, 6])
	Demand k = 3
		1.0 on VertexPath{Int64}([1, 4, 7])

```
Problem                    |  Solution
:-------------------------:|:-------------------------:
![](grid3x3_problem_1.png)  |  ![](grid3x3_solution_ma.png)


## Index

```@index
Pages = ["compact_solver.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["compact_solver.jl"]

```

