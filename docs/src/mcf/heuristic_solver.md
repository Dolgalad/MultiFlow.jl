# Heuristics

Heuristics for solving MCF problems.

```julia
julia> pb = load("instances/toytests/test1/", edge_dir=:single)
MCF(nv = 7, ne = 10, nk = 3)
	Demand{Int64, Int64}(1, 7, 5)
	Demand{Int64, Int64}(2, 6, 5)
	Demand{Int64, Int64}(3, 7, 5)

julia> solve_shortest_paths(pb)
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 4, 6, 7])
	Demand k = 2
		1.0 on VertexPath{Int64}([2, 4, 6])
	Demand k = 3
		1.0 on VertexPath{Int64}([3, 4, 7])

```

By default `solve_shortest_paths` searches for a path for each demand, taking them in the order in wich they appear in `pb.demands`. The `demand_permutation` keyword arguments allows users to apply a different ordering scheme : 

```julia
julia> solve_shortest_paths(pb, demand_permutation=shuffle)
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 4, 6, 7])
	Demand k = 2
		1.0 on VertexPath{Int64}([2, 4, 6])
	Demand k = 3
		1.0 on VertexPath{Int64}([3, 4, 7])

```


## Index

```@index
Pages = ["heuristic_solver.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["heuristic_solver.jl"]

```

