"""
    MCFSolution

Container representing the solution of an [`MCF`](@ref) problem. Contains a list of paths such that `sol.paths[k]` contains the paths used to route demand `k` and `flows[k]` are the flow values
"""
struct MCFSolution
    paths::Vector{Vector{AbstractPath}}
    flows::Vector{Vector{Float64}}
end

"""
    Base.show(sol::MCFSolution)

Show solutions.
"""
function Base.show(io::IO, sol::MCFSolution)
    println(io, "MCFSolution")
    for k in 1:length(sol.paths)
        println(io, "\tDemand k = $k")
        for i in 1:length(sol.paths[k])
            println(io, "\t\t$(sol.flows[k][i]) on $(sol.paths[k][i])")
        end
    end
end

"""
    is_solution(sol::MCFSolution, pb::MCF)

Check if `sol` is a solution for problem `pb`. Checks if `length(paths) == length(flows) == nk(pb)`, that each path is a valid path on the graph and has the correct endpoints, and that `sum(flows[k]) <= 1`.

# Example
```jldoctest mcfsol; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,2,1.)])
MCF(nv = 6, ne = 14, nk = 1)
	Demand{Int64, Float64}(1, 2, 1.0)

julia> sol = MCFSolution([[VertexPath([1,2])]], [[1.]])
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 2])

julia> is_solution(sol, pb)
true

julia> sol = MCFSolution([[VertexPath([1,2]), VertexPath([1,4,5,2])]], [[.5, .5]])
MCFSolution
	Demand k = 1
		0.5 on VertexPath{Int64}([1, 2])
		0.5 on VertexPath{Int64}([1, 4, 5, 2])

julia> is_solution(sol, pb)
true

```

Lets create invalid solutions : 
```jldoctest mcfsol
julia> # path is not valid

julia> is_solution(MCFSolution([[VertexPath([1,3])]], [[1.]]), pb)
false

julia> # total flow for demand 1 is greater than 1

julia> is_solution(MCFSolution([[VertexPath([1,2])]], [[2.]]), pb)
false
```
"""
function is_solution(sol::MCFSolution, pb::MCF)
    return (length(sol.paths) == length(sol.flows) == nk(pb)) && all(all(is_path(p, pb.graph) for p in sol.paths[k]) for k in 1:nk(pb)) && all(all(is_path(p, pb.demands[k].src, pb.demands[k].dst) for p in sol.paths[k]) for k in 1:nk(pb)) && all(sum(sol.flows[k]) <= 1 for k in 1:nk(pb))
end

"""
    arc_flow_value(sol::MCFSolution, pb::MCF)

Returns a `(nk(pb), ne(pb))` sized matrix where coefficients `x[k,a]` is the amount of flow for demand `k` circulating through `a`.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), collect(1:7.), collect(10:16.), [Demand(1,2,1.)]);

julia> sol = MCFSolution([[VertexPath([1,2]), VertexPath([1,4,5,2])]], [[.5, .5]]);

julia> arc_flow_value(sol, pb)
1×14 Matrix{Float64}:
 0.5  0.5  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0

```
"""
function arc_flow_value(sol::MCFSolution, pb::MCF)
    if !is_solution(sol, pb)
        throw(ArgumentError("Provided solution is not valid for the problem"))
    end
    x = zeros(nk(pb), ne(pb))
    for k in 1:nk(pb)
        for i in 1:length(sol.paths[k])
            x[k,edge_indices(sol.paths[k][i], pb.graph)] .+= sol.flows[k][i]
        end
    end
    return x
end

"""
    used_capacity(sol::MCFSolution, pb::MCF)

Compute total edge capacity used by the solution.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,2.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> used_capacity(sol, pb)
14-element Vector{Float64}:
 0.0
 2.0
 0.0
 0.0
 0.0
 2.0
 0.0
 0.0
 0.0
 0.0
 2.0
 0.0
 0.0
 0.0

```
"""
function used_capacity(sol::MCFSolution, pb::MCF)
  used_cap = zeros(ne(pb))
    for k in 1:nk(pb)
        demand = pb.demands[k]
        for i in 1:length(sol.paths[k])
            used_cap[edge_indices(sol.paths[k][i], pb.graph)] .+= sol.flows[k][i] * demand.amount
        end
    end
    return used_cap
end

"""
    is_feasible(sol::MCFSolution, pb::MCF)

Check if the solution is feasible, has to be a valid solution for the problem and the total amount circulating on the graph must not be greater than the edge capacities and all demands must be routed entirely.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,2.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> is_feasible(sol, pb)
false

julia> sol = MCFSolution([[VertexPath([1,2]), VertexPath([1,4,5,2])]], [[.5, .5]]);

julia> is_feasible(sol, pb)
true
```

"""
function is_feasible(sol::MCFSolution, pb::MCF)
    return all(used_capacity(sol, pb) .<= capacities(pb)) && all(sum(sol.flows[k])==1 for k in 1:nk(pb))
end

"""
    objective_value(sol::MCFSolution, pb::MCF)

Compute the objective value of `sol`. 

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,2.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> objective_value(sol, pb)
6.0

julia> sol = MCFSolution([[VertexPath([1,2]), VertexPath([1,4,5,2])]], [[.5, .5]]);

julia> objective_value(sol, pb)
4.0
```

"""
function objective_value(sol::MCFSolution, pb::MCF)
    return sum(pb.demands[k].amount * sum(sol.flows[k][i] * path_weight(sol.paths[k][i], pb.graph) for i in 1:length(sol.flows[k])) for k in 1:nk(pb))
end

