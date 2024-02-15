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
