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
    Base.:(==)(s1::MCFSolution, s2::MCFSolution)

Check solution equality. Checks `s1.paths==s2.paths` and `s1.flows==s2.flows`.
"""
function Base.:(==)(s1::MCFSolution, s2::MCFSolution)
    return (s1.paths == s2.paths) && (s1.flows == s2.flows)
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
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,1.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> used_capacity(sol, pb)
14-element Vector{Float64}:
 0.0
 1.0
 0.0
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 0.0
 1.0
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
    available_capacity(sol::MCFSolution, pb::MCF)

Compute available capacities on the graph for a given solution. 

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,1.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> available_capacity(sol, pb)
14-element Vector{Float64}:
 1.0
 0.0
 1.0
 1.0
 1.0
 0.0
 1.0
 1.0
 1.0
 1.0
 0.0
 1.0
 1.0
 1.0

```

"""
function available_capacity(sol::MCFSolution, pb::MCF)
    return capacities(pb) .- used_capacity(sol, pb)
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
    return sum(pb.demands[k].amount * sum(sol.flows[k][i] * path_weight(sol.paths[k][i], pb.graph) for i in 1:length(sol.flows[k])) for k in 1:nk(pb) if !Base.isempty(sol.paths[k]))
end

"""
    paths(sol::MCFSolution)

Return all paths used in the solution with non-zero flow.

# Example
```jldoctest; setup = :(using Graphs)
julia> sol = MCFSolution([[VertexPath([1,2]), VertexPath([1,4,5,2])]], [[.5, .5]]);

julia> paths(sol)
2-element Vector{VertexPath{Int64}}:
 VertexPath{Int64}([1, 2])
 VertexPath{Int64}([1, 4, 5, 2])

```
"""
function paths(sol::MCFSolution)
    return vcat([[sol.paths[k][i] for i in 1:length(sol.paths[k]) if sol.flows[k][i]>0] for k in 1:length(sol.paths)]...)
end

"""
    paths(sol::MCFSolution, k::Int64)

Return paths used in the solution with non-zero flow for demand `k`.

# Example
```jldoctest; setup = :(using Graphs)
julia> sol = MCFSolution([[VertexPath([1,2])], [VertexPath([1,4,5,2])]], [[.5], [.5]]);

julia> paths(sol, 1)
1-element Vector{VertexPath{Int64}}:
 VertexPath{Int64}([1, 2])

```
"""
function paths(sol::MCFSolution, k::Int64)
    return [sol.paths[k][i] for i in 1:length(sol.paths[k]) if sol.flows[k][i]>0]
end


"""
    Graphs.has_edge(sol::MCFSolution, s::T, t::T) where {T}

Check if the solution uses edge `(s,t)`.

# Example
```jldoctest; setup = :(using Graphs)
julia> sol = MCFSolution([[VertexPath([1,2])], [VertexPath([1,4,5,2])]], [[.5], [.5]]);

julia> has_edge(sol, 1, 2)
true

julia> has_edge(sol, 5, 4)
false

```

"""
function Graphs.has_edge(sol::MCFSolution, s::T, t::T) where {T}
    return any([has_edge(p, s, t) for p in paths(sol)])
end

"""
    Graphs.has_edge(sol::MCFSolution, k::Int64, s::T, t::T) where {T}

Check if the solution uses edge `(s,t)` for demand `k`.

# Example
```jldoctest; setup = :(using Graphs)
julia> sol = MCFSolution([[VertexPath([1,2])], [VertexPath([1,4,5,2])]], [[.5], [.5]]);

julia> has_edge(sol, 1, 1, 2)
true

julia> has_edge(sol, 1, 5, 4)
false

```

"""
function Graphs.has_edge(sol::MCFSolution, k::Int64, s::T, t::T) where {T}
    return any([has_edge(p, s, t) for p in paths(sol, k)])
end

"""
    paths_from_arc_flow_values(x::Vector, k::Int64, pb::MCF)

Compute paths and flows from arc flow values.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,2.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> x = arc_flow_value(sol, pb)
1×14 Matrix{Float64}:
 0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0

julia> paths_from_arc_flow_values(x[1,:], 1, pb)
(VertexPath[VertexPath{Int64}([1, 4, 5, 2])], [1.0])
```
  
"""
function paths_from_arc_flow_values(x::Vector, k::Int64, pb::MCF)
    arc_flows = copy(x)
    demand = pb.demands[k]
    temp_costs = Float64.(costs(pb))
    M = typemax(eltype(temp_costs))
    #println("cost type ", eltype(temp_costs), ", ", M)
    temp_costs[arc_flows .== 0] .= M
    paths = VertexPath[]
    flows = Float64[]
    while !all(temp_costs .== M)
        g = SimpleWeightedDiGraph(pb.graph.srcnodes, pb.graph.dstnodes, temp_costs)
        p = VertexPath(enumerate_paths(dijkstra_shortest_paths(g, demand.src), demand.dst))
        if Base.isempty(p)
            break
        end
        # get flow
        path_flow = minimum(arc_flows[edge_indices(p, pb.graph)])
        push!(paths, p)
        push!(flows, path_flow)
        # update the arc flow vector
        arc_flows[edge_indices(p, pb.graph)] .-= path_flow
        temp_costs[arc_flows .== 0] .= M
        #println(sum(temp_costs .== M))
    end
    return paths, flows
end

"""
    from_arc_flow_values(x::AbstractMatrix{Float64}, pb::MCF)

Compute solution paths from ``x_a^k`` values.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(14), ones(14), [Demand(1,2,2.)]);

julia> sol = MCFSolution([[VertexPath([1,4,5,2])]], [[1.]]);

julia> x = arc_flow_value(sol, pb)
1×14 Matrix{Float64}:
 0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0

julia> solution_from_arc_flow_values(x, pb)
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 4, 5, 2])

```
 
"""
function solution_from_arc_flow_values(x::AbstractMatrix{Float64}, pb::MCF)
    # check dimension
    if size(x) != (nk(pb), ne(pb))
        throw(DimensionMismatch("Solution dimension does not match problem. Expected $((nk(pb), ne(pb))) and got $(size(x))"))
    end
    sol_paths, sol_flows = [], []
    for k in 1:nk(pb)
        # get paths from vector of arc-flows
        paths, flows = paths_from_arc_flow_values(x[k,:], k, pb)
        push!(sol_paths, paths)
        push!(sol_flows, flows)
    end
    return MCFSolution(sol_paths, sol_flows)
end

"""
    path_capacity(p::VertexPath, pb::MCF)

Get minimum capacity of arcs belonging to path `p`.

# Example
```jldoctest; setup = :(using Graphs; )
julia> pb = MCF(grid((3,2)), ones(Int64,7), 1:7, [Demand(1,2,2)]);

julia> p = get_path(pb, 1, 6)
VertexPath{Int64}([1, 2, 5, 6])

julia> path_capacity(p, pb)
1

```
"""
path_capacity(p::VertexPath, pb::MCF) = path_weight(p, pb.graph, aggr=minimum, dstmx=capacity_matrix(pb))

"""
    FileIO.save(sol::MCFSolution, filename::String)

Save solution to file. Uses _JLD2.jl_ for writing data to file.

# Example
```jldoctest; setup = :(using Graphs,FileIO; )
julia> pb = MCF(grid((3,2)), ones(Int64,7), 1:7, [Demand(1,2,2)]);

julia> sol,_ = solve_column_generation(pb);

julia> FileIO.save(sol, "sol.jld2")

julia> isfile("sol.jld2")
true

```
"""
function FileIO.save(sol::MCFSolution, filename::String)
    # maximum number of paths for a demand
    max_n_paths = maximum(length(pths) for pths in sol.paths)
    nK = length(sol.paths)
    nV = maximum(maximum(maximum(p.vertices) for p in pths) for pths in sol.paths)
    # initialize demand path and flow tensors
    demand_path_tensor = zeros(Int64, nK, max_n_paths, nV)
    demand_flow_tensor = zeros(Float64, nK, max_n_paths)
    for k in 1:nK
        for (i,p) in enumerate(sol.paths[k])
            demand_path_tensor[k,i,p.vertices] .= 1:length(p.vertices)
            demand_flow_tensor[k,i] = sol.flows[k][i]
        end
    end
    jldsave(filename; paths=demand_path_tensor, flows=demand_flow_tensor)
end

"""
    load_solution(filename::String)

Load [`MCFSolution`](@ref) from JLD2 data file.

# Example
```jldoctest; setup = :(using FileIO,Graphs; pb = MCF(grid((3,2)), ones(Int64,7), 1:7, [Demand(1,2,2)]);(sol,_) = solve_column_generation(pb); FileIO.save(sol,"sol.jld2"))
julia> sol = load_solution("sol.jld2")
MCFSolution
	Demand k = 1
		0.5 on VertexPath{Int64}([1, 2])
		0.5 on VertexPath{Int64}([1, 4, 5, 2])

```
"""
function load_solution(filename::String)
    data = FileIO.load(filename)
    nK = size(data["paths"], 1)
    paths = [VertexPath[] for k in 1:nK]
    flows = [Float64[] for k in 1:nK]

    for k in 1:nK
        for i in 1:size(data["paths"], 2)
            if any(data["paths"][k,i,:] .> 0)
                vidx = findall(>(0), data["paths"][k,i,:])
                push!(paths[k], VertexPath(vidx[sortperm(data["paths"][k,i,vidx])]))
                push!(flows[k], data["flows"][k,i])
            end
        end
    end
    return MCFSolution(paths, flows)
end


