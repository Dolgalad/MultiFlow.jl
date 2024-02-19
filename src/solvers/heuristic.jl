"""
    solve_shortest_paths(pb::MCF)

Solve an MCF instance by taking the shortest path for each demand.
"""
function solve_shortest_paths(pb::MCF{T,N};
                              demand_permutation::Function=identity,
    ) where {T,N}
    paths = [VertexPath{T}[] for _ in 1:nk(pb)]
    flows = [Float64[] for _ in 1:nk(pb)]
    available_capacities = capacities(pb)
    edge_costs = costs(pb)
    bigM = sum(edge_costs)
    # apply demand index permutation
    demand_idx = demand_permutation(1:nk(pb))
    for k in demand_idx
        d = pb.demands[k]
        edge_costs = costs(pb)
        edge_costs[available_capacities .< d.amount] .= bigM
        g = SimpleWeightedDiGraph(pb.graph.srcnodes, pb.graph.dstnodes, edge_costs)
        ds = dijkstra_shortest_paths(g, d.src)
        p = VertexPath(enumerate_paths(ds, d.dst))
        if !Base.isempty(p) && sum(edge_costs[edge_indices(p, pb.graph)]) < bigM
            push!(paths[k], p)
            push!(flows[k], 1.0)
        end
        # update edge costs and available capacities
        available_capacities[edge_indices(p, pb.graph)] .-= d.amount
    end
    return MCFSolution(paths, flows)
end
