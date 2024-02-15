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
    for k in 1:length(sol.paths)
        println(io, "k = $k")
        for i in 1:length(sol.paths[k])
            println(io, "\tflow = $(sol.flows[k][i])")
            println(io, "\tpath = $(sol.paths[k][i])")
        end
    end
end

"""
    is_solution(sol::MCFSolution, pb::MCF)

Check if `sol` is a solution for problem `pb`. Checks if `length(paths) == length(flows) == nk(pb)`, that each path is a valid path on the graph and has the correct endpoints, and that `sum(flows[k]) <= 1`.
"""
function is_solution(sol::MCFSolution, pb::MCF)
    return (length(sol.paths) == length(sol.flows) == nk(pb)) && all(all(is_path(p, pb.graph) for p in sol.paths[k]) for k in 1:nk(pb)) && all(all(is_path(p, pb.demands[k].src, pb.demands[k].dst) for p in sol.paths[k]) for k in 1:nk(pb)) && all(sum(sol.flows[k]) <= 1 for k in 1:nk(pb))
end
