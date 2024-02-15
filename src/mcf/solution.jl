"""
    MCFSolution

Container representing the solution of an [`MCF`](@ref) problem. Contains a list of paths such that `sol.paths[k]` contains the paths used to route demand `k` and `flows[k]` are the flow values
"""
struct MCFSolution
    paths::Vector{Vector{AbstractPath}}
    flows::Vector{Vector{Float64}}
end
