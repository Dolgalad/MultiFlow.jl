"""
    MultiFlows

MultiFlows package documentation.
"""
module MultiFlows

using DataFrames
using CSV
using Graphs
import Graphs: nv, ne, edges, add_edge!, weights
using SimpleWeightedGraphs
using SparseArrays

export 
    # Feature graph
    FeatureDiGraphEdge,
    FeatureDiGraph,
    feature_dim,
    feature_matrix,
    nk,
    nv,
    ne,
    scale_features,
    double_edges!,
    edge_features,
    edge_index_matrix,
    # Paths
    VertexPath,
    path_weight,
    edge_indices,
    is_path,
    # MCF
    Demand,
    MCF,
    weight_matrix,
    cost_matrix,
    capacity_matrix,
    scale_demands,
    scale,
    normalize,
    has_demand,
    costs,
    capacities,
    demands,
    # MCF solutions
    MCFSolution,
    is_solution,
    arc_flow_value,
    is_feasible,
    used_capacity,
    objective_value,
    paths,
    paths_from_arc_flow_values,
    solution_from_arc_flow_values,
    # reading/writing
    load,
    load_csv,
    save,
    is_instance_dir,
    # plotting
    mcfplot,
    mcfsolplot,
    # heuristic solver
    solve_shortest_paths,
    # compact solver
    solve_compact

#MultiFlows
include("feature_graph.jl")
include("path.jl")
include("mcf/demand.jl")
include("mcf/mcf.jl")
include("mcf/solution.jl")
include("mcf/io.jl")
include("mcf/plot.jl")
include("mcf/heuristic_solver.jl")
include("compact_solver.jl")
#include("graph_utils.jl")

end # module MultiFlows
