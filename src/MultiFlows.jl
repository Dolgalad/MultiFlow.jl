module MultiFlows

using DataFrames
using CSV
using Graphs
import Graphs: nv, ne, edges, add_edge!, weights
using SimpleWeightedGraphs
using SparseArrays
using JuMP
using StatsBase
using FileIO
using JLD2
using ProgressBars

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
    convert_features,
    zero_features,
    # Paths
    VertexPath,
    path_weight,
    edge_indices,
    is_path,
    path_from_edge_indices,
    shortest_paths,
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
    add_demand!,
    demand_amounts,
    demand_endpoints,
    aggregate_demands,
    get_path,
    # MCF solutions
    MCFSolution,
    is_solution,
    arc_flow_value,
    is_feasible,
    used_capacity,
    available_capacity,
    objective_value,
    paths,
    paths_from_arc_flow_values,
    solution_from_arc_flow_values,
    path_capacity,
    load_solution,
    # reading/writing
    load,
    load_csv,
    save,
    is_instance_dir,
    # plotting
    mcfplot,
    mcfsolplot,
    # solver statistics
    SolverStatistics,
    # heuristic solver
    solve_shortest_paths,
    # compact solver
    solve_compact,
    # column generation
    MCFRestrictedMasterProblem,
    MCFPricingProblem,
    solve!,
    add_column!,
    update_pricing_problem!,
    solve_column_generation,
    # instance generation
    random_demand_amounts,
    random_demand_endpoints,
    shake,
    non_saturated_path_exists,
    saturate,
    generate_example,
    make_dataset,
    # Sparsification
    AbstractSparsifier,
    SPSparsifier,
    sparsify





#MultiFlows
include("feature_graph.jl")
include("path.jl")
include("mcf/demand.jl")
include("mcf/mcf.jl")
include("mcf/solution.jl")
include("mcf/io.jl")
include("mcf/plot.jl")
include("solver_stats.jl")
include("solvers/heuristic.jl")
include("solvers/compact.jl")
include("solvers/column_generation.jl")

include("generators.jl")
include("sparsify.jl")

end # module MultiFlows
