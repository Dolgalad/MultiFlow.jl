"""
    MultiFlows

MultiFlows package documentation.
"""
module MultiFlows

using DataFrames
using CSV
using Graphs
import Graphs: nv, ne, edges
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
    # MCF
    Demand,
    MCF,
    weight_matrix,
    cost_matrix,
    capacity_matrix,
    scale_demands,
    scale,
    normalize,
    # reading/writing
    load,
    load_csv,
    save,
    is_instance_dir,
    # plotting
    mcfplot

#MultiFlows
include("feature_graph.jl")
include("path.jl")
include("mcf/demand.jl")
include("mcf/mcf.jl")
include("mcf/solution.jl")
include("mcf/io.jl")
include("mcf/plot.jl")
#include("paths.jl")
#include("graph_utils.jl")

end # module MultiFlows
