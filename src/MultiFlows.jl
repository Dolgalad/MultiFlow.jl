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
    FeatureDiGraphEdge,
    FeatureDiGraph,
    feature_dim,
    feature_matrix,
    nk,
    nv,
    ne,
    scale_features,
    double_edges!,
    Demand,
    MCF,
    weight_matrix,
    cost_matrix,
    capacity_matrix,
    scale_demands,
    scale,
    normalize,
    load,
    load_csv

#MultiFlows
include("feature_graph.jl")
include("demand.jl")
include("mcf.jl")
include("mcf_file.jl")
#include("paths.jl")
#include("graph_utils.jl")

end # module MultiFlows
