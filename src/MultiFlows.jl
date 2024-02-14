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
    arc_features,
    Demand,
    MCF,
    weight_matrix,
    cost_matrix,
    capacity_matrix,
    scale_demands,
    scale,
    normalize,
    load,
    load_csv,
    save,
    is_instance_dir,
    # plotting
    mcfplot

#MultiFlows
include("feature_graph.jl")
include("demand.jl")
include("mcf.jl")
include("mcf_io.jl")
include("mcf_plot.jl")
#include("paths.jl")
#include("graph_utils.jl")

end # module MultiFlows
