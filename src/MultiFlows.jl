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
    #Demand,
    FeatureDiGraphEdge,
    FeatureDiGraph,
    feature_dim,
    feature_matrix,
    #MCF,
    nk,
    nv,
    ne
    #graph,
    #weight_matrix,
    #cost_matrix,
    #capacity_matrix,
    #scale,
    #normalize,
    ## loading MCFs
    #load,
    ## plotting
    #plot,
    ## paths
    #AbstractPath,
    #Path,
    #weight,
    #edges,
    ## graph utilities
    #arc_index_matrix

#MultiFlows
#include("demand.jl")
include("feature_graph.jl")
#include("mcf.jl")
#include("paths.jl")
#include("graph_utils.jl")

end # module MultiFlows
