module MultiFlows

import Graphs: AbstractGraph, nv, ne, edges
using GraphPlot
using Compose
using Measures

import Cairo, Fontconfig

export 
    AbstractDemand,
    AbstractMultiFlow,
    Demand,
    MultiFlow,
    nk,
    # plotting
    plot,
    # paths
    AbstractPath,
    Path,
    weight,
    edges,
    # graph utilities
    arc_index_matrix

include("interface.jl")

nv(mf::AbstractMultiFlow) = nv(mf.graph)
ne(mf::AbstractMultiFlow) = ne(mf.graph)

include("plotting.jl")
include("paths.jl")
include("graph_utils.jl")
include("mcf.jl")



export func

"""
    func(x)

Return double the number `x` plus `1`.
"""
func(x) = 2x + 1

end # module MultiFlows
