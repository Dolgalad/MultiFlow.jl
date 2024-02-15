"""
    AbstractPath

Abstract type for representing paths in a graph.
"""
abstract type AbstractPath end

"""
    VertexPath

Concrete type for paths composed of a sequence of vertex indices.
"""
struct VertexPath{T} <: AbstractPath
    vertices::Vector{T}
end

"""
    Base.isempty(p::VertexPath)

Check if the path is empty.
"""
Base.isempty(p::VertexPath) = isempty(p.vertices)

"""
"""
Graphs.edges(p::VertexPath) = [(p.vertices[i], p.vertices[i+1]) for i in 1:length(p.vertices)-1]

"""
    path_weight(p::VertexPath, g::AbstractGraph, aggr::Function=sum)

Compute the weight of the path in graph `g`. 
"""
function path_weight(p::VertexPath, g::AbstractGraph; aggr::Function=sum, dstmx=weights(g))
    aggr(dstmx[u,v] for (u,v) in edges(p))
end

"""
    edge_indices(p::VertexPath, g::AbstractGraph)

Get list of edge indices corresponding to the path.
"""
function edge_indices(p::VertexPath, g::AbstractGraph)
    
end

#"""
#    EdgeIndexPath
#
#Concrete type for paths composed of a sequence of edge indices.
#"""
#struct EdgeIndexPath <: AbstractPath
#    edge_idx::Vector{Int64}
#end
#
#"""
#    Base.isempty(p::VertexPath)
#
#Check if the path is empty.
#"""
#Base.isempty(p::EdgeIndexPath) = isempty(p.edge_idx)
#
#"""
#"""
#Graphs.edges(p::VertexPath) = [(p.vertices[i], p.vertices[i+1]) for i in 1:length(p.vertices)-1]
#
#"""
#    path_weight(p::VertexPath, g::AbstractGraph, aggr::Function=sum)
#
#Compute the weight of the path in graph `g`. 
#"""
#function path_weight(p::VertexPath, g::AbstractGraph; aggr::Function=sum, dstmx=weights(g))
#    aggr(dstmx[u,v] for (u,v) in edges(p))
#end
