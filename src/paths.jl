import Base: length
import Graphs: Edge
"""
    AbstractPath

An abstract type representing a path object with node type T.
"""
abstract type AbstractPath{T} end


"""
    Path

Concrete type representing a path with node type T
"""
struct Path{T} <: AbstractPath{T}
    nodes::Vector{T}
end

"""
    length(p::Path)
Returns the length of the path, i.e. number of nodes.
"""
length(p::Path) = Base.length(p.nodes)

"""
    nv(p::Path)
Number of nodes in path p.
"""
nv(p::Path) = length(p)

"""
    ne(p::Path)
Number of edges in path p.
"""
ne(p::Path) = nv(p)<=1 ? 0 : nv(p)-1


"""
    edges(p::Path)

Returns list of edges.
"""
function edges(p::Path{T}) where {T<:Integer}
    if ne(p)==0
        return []
    end
    edge_lst = Edge{T}[]
    for i in 2:nv(p)
        push!(edge_lst, Edge(p.nodes[i-1], p.nodes[i]))
    end
    return edge_lst
end

"""
    weight(p::Path, w::AbstractMatrix{T})

Return the weight of path p
"""
function weight(p::Path, w::AbstractMatrix{T}) where {T<:Number}
    return sum([w[src(e), dst(e)] for e in edges(p)])
end

"""
    path_to_column(p::Path, g::AbstractGraph{T})

return column representation of the path.
"""
function path_to_column(p::Path{T}, g::AbstractGraph{T}) where {T<:Integer}
    col = zeros(Int, ne(g))
    am = arc_index_matrix(g)
    for e in edges(p)
        col[am[src(e), dst(e)]] = 1
    end
    return col
end
