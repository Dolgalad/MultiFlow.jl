"""
    AbstractPath

Abstract type for representing paths in a graph.
"""
abstract type AbstractPath end

"""
    VertexPath

Concrete type for paths composed of a sequence of vertex indices.

# Example
```jldoctest; setup = :(using Graphs)
julia> g = grid((3,2))
{6, 7} undirected simple Int64 graph

julia> ds = dijkstra_shortest_paths(g, 1);

julia> p = VertexPath(enumerate_paths(ds, 6))
VertexPath{Int64}([1, 2, 5, 6])
```
"""
struct VertexPath{T} <: AbstractPath
    vertices::Vector{T}
end

"""
    Base.isempty(p::VertexPath)

Check if the path is empty.
"""
Base.isempty(p::VertexPath) = Base.isempty(p.vertices)

"""
    Graphs.edges(p::VertexPath)

Return edge iterator of the path.

# Example
```jldoctest; setup = :(using Graphs)
julia> p = VertexPath([1,2,3])
VertexPath{Int64}([1, 2, 3])

julia> edges(p)
2-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (2, 3)
```
"""
Graphs.edges(p::VertexPath) = [(p.vertices[i], p.vertices[i+1]) for i in 1:length(p.vertices)-1]

"""
    path_weight(p::VertexPath, g::AbstractGraph; aggr::Function=sum, dstmx=weights(g))

Compute the weight of the path in graph `g`. 

# Example
```jldoctest path_weight; setup = :(using Graphs; g = grid((3,2)); p = VertexPath(enumerate_paths(dijkstra_shortest_paths(g, 1), 6)))
julia> path_weight(p, g)
3
```

You may provide a different weight matrix 
```jldoctest path_weight
julia> w = zeros(nv(g), nv(g));

julia> w[Bool.(adjacency_matrix(g))] .= 1:2*ne(g);

julia> w
6×6 Matrix{Float64}:
 0.0  3.0  0.0  8.0   0.0   0.0
 1.0  0.0  6.0  0.0  10.0   0.0
 0.0  4.0  0.0  0.0   0.0  13.0
 2.0  0.0  0.0  0.0  11.0   0.0
 0.0  5.0  0.0  9.0   0.0  14.0
 0.0  0.0  7.0  0.0  12.0   0.0

julia> path_weight(p, g, dstmx=w)
27.0

julia> path_weight(p, g, dstmx=w, aggr=minimum)
3.0
```
"""
function path_weight(p::VertexPath, g::AbstractGraph; aggr::Function=sum, dstmx=weights(g))
    aggr(dstmx[u,v] for (u,v) in edges(p))
end

"""
    edge_indices(p::VertexPath, g::AbstractGraph)

Get list of edge indices corresponding to the path.

# Example
```jldoctest; setup = :(using Graphs; g = grid((3,2)))
julia> p = VertexPath([1,2,5,6]);

julia> edge_indices(p, g)
3-element Vector{Int64}:
 1
 4
 7

julia> collect(edges(g))[edge_indices(p,g)]
3-element Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}}:
 Edge 1 => 2
 Edge 2 => 5
 Edge 5 => 6

```
"""
function edge_indices(p::VertexPath, g::AbstractGraph)
    eim = edge_index_matrix(g)
    return [eim[u,v] for (u,v) in edges(p)]
end

"""
    is_path(p::VertexPath, g::AbstractGraph)

Check if `p` is a valid path in graph `g`.

# Example
```jldoctest; setup = :(using Graphs)
julia> g = grid((3,2));

julia> p = VertexPath([1,2,5,6]);

julia> is_path(p, g)
true

julia> is_path(p, grid((2,1)))
false
```
"""
is_path(p::VertexPath, g::AbstractGraph) = all(has_edge(g, u, v) for (u,v) in edges(p))

"""
    is_path(p::VertexPath{T}, s::T, t::T)

Check if path is an ``s-t``-path.

# Example
```jldoctest
julia> p = VertexPath([1,2,3]);

julia> is_path(p, 1, 2)
false

julia> is_path(p, 1, 3)
true
```

"""
is_path(p::VertexPath{T}, s::T, t::T) where {T} = p.vertices[1]==s && p.vertices[end]==t

"""
    Graphs.has_edge(p::VertexPath{T}, s::T, t::T) where {T}

Check if the path contains edge `(s,t)`.

# Example
```jldoctest; setup = :(using Graphs)
julia> p = VertexPath([1,2,3]);

julia> has_edge(p, 1, 2)
true

julia> has_edge(p, 1, 3)
false
```
"""
Graphs.has_edge(p::VertexPath{T}, s::T, t::T) where {T} = (s,t) in edges(p)


"""
    path_from_edge_indices(ei::Vector{Int64}, g::AbstractGraph)

Convert a list of edge indices to a path in graph `g`.

# Example
```jldoctest; setup = :(using Graphs)
julia> g = grid((3,3))
{9, 12} undirected simple Int64 graph

julia> p = VertexPath(enumerate_paths(dijkstra_shortest_paths(g, 1), 9))
VertexPath{Int64}([1, 2, 5, 6, 9])

julia> ei = edge_indices(p, g)
4-element Vector{Int64}:
  1
  4
  8
 10

julia> path_from_edge_indices(ei, g)
VertexPath{Int64}([1, 2, 5, 6, 9])

```
"""
function path_from_edge_indices(ei::Vector{Int64}, g::AbstractGraph)
    elst = collect(edges(g))
    p = [src(elst[ei[1]])]
    for eidx in ei
        push!(p, dst(elst[eidx]))
    end
    return VertexPath(p)
end

"""
    Base.length(p::VertexPath; edges=false)

Returns length of the path. Default is to return number of vertices, if `edges=true` returns number of edges.

# Example
```jldoctest
julia> p = VertexPath([1,2,3])
VertexPath{Int64}([1, 2, 3])

julia> length(p)
3

julia> length(p, edges=true)
2
```
"""
function Base.length(p::VertexPath; edges=false)
    if edges
        return length(p.vertices)-1
    else
        return length(p.vertices)
    end
end

"""
    Base.:(==)(p1::VertexPath{T}, p2::VertexPath{T}) where {T} 

Check path equality. Checks `p1.vertices == p2.vertices`.
"""
function Base.:(==)(p1::VertexPath{T}, p2::VertexPath{T}) where {T} 
    return p1.vertices==p2.vertices
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
