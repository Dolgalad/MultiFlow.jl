"""
    FeatureDiGraphEdge

Concrete type representing FeatureDiGraph edges.
"""
struct FeatureDiGraphEdge{T,N} <: AbstractEdge{T}
    src::T
    dst::T
    features::N
end

"""
    Graphs.src(e::FeatureDiGraphEdge)

Get edge source.
"""
Graphs.src(e::FeatureDiGraphEdge) = e.src

"""
    Graphs.dst(e::FeatureDiGraphEdge)

Get edge destination.
"""
Graphs.dst(e::FeatureDiGraphEdge) = e.dst

"""
    FeatureDiGraph{T,N}

Concrete directed graph with a feature vector for each arc.

The default constructor expects `srcnodes, dstnodes` to be vectors of source and destination vertices for each edge and `arc_features` to by a vector of numbers. The following example initialises a directed graph with three vertices and edges and a single `Float64` feature on each edge.

# Examplex
```jldoctest featuregraph
julia> g1 = FeatureDiGraph([1,2,3], [2,3,1], [5., 5., 5.])
FeatureDiGraph{Int64, Float64}([1, 2, 3], [2, 3, 1], [5.0, 5.0, 5.0])
```
"""
struct FeatureDiGraph{T<:Number,N} <: AbstractGraph{T}
    srcnodes::Vector{T}
    dstnodes::Vector{T}
    arc_features::Vector{N}
end

"""
    FeatureDiGraph(srcnodes::Vector{T}, dstnodes::Vector{T}, arc_features::AbstractArray{N}) where {T<:Number, N<:Number}

Construct a `FeatureDiGraph` object where edge features are given by `arc_features[i,:]`.

For example we can build a graph with a two-dimensional feature on each edge : 
```jldoctest featuregraph
julia> g = FeatureDiGraph([1,2,3], [2,3,1], 5 * ones(3,2))
FeatureDiGraph{Int64, Vector{Float64}}([1, 2, 3], [2, 3, 1], [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
```
"""
function FeatureDiGraph(srcnodes::Vector{T}, dstnodes::Vector{T}, arc_features::AbstractArray{N}) where {T<:Number, N<:Number}
    FeatureDiGraph(srcnodes, dstnodes, [arc_features[i,:] for i in 1:size(arc_features,1)])
end

"""
    Graphs.is_directed(g::FeatureDiGraph{T,N})

Check if the graph is directed, always returns true. Needed for compatibility with the _Graphs.jl_ package.
"""
Graphs.is_directed(g::FeatureDiGraph) = true

"""
    Graphs.has_edge(g::FeatureDiGraph{T,N}, s::T, d::T)

Check if graph contains edge (s,d). 

# Examples
```jldoctest featuregraph
julia> using Graphs

julia> has_edge(g, 1, 2)
true

julia> has_edge(g, 2, 1)
false
```
"""
Graphs.has_edge(g::FeatureDiGraph{T,N}, s::T, d::T) where {T<:Number,N} = any(g.dstnodes[g.srcnodes .== s] .== d)

"""
    nv(g::FeatureDiGraph)

Returns number of vertices of the graph. Needed for compatibility with the _Graphs.jl_ package.

# Examples
```jldoctest featuregraph
julia> nv(g)
3
```
"""
Graphs.nv(g::FeatureDiGraph) = length(Set(vcat(g.srcnodes, g.dstnodes)))

"""
    ne(g::FeatureDiGraph)

Return number of arcs of the graph. Needed for compatibility with the _Graphs.jl_ package.

# Examples
```jldoctest featuregraph
julia> ne(g)
3
```
"""
Graphs.ne(g::FeatureDiGraph) = size(g.srcnodes,1)

"""
    feature_dim(g::FeatureDiGraph)

Get arc feature dimension. 

# Examplex
```jldoctest featuregraph
julia> feature_dim(g1) # scalar features
()

julia> feature_dim(g)  # two-dimension features
(2,)
```
"""
feature_dim(g::FeatureDiGraph) = size(g.arc_features[1])

"""
    Graphs.add_edge!(g::FeatureDiGraph{T,N}, src::T, dst::T, feat::N}

Add arc to a FeatureDiGraph object going from vertex `src` to `dst` and with features `feat`. Return `true` on success and `false` if graph already has an edge `(src, dst)`. 

# Examples
```jldoctest featuregraph
julia> add_edge!(g1, 1, 4, 2.)
true

julia> add_edge!(g1, 1, 2, 3.)
false

julia> nv(g1), ne(g1)
(4, 4)

julia> add_edge(g, 1, 4, ones(3)) # must have feature_dim(g) == size(feat)
ERROR: DimensionMismatch("Expected feature dimension (2,) got (3,)")
[...]
```
"""
function Graphs.add_edge!(g::FeatureDiGraph{T,N}, src::T, dst::T, feat::N) where {T<:Number,N}
    # throw error if dimension of the feature is not 1
    if feature_dim(g) != size(feat)
        throw(DimensionMismatch("Expected feature dimension $(feature_dim(g)) got $(size(feat))"))
    end
    if has_edge(g, src, dst)
        return false
    end
    push!(g.srcnodes, src)
    push!(g.dstnodes, dst)
    push!(g.arc_features, feat)
    return true
end

"""
    edges(g::FeatureDiGraph)

Return list of edges of the graph. Needed for compatibility with _Graphs.jl_ package.

# Examples
```jldoctest featuregraph
julia> edges(g1)
4-element Vector{MultiFlows.FeatureDiGraphEdge{Int64, Int64}}:
 MultiFlows.FeatureDiGraphEdge{Int64, Int64}(1, 2, 1)
 MultiFlows.FeatureDiGraphEdge{Int64, Int64}(2, 3, 1)
 MultiFlows.FeatureDiGraphEdge{Int64, Int64}(3, 1, 1)
 MultiFlows.FeatureDiGraphEdge{Int64, Int64}(1, 4, 2)

```
"""
function Graphs.edges(g::FeatureDiGraph)
    return [FeatureDiGraphEdge(g.srcnodes[i],g.dstnodes[i],g.arc_features[i]) for i in 1:ne(g)]
end

"""
    Graphs.outneighbors(g::FeatureDiGraph{T,N}, v::T)

Get outgoing neighbors of vertex v in the graph. Needed for compatibility with _Graphs.jl_ package.

# Examples
```jldoctest featuregraph
julia> outneighbors(g1, 1)
2-element Vector{Int64}:
 2
 4

```
"""
function Graphs.outneighbors(g::FeatureDiGraph{T,N}, v::T) where {T<:Number, N}
    return g.dstnodes[g.srcnodes .== v]
end

"""
    Graphs.inneighbors(g::FeatureDiGraph{T,N}, v::T)

Get neighbors u of vertex v such that edge (u,v) belongs to the graph. Needed for compatibility with _Graphs.jl_ package.

# Examples
```jldoctest featuregraph
julia> inneighbors(g1, 1)
1-element Vector{Int64}:
 3

```
"""
function Graphs.inneighbors(g::FeatureDiGraph{T,N}, v::T) where {T<:Number, N}
    return g.srcnodes[g.dstnodes .== v]
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::Int64)

Get a `nv(g) x nv(g)` matrix with coefficients equal to arc feature values. Values returned as a sparse matrix.

# Examples
```jldoctest featuregraph
julia> feature_matrix(g, 1)
4×4 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅   5.0   ⋅   1.0
  ⋅    ⋅   5.0   ⋅
 5.0   ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅

```
"""
function feature_matrix(g::FeatureDiGraph, feature_idx::Int64) where {T<:Number, N}
    return sparse(g.srcnodes, g.dstnodes, [f[feature_idx] for f in g.arc_features], nv(g), nv(g))
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::AbstractArray{Int64})

Get a `nv(g) x nv(g) x size(feature_idx)` matrix with coefficients equal to arc feature values corresponding to indexes in `feature_idx`.
TODO : managing feature_idx with multiple dimensions.

# Examples
```jldoctest featuregraph
julia> feature_matrix(g, [2, 1])
4×4×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  5.0  0.0  1.0
 0.0  0.0  5.0  0.0
 5.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

[:, :, 2] =
 0.0  5.0  0.0  1.0
 0.0  0.0  5.0  0.0
 5.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

```
"""
function feature_matrix(g::FeatureDiGraph, feature_idx::AbstractArray{Int64})
    ans = zeros(nv(g), nv(g), size(feature_idx)...)
    for i in 1:size(feature_idx,1)
        ans[:,:,i] = feature_matrix(g, feature_idx[i])
    end
    return ans
end
