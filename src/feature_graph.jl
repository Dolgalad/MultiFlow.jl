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

Concrete directed graph with a feature vector for each edge.

The default constructor expects `srcnodes, dstnodes` to be vectors of source and destination vertices for each edge and `features` to by a vector of numbers. The following example initialises a directed graph with three vertices and edges and a single `Float64` feature on each edge.

# Examplex
```jldoctest
julia> using MultiFlows

julia> g1 = FeatureDiGraph([1,2,3], [2,3,1], [5., 5., 5.])
FeatureDiGraph{Int64, Float64}([1, 2, 3], [2, 3, 1], [5.0, 5.0, 5.0])

```
"""
struct FeatureDiGraph{T<:Number,N} <: AbstractGraph{T}
    srcnodes::Vector{T}
    dstnodes::Vector{T}
    features::Vector{N}
end

"""
    FeatureDiGraph(srcnodes::Vector{T}, dstnodes::Vector{T}, features::AbstractArray{N}) where {T<:Number, N<:Number}

Construct a `FeatureDiGraph` object where edge features are given by `features[i,:]`.

# Examplex
For example we can build a graph with a two-dimensional feature on each edge : 

```jldoctest
julia> g = FeatureDiGraph([1,2,3], [2,3,1], 5 * ones(3,2))
FeatureDiGraph{Int64, Vector{Float64}}([1, 2, 3], [2, 3, 1], [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
```

"""
function FeatureDiGraph(srcnodes::Vector{T}, dstnodes::Vector{T}, features::AbstractArray{N}) where {T<:Number, N<:Number}
    FeatureDiGraph(srcnodes, dstnodes, [features[i,:] for i in 1:size(features,1)])
end


"""
    FeatureDiGraph(g::AbstractGraph{T}, features::AbstractArray{N})

Construct a feature graph from an `AbstractGraph` object and a set of features.

# Examples
```jldoctest
julia> using Graphs

julia> gr = grid((3,2));

julia> FeatureDiGraph(gr, ones(ne(gr), 2))
FeatureDiGraph{Int64, Vector{Float64}}([1, 1, 2, 2, 3, 4, 5], [2, 4, 3, 5, 6, 5, 6], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

```
"""
function FeatureDiGraph(g::AbstractGraph{T}, features::AbstractArray{N}) where {T<:Number, N<:Number}
    edge_list = edges(g)
    FeatureDiGraph(src.(edge_list), dst.(edge_list), features)
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
```jldoctest; setup = :(g = FeatureDiGraph([1,2,3],[2,3,1],5*ones(3,2)))
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
```jldoctest;setup = :(g = FeatureDiGraph([1,2,3],[2,3,1],5*ones(3,2)))
julia> nv(g)
3
```
"""
Graphs.nv(g::FeatureDiGraph) = length(Set(vcat(g.srcnodes, g.dstnodes)))

"""
    ne(g::FeatureDiGraph)

Return number of edge of the graph. Needed for compatibility with the _Graphs.jl_ package.

# Examples
```jldoctest;setup = :(g = FeatureDiGraph([1,2,3],[2,3,1],5*ones(3,2)))
julia> ne(g)
3
```
"""
Graphs.ne(g::FeatureDiGraph) = size(g.srcnodes,1)

"""
    feature_dim(g::FeatureDiGraph)

Get edge feature dimension. 

# Examplex
```jldoctest featuregraph
julia> g1 = FeatureDiGraph([1,2,3], [2,3,1], ones(3));

julia> feature_dim(g1) # scalar features
()

julia> g2 = FeatureDiGraph([1,2,3], [2,3,1], ones(3, 2));

julia> feature_dim(g2)  # two-dimension features
(2,)
```
"""
feature_dim(g::FeatureDiGraph) = size(g.features[1])

"""
    Graphs.add_edge!(g::FeatureDiGraph{T,N}, src::T, dst::T, feat::N}

Add edge to a FeatureDiGraph object going from vertex `src` to `dst` and with features `feat`. Return `true` on success and `false` if graph already has an edge `(src, dst)`. 

# Examples
```jldoctest; setup = :(using Graphs)
julia> g = FeatureDiGraph([1,2,3], [2,3,1], ones(3,2));

julia> add_edge!(g, 1, 4, [2., 2.])
true

julia> add_edge!(g, 1, 2, [3., 3.])
false

julia> nv(g), ne(g)
(4, 4)

julia> add_edge!(g, 1, 4, ones(3)) # must have feature_dim(g) == size(feat)
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
    push!(g.features, feat)
    return true
end

"""
    edges(g::FeatureDiGraph)

Return list of edges of the graph. Needed for compatibility with _Graphs.jl_ package.

# Examples
```jldoctest; setup = :(using Graphs)
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], [1,1,1,2]);

julia> edges(g)
4-element Vector{FeatureDiGraphEdge{Int64, Int64}}:
 FeatureDiGraphEdge{Int64, Int64}(1, 2, 1)
 FeatureDiGraphEdge{Int64, Int64}(2, 3, 1)
 FeatureDiGraphEdge{Int64, Int64}(3, 1, 1)
 FeatureDiGraphEdge{Int64, Int64}(1, 4, 2)

```
"""
function Graphs.edges(g::FeatureDiGraph)
    return [FeatureDiGraphEdge(g.srcnodes[i],g.dstnodes[i],g.features[i]) for i in 1:ne(g)]
end

"""
    Graphs.outneighbors(g::FeatureDiGraph{T,N}, v::T)

Get outgoing neighbors of vertex v in the graph. Needed for compatibility with _Graphs.jl_ package.

# Examples
```jldoctest; setup = :(using Graphs; g = FeatureDiGraph([1,2,3,1],[2,3,1,4],[1,1,1,2]))
julia> outneighbors(g, 1)
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
```jldoctest; setup = :(using Graphs; g = FeatureDiGraph([1,2,3,1],[2,3,1,4],[1,1,1,2]))
julia> inneighbors(g, 1)
1-element Vector{Int64}:
 3

```
"""
function Graphs.inneighbors(g::FeatureDiGraph{T,N}, v::T) where {T<:Number, N}
    return g.srcnodes[g.dstnodes .== v]
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::Int64)

Get a `nv(g) x nv(g)` matrix with coefficients equal to edge feature values. Values returned as a sparse matrix.

# Examples

Graph with scalar features : 
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], [5.0, 5.0, 5.0, 1.0]);

julia> feature_matrix(g)
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅   5.0   ⋅   1.0
  ⋅    ⋅   5.0   ⋅
 5.0   ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅

```

Graph with multi-dimensional features : 
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], hcat(3*ones(4), 4*ones(4)));

julia> feature_matrix(g, 2)
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅   4.0   ⋅   4.0
  ⋅    ⋅   4.0   ⋅
 4.0   ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅

```

"""
function feature_matrix(g::FeatureDiGraph, feature_idx::Int64=1) where {T<:Number, N}
    if isempty(feature_dim(g)) # scalar features
        return sparse(g.srcnodes, g.dstnodes, [f for f in g.features], nv(g), nv(g))
    end
    return sparse(g.srcnodes, g.dstnodes, [f[feature_idx] for f in g.features], nv(g), nv(g))
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::AbstractArray{Int64})

Get a `nv(g) x nv(g) x size(feature_idx)` matrix with coefficients equal to edge feature values corresponding to indexes in `feature_idx`.
TODO : managing feature_idx with multiple dimensions.

# Examples
```jldoctest; setup = :(g = FeatureDiGraph([1,2,3,1], [2,3,1,4], hcat(3*ones(4), 4*ones(4))))
julia> feature_matrix(g, [2, 1])
4×4×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  4.0  0.0  4.0
 0.0  0.0  4.0  0.0
 4.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

[:, :, 2] =
 0.0  3.0  0.0  3.0
 0.0  0.0  3.0  0.0
 3.0  0.0  0.0  0.0
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

"""
    Graphs.weights(g::FeatureDiGraph, feature_idx::Int64=1)

Get graph weight matrix corresponding to in feature with index `idx`. Needed for compatibility with _Graphs.jl_.

# Examples
```jldoctest
julia> using Graphs

julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,3],[[1,1],[1,1],[1,1],[4,1]])
FeatureDiGraph{Int64, Vector{Int64}}([1, 2, 3, 1], [2, 3, 1, 3], [[1, 1], [1, 1], [1, 1], [4, 1]])

julia> ds = dijkstra_shortest_paths(g, 1);

julia> enumerate_paths(ds, 3)
3-element Vector{Int64}:
 1
 2
 3

julia> ds = dijkstra_shortest_paths(g, 1, weights(g, 2));

julia> enumerate_paths(ds, 3)
2-element Vector{Int64}:
 1
 3
```

"""
Graphs.weights(g::FeatureDiGraph, idx::Int64=1) = feature_matrix(g, idx)

"""
    scale_features(g::FeatureDiGraph{T,N}, factor::N)

Scale the features of the graph by `factor`. `factor` should have the same dimension as the edge features.
"""
function scale_features(g::FeatureDiGraph{T,N}, factor::N) where {T<:Number,N}
    return FeatureDiGraph(g.srcnodes, g.dstnodes, [factor .* f for f in g.features])
end

"""
    double_edges!(g::FeatureDiGraph)

For each edge `(u,v)` add edge `(v,u)` if not already present in the graph. Features of the edge are copied.

# Examples
```jldoctest
julia> using Graphs

julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,3],[[1,1],[1,1],[1,1],[4,1]])
FeatureDiGraph{Int64, Vector{Int64}}([1, 2, 3, 1], [2, 3, 1, 3], [[1, 1], [1, 1], [1, 1], [4, 1]])

julia> ne(g)
4

julia> double_edges!(g)
FeatureDiGraph{Int64, Vector{Int64}}([1, 2, 3, 1, 2, 3], [2, 3, 1, 3, 1, 2], [[1, 1], [1, 1], [1, 1], [4, 1], [1, 1], [1, 1]])

julia> ne(g)
6
```

"""
function double_edges!(g::FeatureDiGraph)
    orig_edges = edges(g)
    for e in orig_edges
        if !has_edge(g, dst(e), src(e))
            add_edge!(g, dst(e), src(e), e.features)
        end
    end
    return g
end

"""
    edge_features(g::FeatureDiGraph)

Get array of edge features. Returns a `(ne(g), feature_dim(g))` array.

# Examples
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], hcat(3*ones(4), 4*ones(4)));

julia> edge_features(g)
4×2 Matrix{Float64}:
 3.0  4.0
 3.0  4.0
 3.0  4.0
 3.0  4.0
```
"""
function edge_features(g::FeatureDiGraph)
    return collect(transpose(hcat([e.features for e in edges(g)]...)))
end

"""
    edge_features(g::FeatureDiGraph, idx::Int64)

Get array of edge features corresponding to index `idx`. Returns a `(ne(g), 1)` array.

# Examples
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], hcat(3*ones(4), 4*ones(4)));

julia> edge_features(g, 1)
4-element Vector{Float64}:
 3.0
 3.0
 3.0
 3.0
```
"""
function edge_features(g::FeatureDiGraph, idx::Int64)
    return vcat([e.features[idx] for e in edges(g)]...)
end

"""
    edge_index_matrix(g::AbstractGraph)

Get `(nv(g), nv(g))` sized matrix where `M[i,j]` is the index of edge `i,j`.

# Examples
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,4], hcat(3*ones(4), 4*ones(4)));

julia> edge_index_matrix(g)
3×4 SparseArrays.SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 ⋅  1  ⋅  4
 ⋅  ⋅  2  ⋅
 3  ⋅  ⋅  ⋅
```

Simple undirected graphs : 
```jldoctest
julia> using Graphs

julia> g = grid((3,2))
{6, 7} undirected simple Int64 graph

julia> edge_index_matrix(g)
6×6 SparseArrays.SparseMatrixCSC{Int64, Int64} with 14 stored entries:
 ⋅  1  ⋅  2  ⋅  ⋅
 1  ⋅  3  ⋅  4  ⋅
 ⋅  3  ⋅  ⋅  ⋅  5
 2  ⋅  ⋅  ⋅  6  ⋅
 ⋅  4  ⋅  6  ⋅  7
 ⋅  ⋅  5  ⋅  7  ⋅
```

"""
function edge_index_matrix(g::AbstractGraph{T}) where {T}
    if is_directed(g)
        return sparse([src(e) for e in edges(g)],
                      [dst(e) for e in edges(g)],
                      1:ne(g))
    else
        s,t,idx = T[], T[], T[]
        for (i,e) in enumerate(edges(g))
            push!(s, src(e)); push!(s, dst(e))
            push!(t, dst(e)); push!(t, src(e))
            push!(idx, i); push!(idx, i)
        end
        return sparse(s,t,idx)
    end
end
