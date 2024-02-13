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
    FeatureDiGraph

Concrete directed graph with a feature vector for each arc.
"""
struct FeatureDiGraph{T,N} <: AbstractGraph{T}
    srcnodes::Vector{T}
    dstnodes::Vector{T}
    arc_features::Vector{N}
end

function FeatureDiGraph(srcnodes::Vector{T}, dstnodes::Vector{T}, features::AbstractArray{N}) where {T<:Number, N<:Number}
    FeatureDiGraph(srcnodes, dstnodes, [features[i,:] for i in 1:size(features,1)])
end

"""
    Graphs.is_directed(g::FeatureDiGraph{T,N})

Always returns true.
"""
Graphs.is_directed(g::FeatureDiGraph) = true

"""
    Graphs.has_edge(g::FeatureDiGraph{T,N}, s::T, d::T)

Check if graph contains edge (s,d)
"""
Graphs.has_edge(g::FeatureDiGraph{T,N}, s::T, d::T) where {T<:Number, N<:Number} = any(g.dstnodes[g.srcnodes .== s] .== d)

"""
    nv(g::FeatureDiGraph)

Returns number of vertices of the graph.
"""
Graphs.nv(g::FeatureDiGraph) = length(Set(vcat(g.srcnodes, g.dstnodes)))

"""
    ne(g::FeatureDiGraph)

Return number of arcs of the graph.
"""
Graphs.ne(g::FeatureDiGraph) = size(g.srcnodes,1)

"""
    feature_dim(g::FeatureDiGraph)

Get arc feature dimension.
"""
feature_dim(g::FeatureDiGraph) = size(g.arc_features[1])

"""
    Graphs.add_edge!(g::FeatureDiGraph{T,N}, src, dst, feat}

Add arc to a FeatureDiGraph object going from vertex `src` to `dst` and with features `feat`.
"""
function Graphs.add_edge!(g::FeatureDiGraph{T,N}, src::T, dst::T, feat::N) where {T,N}
    # throw error if dimension of the feature is not 1
    if feature_dim(g) != (1,)
        throw(DimensionMismatch("Feature dimension must be $(feature_dim(g))"))
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
    Graphs.add_edge!(g::FeatureDiGraph{T,N}, src, dst, feat::AbstractArray{N}}

Add arc to a FeatureDiGraph object going from vertex `src` to `dst` and with features `feat`.
"""
function Graphs.add_edge!(g::FeatureDiGraph{T,N}, src::T, dst::T, feat::AbstractArray{N}) where {T,N}
    # throw error if dimension of the feature is not 1
    if feature_dim(g) != size(feat)
        throw(DimensionMismatch("Feature dimension must be $(feature_dim(g))"))
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

Return list of edges of the graph.
"""
function Graphs.edges(g::FeatureDiGraph)
    return [FeatureDiGraphEdge(g.srcnodes[i],g.dstnodes[i],g.arc_features[i]) for i in 1:ne(g)]
end

"""
    Graphs.outneighbors(g::FeatureDiGraph{T,N}, v::T)

Get outgoing neighbors of vertex v in the graph.
"""
function Graphs.outneighbors(g::FeatureDiGraph{T,N}, v::T) where {T<:Number, N}
    return g.dstnodes[g.srcnodes .== v]
end

"""
    Graphs.inneighbors(g::FeatureDiGraph{T,N}, v::T)

Get neighbors u of vertex v such that edge (u,v) belongs to the graph.
"""
function Graphs.inneighbors(g::FeatureDiGraph{T,N}, v::T) where {T<:Number, N}
    return g.srcnodes[g.dstnodes .== v]
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::Int64)

Get a (nv(g) x nv(g)) matrix with coefficients equal to arc feature values.
"""
function feature_matrix(g::FeatureDiGraph, feature_idx::Int64) where {T<:Number, N}
    return sparse(g.srcnodes, g.dstnodes, [f[feature_idx] for f in g.arc_features])
end

"""
    feature_matrix(g::FeatureDiGraph, feature_idx::AbstractArray{Int64})

Get a (nv(g) x nv(g) x size(feature_idx)) matrix with coefficients equal to arc feature values.
TODO : managing feature_idx with multiple dimensions.
"""
function feature_matrix(g::FeatureDiGraph, feature_idx::AbstractArray{Int64})
    ans = zeros(nv(g), nv(g), size(feature_idx)...)
    for i in 1:size(feature_idx,1)
        ans[:,:,i] = feature_matrix(g, feature_idx[i])
    end
    return ans
end
