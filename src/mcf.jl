"""
    MCF

Multi-Commodity Flow problem data container. The default expects `graph` to be a [`FeatureDiGraph`](@ref) object with 2-dimension edge features `[cost, capacity]`. `demands` must be a list of [`Demand`](@ref) objects.

# Examples
```jldoctest
julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,3], ones(4,2))
FeatureDiGraph{Int64, Vector{Float64}}([1, 2, 3, 1], [2, 3, 1, 3], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

julia> MCF(g, [Demand(1, 2, 1.0)])
MCF(nv = 3, ne = 4, nk = 1)
	Demand{Int64, Float64}(1, 2, 1.0)
```
"""
struct MCF{T<:Number, N<:Number}
    graph::FeatureDiGraph{T, Vector{N}}
    demands::Vector{Demand{T, N}}
end

"""
    MCF(g::AbstractGraph{T}, cost::Vector{N}, capacity::Vector{N}, demands::Vector{Demand{T,N})

Create a `MCF` object from an `AbstractGraph` object, a cost and capacity vector with length `ne(g)` and a set of demands.

# Examples
```jldoctest mcf
julia> using Graphs

julia> gr = grid((3,2));

julia> MCF(gr, ones(ne(gr)), ones(ne(gr)), [Demand(1,6,1.0)])
MCF(nv = 6, ne = 7, nk = 1)
	Demand{Int64, Float64}(1, 6, 1.0)
```

`cost, capacity` vectors must have same length as `ne(gr)` : 
"""
function MCF(g::AbstractGraph{T}, cost::Vector{N}, capacity::Vector{N}, demands::Vector{Demand{T,N}}) where {T<:Number, N<:Number}
    if (ne(g) != length(cost)) || (ne(g) != length(capacity))
        throw(DimensionMismatch("Expect ne(g) == length(cost) == length(capacity), got $((ne(g), length(cost), length(capacity)))"))
    end
    fg = FeatureDiGraph(g, hcat(cost, capacity))
    MCF(fg, demands)
end

"""
    nv(pb::MCF)

Number of vertices.

# Examples
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,6,1.0)]))
julia> nv(pb)
6
```
"""
Graphs.nv(pb::MCF) = nv(pb.graph)

"""
    ne(pb::MCF)

Number of edges in the MCF network.

# Examples
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,6,1.0)]))
julia> ne(pb)
7
```
"""
Graphs.ne(pb::MCF) = ne(pb.graph)

"""
    nk(pb::MCF)

Number of demands in the MCF instance.

# Examples
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,6,1.0)]))
julia> nk(pb)
1
```
"""
nk(pb::MCF) = size(pb.demands, 1)

"""
    Base.show(io::IO, pb::MCF)

Show MCF object.
"""
function Base.show(io::IO, pb::MCF)
    println(io, "MCF(nv = $(nv(pb)), ne = $(ne(pb)), nk = $(nk(pb)))")
    for k in 1:nk(pb)
        println(io, "\t$(pb.demands[k])")
    end
end


"""
    weight_matrix(pb::MCF, idx::Int64=1)

Returns a `(nv(pb), nv(pb))` matrix with elements equal to edge features corresponding to `idx`.

# Examples
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(7), 2*ones(7), [Demand(1,6,1.0)]);

julia> weight_matrix(pb)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   1.0   ⋅   1.0   ⋅    ⋅
  ⋅    ⋅   1.0   ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅

```
"""
weight_matrix(pb::MCF, idx::Int64=1) = weights(pb.graph, idx)

"""
    cost_matrix(pb::MCF)

Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc costs.

# Examples
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), 2*ones(7), [Demand(1,6,1.0)]))
julia> cost_matrix(pb)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   1.0   ⋅   1.0   ⋅    ⋅
  ⋅    ⋅   1.0   ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
```

"""
cost_matrix(pb::MCF) = weight_matrix(pb, 1)

"""
    capacity_matrix(pb::MCF)

Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc capacity.

# Examples
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), 2*ones(7), [Demand(1,6,1.0)]))
julia> capacity_matrix(pb)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   2.0   ⋅   2.0   ⋅    ⋅
  ⋅    ⋅   2.0   ⋅   2.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   2.0
  ⋅    ⋅    ⋅    ⋅   2.0   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   2.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
```

"""
capacity_matrix(pb::MCF) = weight_matrix(pb, 2)

"""
    scale_demands(pb::MCF, factor)

Scale the amount of the demands by `factor`. Returns new list of demands.

# Example
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), 2*ones(7), [Demand(1,6,1.0)]))
julia> dems = scale_demands(pb, 2)
1-element Vector{Demand{Int64, Float64}}:
 Demand{Int64, Float64}(1, 6, 2.0)

```
"""
function scale_demands(pb::MCF, factor)
    return [Demand(d.src, d.dst, factor*d.amount) for d in pb.demands]
end

"""
    scale(pb::MCF, cost_factor=1.0, capacity_factor=1.0)

Return a new MCF instance with costs scaled by a `cost_factor`, capacity and demand amounts scaled by `capacity_factor`.

# Example
```jldoctest; setup = :(using Graphs; pb = MCF(grid((3,2)), ones(7), 2*ones(7), [Demand(1,6,1.0)]))
julia> pb1 = scale(pb, 1.5, 3)
MCF(nv = 6, ne = 7, nk = 1)
	Demand{Int64, Float64}(1, 6, 3.0)

julia> cost_matrix(pb1)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   1.5   ⋅   1.5   ⋅    ⋅
  ⋅    ⋅   1.5   ⋅   1.5   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.5
  ⋅    ⋅    ⋅    ⋅   1.5   ⋅
  ⋅    ⋅    ⋅    ⋅    ⋅   1.5
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅

julia> capacity_matrix(pb1)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   6.0   ⋅   6.0   ⋅    ⋅ 
  ⋅    ⋅   6.0   ⋅   6.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   6.0
  ⋅    ⋅    ⋅    ⋅   6.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   6.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
```
"""
function scale(pb::MCF{T,N}, cost_factor=1.0, capacity_factor=1.0) where {T,N}
    return MCF(
               scale_features(pb.graph, N.([cost_factor, capacity_factor])),
               scale_demands(pb, capacity_factor)
              )
end

"""
    normalize(pb::MCF)

Normalize MCF instance. Costs are scaled by `1 / max(pb.cost)`, capacity and demand amount are scaled by `1 / max(max(pb.capacity), max(pb.amount))`.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), collect(1.0:7.0), collect(0.0:2:13.0), [Demand(1,6,10.0)])
MCF(nv = 6, ne = 7, nk = 1)
	Demand{Int64, Float64}(1, 6, 10.0)

julia> pbn = normalize(pb)
MCF(nv = 6, ne = 7, nk = 1)
	Demand{Int64, Float64}(1, 6, 0.8333333333333333)

julia> cost_matrix(pbn)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   0.142857   ⋅        0.285714   ⋅         ⋅
  ⋅    ⋅        0.428571   ⋅        0.571429   ⋅
  ⋅    ⋅         ⋅         ⋅         ⋅        0.714286
  ⋅    ⋅         ⋅         ⋅        0.857143   ⋅
  ⋅    ⋅         ⋅         ⋅         ⋅        1.0
  ⋅    ⋅         ⋅         ⋅         ⋅         ⋅

julia> capacity_matrix(pbn)
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
  ⋅   0.0   ⋅        0.166667   ⋅         ⋅
  ⋅    ⋅   0.333333   ⋅        0.5        ⋅
  ⋅    ⋅    ⋅         ⋅         ⋅        0.666667
  ⋅    ⋅    ⋅         ⋅        0.833333   ⋅
  ⋅    ⋅    ⋅         ⋅         ⋅        1.0
  ⋅    ⋅    ⋅         ⋅         ⋅         ⋅
```
"""
function normalize(pb::MCF)
    return scale(pb, 1.0/maximum(cost_matrix(pb)), 1.0/max(maximum(capacity_matrix(pb)), maximum([d.amount for d in pb.demands])))
end

"""
    has_demand(pb::MCF{T,N}, s::T, d::T)

Check if problem has a demand originating at vertex `s` and with destination `d`.
"""
function has_demand(pb::MCF{T,N}, s::T, d::T) where {T,N}
    for d in pb.demands
        if d.src == s && d.dst == d
            return true
        end
    end
    return false
end
