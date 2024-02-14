#"""
#    MCF
#
#Multi-Commodity Flow problem data container. The default expects `graph` to be a [`FeatureDiGraph`](@ref) object with 2-dimension edge features `[cost, capacity]`. `demands` must be a list of [`Demand`](@ref) objects.
#
## Examples
#```jldoctest mcf
#julia> using MultiFlows
#
#julia> g = FeatureDiGraph([1,2,3,1], [2,3,1,3], ones(4,2))
#FeatureDiGraph{Int64, Vector{Float64}}([1, 2, 3, 1], [2, 3, 1, 3], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
#
#julia> pb = MCF(g, [Demand(1, 2, 1.0)])
#MCF(nv = 3, ne = 4, nk = 1)
#	Demand{Int64, Float64}(1, 2, 1.0)
#
#```
#"""
#struct MCF{T<:Number, N<:Number}
#    graph::FeatureDiGraph{T, Vector{N}}
#    demands::Vector{Demand{T, N}}
#end
#
#"""
#    MCF(g::AbstractGraph{T}, cost::Vector{N}, capacity::Vector{N}, demands::Vector{Demand{T,N})
#
#Create a `MCF` object from an `AbstractGraph` object, a cost and capacity vector with length `ne(g)` and a set of demands.
#
## Examples
#```jldoctest mcf
#julia> using Graphs
#
#julia> gr = grid((3,2));
#
#julia> pb = MCF(gr, ones(ne(gr)), ones(ne(gr)), [Demand(1,6,1.0)])
#MCF(nv = 6, ne = 7, nk = 1)
#	Demand{Int64, Float64}(1, 6, 1.0)
#
#```
#"""
#function MCF(g::AbstractGraph{T}, cost::Vector{N}, capacity::Vector{N}, demands::Vector{Demand{T,N}}) where {T<:Number, N<:Number}
#    if (ne(g) != length(cost)) || (ne(g) != length(capacity))
#        throw(DimensionMismatch("Expect ne(g) == length(cost) == length(capacity), got $((length(cost), length(capacity)))"))
#    end
#    fg = FeatureDiGraph(g, hcat(cost, capacity))
#    MCF(fg, demands)
#end
#
#"""
#    nv(pb::MCF)
#
#Number of vertices.
#
## Examples
#```jldoctest mcf
#julia> nv(pb)
#6
#```
#"""
#Graphs.nv(pb::MCF) = nv(pb.graph)
#
#"""
#    ne(pb::MCF)
#
#Number of edges in the MCF network.
#
## Examples
#```jldoctest mcf
#julia> ne(pb)
#7
#```
#"""
#Graphs.ne(pb::MCF) = ne(pb.graph)
#
#"""
#    nk(pb::MCF)
#
#Number of demands in the MCF instance.
#
## Examples
#```jldoctest mcf
#julia> nk(pb)
#1
#```
#"""
#nk(pb::MCF) = size(pb.demands, 1)
#
#"""
#    Base.show(io::IO, pb::MCF)
#
#Show MCF object.
#"""
#function Base.show(io::IO, pb::MCF)
#    println("MCF(nv = $(nv(pb)), ne = $(ne(pb)), nk = $(nk(pb)))")
#    for k in 1:nk(pb)
#        println("\t$(pb.demands[k])")
#    end
#end
#
#"""
#    UnknownMultiFlowFormat Exception
#"""
#struct UnknownMultiFlowFormat <: Exception end
#
#"""
#    load(dirname::String; format=:csv, edge_dir=:single)
#
#Load MultiFlow problem from file. If format=:csv uses load_csv(dirname::String). `edge_dir` can be either `:single, :double, :undirected`.
#"""
#function load(dirname::String; format=:csv, directed=:single)
#    if format==:csv
#        return load_csv(dirname)
#    else
#        throw(UnknownMultiFlowFormat("Unknown format "*format))
#    end
#end
#
#"""
#    load_csv(dirname::String)
#
#Load MultiFlow instance from CSV files. Default is to search for a link.csv and service.csv file.
#"""
#function load_csv(dirname::String; directed=false)
#    linkpath = joinpath(dirname, "link.csv")
#    servicepath = joinpath(dirname, "service.csv")
#    if !isfile(linkpath) || !isfile(servicepath)
#        throw(UnknownMultiFlowFormat("Could not find $linkfile and $servicefile"))
#    end
#    dflinks = CSV.read(linkpath, DataFrame)
#    rename!(dflinks, strip.(lowercase.(names(dflinks))))
#    dfservices = CSV.read(servicepath, DataFrame)
#    rename!(dfservices, strip.(lowercase.(names(dfservices))))
#
#    srcnodes = dflinks.srcnodeid
#    dstnodes = dflinks.dstnodeid
#    nodes = Set(vcat(srcnodes, dstnodes))
#    nnodes = length(nodes)
#    # keep a dictionary of original node ids to new ones
#    vmap = Dict(nodes .=> 1:nnodes)
#
#    capacities = dflinks.bandwidth
#    costs = dflinks.cost
#    narcs = size(dflinks, 1) # number of arcs
#
#    # list of demands
#    #store arrays for demands
#    srcdemands = dfservices.srcnodeid
#    dstdemands = dfservices.dstnodeid
#    bandwidths = dfservices.bandwidth
#    ndemands = size(dfservices, 1)# number of demands
#
#    return MCF(srcnodes, dstnodes, costs, capacities, srcdemands, dstdemands, bandwidths)
#
#end
#
#"""
#    weight_matrix(pb::MCF, idx::Int64=1)
#
#Returns a `(nv(pb), nv(pb))` matrix with elements equal to edge features corresponding to `idx`.
#
## Examples
#```jldoctest mcf
#julia> weight_matrix(pb)
#6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 7 stored entries:
#  ⋅   1.0   ⋅   1.0   ⋅    ⋅
#  ⋅    ⋅   1.0   ⋅   1.0   ⋅
#  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
#  ⋅    ⋅    ⋅    ⋅   1.0   ⋅
#  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
#  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅
#
#```
#"""
#weight_matrix(pb::MCF, idx::Int64=1) = weights(pb.graph, idx)
#
#"""
#    cost_matrix(pb::MCF)
#
#Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc costs.
#"""
#cost_matrix(pb::MCF) = weight_matrix(pb, 1)
#
#"""
#    capacity_matrix(pb::MCF)
#
#Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc capacity.
#"""
#capacity_matrix(pb::MCF) = weight_matrix(pb, 2)
#
#"""
#    scale_cost(pb::MCF, cost_factor::Float64=1.0, capacity_factor::Float64=1.0)
#
#Return a new MCF instance with costs scaled by a `cost_factor`, capacity and demand amounts scaled by `capacity_factor`.
#"""
#function scale(pb::MCF, cost_factor::Float64=1.0, capacity_factor::Float64=1.0)
#    return MCF(
#               pb.srcnodes,
#               pb.dstnodes,
#               pb.cost * cost_factor,
#               pb.capacity * capacity_factor,
#               pb.srcdemands,
#               pb.dstdemands,
#               pb.amount * capacity_factor
#              )
#end
#
#"""
#    normalize(pb::MCF)
#
#Normalize MCF instance. Costs are scaled by `1 / max(pb.cost)`, capacity and demand amount are scaled by `1 / max(max(pb.capacity), max(pb.amount))`.
#"""
#function normalize(pb::MCF)
#    return scale(pb, 1.0/maximum(pb.cost), 1.0/max(maximum(pb.capacity), maximum(pb.amount)))
#end
