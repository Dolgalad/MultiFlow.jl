"""
    MCF

Multi-Commodity Flow problem data container
"""
struct MCF
    graph::FeatureDiGraph{Int64, Float64}
    demands::Vector{Demand{Int64, Float64}}
end

"""
    nv(pb::MCF)

Number of vertices.
"""
Graphs.nv(pb::MCF) = nv(pb.graph)

"""
    ne(pb::MCF)

Number of edges in the MCF network.
"""
Graphs.ne(pb::MCF) = ne(pb.graph)

"""
    nk(pb::MCF)

Number of demands in the MCF instance.
"""
nk(pb::MCF) = size(pb.demands, 1)

"""
    Base.show(io::IO, pb::MCF)

Show MCF object.
"""
function Base.show(io::IO, pb::MCF)
    println("MCF(nv = $(nv(pb)), ne = $(ne(pb)), nk = $(nk(pb)))")
end

"""
    UnknownMultiFlowFormat Exception
"""
struct UnknownMultiFlowFormat <: Exception end

"""
    load(dirname::String; format=:csv, edge_dir=:single)

Load MultiFlow problem from file. If format=:csv uses load_csv(dirname::String). `edge_dir` can be either `:single, :double, :undirected`.
"""
function load(dirname::String; format=:csv, directed=:single)
    if format==:csv
        return load_csv(dirname)
    else
        throw(UnknownMultiFlowFormat("Unknown format "*format))
    end
end

"""
    load_csv(dirname::String)

Load MultiFlow instance from CSV files. Default is to search for a link.csv and service.csv file.
"""
function load_csv(dirname::String; directed=false)
    linkpath = joinpath(dirname, "link.csv")
    servicepath = joinpath(dirname, "service.csv")
    if !isfile(linkpath) || !isfile(servicepath)
        throw(UnknownMultiFlowFormat("Could not find $linkfile and $servicefile"))
    end
    dflinks = CSV.read(linkpath, DataFrame)
    rename!(dflinks, strip.(lowercase.(names(dflinks))))
    dfservices = CSV.read(servicepath, DataFrame)
    rename!(dfservices, strip.(lowercase.(names(dfservices))))

    srcnodes = dflinks.srcnodeid
    dstnodes = dflinks.dstnodeid
    nodes = Set(vcat(srcnodes, dstnodes))
    nnodes = length(nodes)
    # keep a dictionary of original node ids to new ones
    vmap = Dict(nodes .=> 1:nnodes)

    capacities = dflinks.bandwidth
    costs = dflinks.cost
    narcs = size(dflinks, 1) # number of arcs

    # list of demands
    #store arrays for demands
    srcdemands = dfservices.srcnodeid
    dstdemands = dfservices.dstnodeid
    bandwidths = dfservices.bandwidth
    ndemands = size(dfservices, 1)# number of demands

    return MCF(srcnodes, dstnodes, costs, capacities, srcdemands, dstdemands, bandwidths)

end

"""
    graph(pb::MCF, weight=:cost)

Build the MCFs network. Returns a SimpleWeightedDiGraph object with weights equal to arc costs if `weight=:cost` and arc capacities if `weight=:capacity`.
"""
function graph(pb::MCF, weight=:cost)
    g = SimpleWeightedDiGraph(nv(pb))
    if weight==:cost
        return SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, pb.cost)
    elseif weight==:capacity
        return SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, pb.capacity)
    end
    throw(ArgumentError("Unrecognized argument weight=$(weight), should be either :cost or :capacity"))
end

"""
    weight_matrix(pb::MCF, weight=:cost)

Returns a sparse matrix with dimension `(nv(pb), nv(pb))` with values `[i,j]` equal to the cost of arc `(i,j)` if `weight=:cost` and capacity if `weight=:capacity`.
"""
function weight_matrix(pb::MCF, weight=:cost)
    if weight==:cost
        return sparse(pb.srcnodes, pb.dstnodes, pb.cost)
    elseif weight==:capacity
        return sparse(pb.srcnodes, pb.dstnodes, pb.capacity)
    end
    throw(ArgumentError("Unrecognized argument weight=$(weight), should be either :cost or :capacity"))
end

"""
    cost_matrix(pb::MCF)

Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc costs.
"""
function cost_matrix(pb::MCF)
    return weight_matrix(pb, :cost)
end

"""
    capacity_matrix(pb::MCF)

Return a sparse matrix with dimension `(nv(pb), nv(pb))` with values equal to arc capacity.
"""
function capacity_matrix(pb::MCF)
    return weight_matrix(pb, :capacity)
end

"""
    scale_cost(pb::MCF, cost_factor::Float64=1.0, capacity_factor::Float64=1.0)

Return a new MCF instance with costs scaled by a `cost_factor`, capacity and demand amounts scaled by `capacity_factor`.
"""
function scale(pb::MCF, cost_factor::Float64=1.0, capacity_factor::Float64=1.0)
    return MCF(
               pb.srcnodes,
               pb.dstnodes,
               pb.cost * cost_factor,
               pb.capacity * capacity_factor,
               pb.srcdemands,
               pb.dstdemands,
               pb.amount * capacity_factor
              )
end

"""
    normalize(pb::MCF)

Normalize MCF instance. Costs are scaled by `1 / max(pb.cost)`, capacity and demand amount are scaled by `1 / max(max(pb.capacity), max(pb.amount))`.
"""
function normalize(pb::MCF)
    return scale(pb, 1.0/maximum(pb.cost), 1.0/max(maximum(pb.capacity), maximum(pb.amount)))
end
