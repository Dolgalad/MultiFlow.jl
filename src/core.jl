using DataFrames
using CSV
using Graphs

"""
    UnknownMultiFlowFormat Exception
"""
struct UnknownMultiFlowFormat <: Exception end

"""
    load(dirname::String; format=:csv)

Load MultiFlow problem from file. If format=:csv uses load_csv(dirname::String)
"""
function load(dirname::String; format=:csv, directed=false)
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

    if directed
        graph = Graphs.SimpleDiGraph(nnodes)
    else
        graph = Graphs.SimpleGraph(nnodes)
    end
    # add arcs
    for i in 1:narcs
        add_edge!(graph, vmap[srcnodes[i]], vmap[dstnodes[i]])
    end

    # list of demands
    #store arrays for demands
    srcdemands = dfservices.srcnodeid
    dstdemands = dfservices.dstnodeid
    bandwidths = dfservices.bandwidth
    ndemands = size(dfservices, 1)# number of demands

    demands = Demand{Int64,typeof(bandwidths[1])}[]
    for i in 1:ndemands
        push!(demands, Demand(vmap[srcdemands[i]], vmap[dstdemands[i]], bandwidths[i]))
    end

    return MultiFlow(graph, demands, costs, capacities)

end

"""
    get_graph(scrnodes, dstnodes, costs, capacities)

Build the graph.
"""
function get_graph(srcs, dsts, costs, capacities)
    return nothing
end
