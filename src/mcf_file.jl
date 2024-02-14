"""
    UnknownMCFFormat Exception

Exception raised when trying to load instance from unknown file format.
"""
struct UnknownMultiFlowFormat <: Exception end

"""
    load(dirname::String; format=:csv, edge_dir=:single)

Load MultiFlow problem from file. If format=:csv uses [`load_csv(dirname::String)`](@ref) (at this time CSV files are the only supported format). `edge_dir` can be one of the following : 
- `:single` : each edge in the input file is interpreted as a directed edge
- `:double` : each edge in the input file is interpreted as existing in both directions with the same attributes and features

"""
function load(dirname::String; format=:csv, edge_direction=:single)
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
function load_csv(dirname::String; edge_direction=false)
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

