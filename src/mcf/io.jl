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
function load(dirname::String; format=:csv, edge_dir=:single, keyargs...)
    if format==:csv
        return load_csv(dirname, edge_dir=edge_dir; keyargs...)
    else
        throw(UnknownMultiFlowFormat("Unknown format "*format))
    end
end

"""
    load_csv(dirname::String)

Load MultiFlow instance from CSV files. Default is to search for a link.csv and service.csv file.
"""
function load_csv(dirname::String; 
                  edge_dir=:single, 
                  srcnode_fieldname="srcnodeid",
                  dstnode_fieldname="dstnodeid",
                  cost_fieldname="cost",
                  capacity_fieldname="capacity",
                  demand_srcnode_fieldname="srcnodeid",
                  demand_dstnode_fieldname="dstnodeid",
                  demand_amount_fieldname="amount"
    )
    linkpath = joinpath(dirname, "link.csv")
    servicepath = joinpath(dirname, "service.csv")
    if !isfile(linkpath) || !isfile(servicepath)
        throw(ArgumentError("Could not find $linkpath and $servicepath"))
    end
    dflinks = CSV.read(linkpath, DataFrame)
    rename!(dflinks, strip.(lowercase.(names(dflinks))))
    dfservices = CSV.read(servicepath, DataFrame)
    rename!(dfservices, strip.(lowercase.(names(dfservices))))

    srcnodes = dflinks[!, srcnode_fieldname]
    dstnodes = dflinks[!, dstnode_fieldname]
    # need to check that indexes start at 1
    capacity = dflinks[!, capacity_fieldname]
    cost = dflinks[!, cost_fieldname]
    fg = FeatureDiGraph(srcnodes, dstnodes, hcat(cost, capacity))
    if edge_dir==:double
        fg = double_edges!(fg)
    end

    # list of demands
    #store arrays for demands
    srcdemands = dfservices[!, demand_srcnode_fieldname]
    dstdemands = dfservices[!, demand_dstnode_fieldname]
    amounts = dfservices[!, demand_amount_fieldname]
    demands = [Demand(s,d,a) for (s,d,a) in zip(srcdemands, dstdemands, amounts)]
    return MCF(fg, demands)
end

"""
    save(pb::MCF, dirname::String; verbose::Bool=false)

Save MCF instance to `dirname`. Will create the files `<dirname>/link.csv` and `<dirname>/service.csv`. If folder does not exist it will be created.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(Int64,7), 1:7, [Demand(1,2,2)]);

julia> save(pb, "instance")
("instance/link.csv", "instance/service.csv")

```

"""
function save(pb::MCF, dirname::String; verbose::Bool=false)
    link_filename = joinpath(dirname, "link.csv")
    service_filename = joinpath(dirname, "service.csv")
    # link dataframe
    link_df = DataFrame(srcNodeId=pb.graph.srcnodes, 
                        dstNodeId=pb.graph.dstnodes, 
                        cost=edge_features(pb.graph, 1), 
                        capacity=edge_features(pb.graph, 2)
                       )
    #link_df = unique(link_df)
    service_df = DataFrame(srcNodeId=[d.src for d in pb.demands], 
                           dstNodeId=[d.dst for d in pb.demands], 
                           amount=[d.amount for d in pb.demands])
    if verbose
        println("Saving instance to $dirname")
    end
    mkpath(dirname)
    CSV.write(link_filename, link_df)
    CSV.write(service_filename, service_df)
    return link_filename, service_filename
end

"""
    is_instance_dir(dirname::String)

Check if a directory contains files `link.csv, service.csv`.

```julia
julia> save(pb, "test_instance")
("test_instance/link.csv", "test_instance/service.csv")

julia> is_instance_dir("test_instance")
true
```
"""
is_instance_dir(dirname::String) = isfile(joinpath(dirname, "link.csv")) && isfile(joinpath(dirname, "service.csv"))
