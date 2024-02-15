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
function load(dirname::String; format=:csv, edge_dir=:single)
    if format==:csv
        return load_csv(dirname, edge_dir=edge_dir)
    else
        throw(UnknownMultiFlowFormat("Unknown format "*format))
    end
end

"""
    load_csv(dirname::String)

Load MultiFlow instance from CSV files. Default is to search for a link.csv and service.csv file.
"""
function load_csv(dirname::String; edge_dir=:single)
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
    # need to check that indexes start at 1
    capacity = dflinks.capacity
    cost = dflinks.cost
    fg = FeatureDiGraph(srcnodes, dstnodes, hcat(cost, capacity))
    if edge_dir==:double
        fg = double_edges!(fg)
    end

    # list of demands
    #store arrays for demands
    srcdemands = dfservices.srcnodeid
    dstdemands = dfservices.dstnodeid
    amounts = dfservices.amount
    demands = [Demand(s,d,a) for (s,d,a) in zip(srcdemands, dstdemands, amounts)]
    return MCF(fg, demands)
end

"""
    save(pb::MCF, dirname::String)

Save MCF instance to `dirname`. Will create the files `<dirname>/link.csv` and `<dirname>/service.csv`. If folder does not exist it will be created.
"""
function save(pb::MCF, dirname::String; verbose::Bool=false)
    link_filename = joinpath(dirname, "link.csv")
    service_filename = joinpath(dirname, "service.csv")
    # link dataframe
    link_df = DataFrame(srcNodeId=pb.graph.srcnodes, 
                        dstNodeId=pb.graph.dstnodes, 
                        cost=arc_features(pb.graph, 1), 
                        capacity=arc_features(pb.graph, 2)
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
