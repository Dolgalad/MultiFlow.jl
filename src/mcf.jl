using DataFrames

"""
    UnknownMultiFlowFormat Exception
"""
struct UnknownMultiFlowFormat <: Exception end

"""
    load(dirname::String; format=:csv)

Load MultiFlow problem from file. If format=:csv uses load_csv(dirname::String)
"""
function load(dirname::String; format=:csv, double_arcs=false)
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
function load_csv(dirname::String; double_arcs=false)
    linkpath = joinpath(dirname, "link.csv")
    servicepath = joinpath(dirname, "service.csv")
    if !isfile(linkpath) || !isfile(servicepath)
        throw(UnknownMultiFlowFormat("Could not find $linkfile and $servicefile"))
    end
    dflinks = CSV.read(linkcsvfile, DataFrame)
    rename!(dflinks, strip.(lowercase.(names(dflinks))))
    dfservices = CSV.read(servicecsvfile, DataFrame)
    rename!(dfservices, strip.(lowercase.(names(dfservices))))

    srcnodes = dflinks.srcnodeid
    dstnodes = dflinks.dstnodeid

    capacities = dflinks.bandwidth
    costs = dflinks.cost
    narcs = size(dflinks, 1) # number of arcs

    #store arrays for demands
    srcdemands = dfservices.srcnodeid
    dstdemands = dfservices.dstnodeid
    bandwidths = dfservices.bandwidth
    ndemands = size(dfservices, 1)# number of demands

end
