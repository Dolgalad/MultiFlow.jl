using GraphPlot
using Compose
using Cairo
using Colors

"""
    mcfplot(pb::MCF, layout::Function=spring_layout)

Plot MCF problem. _MultiFlows.jl_ uses the _GraphPlot.jl_ package for generating graph plots.
"""
function mcfplot(pb::MCF, layout::Function=spring_layout)
    edgecolor = [colorant"lightgray", colorant"orange"]
    # create a graph with demand edges
    demand_edge_flag = []
    g = SimpleDiGraph(nv(pb))
    for e in edges(pb.graph)
        add_edge!(g, src(e), dst(e))
        if has_demand(pb, src(e), dst(e))
            push!(demand_edge_flag, 2)
        else
            push!(demand_edge_flag, 1)
        end
    end
    for d in pb.demands
        if !has_edge(g, d.src, d.dst)
            # add edge and mark as demand edge
            add_edge!(g, d.src, d.dst)
            push!(demand_edge_flag, 2)
        end
    end
    edgestrokec = edgecolor[demand_edge_flag]
    # node labels
    nodelabel = collect(1:nv(pb))
    # layout
    loc_x, loc_y = layout(g)
    # Edge colors
    return gplot(g, loc_x, loc_y, edgestrokec=edgestrokec, nodelabel=nodelabel)
end
