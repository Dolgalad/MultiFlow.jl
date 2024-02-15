using GraphPlot
using Compose
using Cairo
using Colors

"""
    mcfplot(pb::MCF, layout::Function=spring_layout;
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
    )

Plot MCF problem. _MultiFlows.jl_ uses the _GraphPlot.jl_ package for generating graph plots. Edge line widths are scaled onto `[minedgelinewidth, maxedgelinewidth]` as a function of edge capacity and demand amounts.
"""
function mcfplot(pb::MCF, layout::Function=spring_layout; 
        minedgelinewidth=1.0,
        maxedgelinewidth=10.0
    )
    edgecolor = [colorant"lightgray", colorant"orange"]
    # create a graph with demand edges
    demand_edge_flag = []
    edge_capacities = capacities(pb)
    demand_amounts = [d.amount for d in pb.demands]
    min_edge_capacity, max_edge_capacity = minimum(edge_capacities), maximum(edge_capacities)
    min_edge_capacity = min(min_edge_capacity, minimum(demand_amounts))
    max_edge_capacity = max(max_edge_capacity, maximum(demand_amounts))
    if min_edge_capacity == max_edge_capacity
        min_edge_capacity = 0
        max_edge_capacity = 1
        maxedgelinewidth = 2
        minedgelinewidth = 1
    end
    edgelinewidth = []
    g = SimpleDiGraph(nv(pb))
    for (i, e) in enumerate(edges(pb.graph))
        add_edge!(g, src(e), dst(e))
        # get demands
        demds = demands(pb, src(e), dst(e))
        if !Base.isempty(demds)
            push!(demand_edge_flag, 2)
            amount = min(max_edge_capacity, max(edge_capacities[i], sum([d.amount for d in demds])))
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))

        else
            push!(demand_edge_flag, 1)
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (edge_capacities[i] - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end

    end
    for d in pb.demands
        if !has_edge(g, d.src, d.dst)
            # add edge and mark as demand edge
            add_edge!(g, d.src, d.dst)
            push!(demand_edge_flag, 2)
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (d.amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end
    end
    edgestrokec = edgecolor[demand_edge_flag]
    # node labels
    nodelabel = collect(1:nv(pb))
    # layout
    loc_x, loc_y = layout(g)
    println([minimum(edgelinewidth), maximum(edgelinewidth)])
    # Edge colors
    return gplot(g, loc_x, loc_y, 
                 edgestrokec=edgestrokec, 
                 nodelabel=nodelabel, 
                 edgelinewidth=edgelinewidth)
end
