using GraphPlot
using Compose
using Cairo
using Colors

function graph_with_demand_edges(pb::MCF)
    g = SimpleDiGraph(nv(pb))
    for e in edges(pb.graph)
        add_edge!(g, src(e), dst(e))
    end
    for demand in pb.demands
        add_edge!(g, demand.src, demand.dst)
    end
    return g
end

"""
    mcfplot(pb::MCF, layout::Function=spring_layout;
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
    )

Plot MCF problem. _MultiFlows.jl_ uses the _GraphPlot.jl_ package for generating graph plots. Edge line widths are scaled onto `[minedgelinewidth, maxedgelinewidth]` as a function of edge capacity and demand amounts.
"""
function mcfplot(pb::MCF, layout::Function=spring_layout; keyargs...)
    # create simple directed graph with demand edges
    g = graph_with_demand_edges(pb)
    loc_x, loc_y = layout(g)
    return mcfplot(pb, loc_x, loc_y; keyargs...)
end

"""
    mcfplot(pb::MCF, loc_x::Vector{Float64}, loc_y::Vector{Float64};
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
    )

Plot MCF problem. _MultiFlows.jl_ uses the _GraphPlot.jl_ package for generating graph plots. Edge line widths are scaled onto `[minedgelinewidth, maxedgelinewidth]` as a function of edge capacity and demand amounts.
"""
function mcfplot(pb::MCF, loc_x::Vector{Float64}, loc_y::Vector{Float64}; 
        minedgelinewidth=1.0,
        maxedgelinewidth=10.0
    )
    edgecolor = [colorant"lightgray", colorant"orange"]
    # create a graph with demand edges
    demand_edge_flag_map = Dict()
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
            demand_edge_flag_map[src(e), dst(e)] = 2
            #push!(demand_edge_flag, 2)
            amount = min(max_edge_capacity, max(edge_capacities[i], sum([d.amount for d in demds])))
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))

        else
            #push!(demand_edge_flag, 1)
            demand_edge_flag_map[src(e), dst(e)] = 1
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (edge_capacities[i] - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end

    end
    for d in pb.demands
        if !has_edge(g, d.src, d.dst)
            # add edge and mark as demand edge
            add_edge!(g, d.src, d.dst)
            #push!(demand_edge_flag, 2)
            demand_edge_flag_map[d.src, d.dst] = 2
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (d.amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end
    end
    edgestrokec = [edgecolor[demand_edge_flag_map[src(e), dst(e)]] for e in edges(g)]
    # node labels
    nodelabel = collect(1:nv(g))
    # Edge colors
    return gplot(g, loc_x, loc_y, 
                 edgestrokec=edgestrokec, 
                 nodelabel=nodelabel, 
                 edgelinewidth=edgelinewidth)
end

"""
    mcfsolplot(sol::MCFSolution, pb::MCF, layout::Function=spring_layout;
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
    )

Plot MCF solution.
"""
function mcfsolplot(sol::MCFSolution, pb::MCF, layout::Function=spring_layout; keyargs...)
    # layout
    loc_x, loc_y = layout(g)
    return mcfsolplot(sol, pb, loc_x, loc_y; keyargs...)
end

"""
    mcfsolplot(sol::MCFSolution, pb::MCF, loc_x::Vector{R1}, loc_y::Vector{R2};
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
    )

Plot MCF solution.
"""
function mcfsolplot(sol::MCFSolution, pb::MCF, loc_x::Vector{Float64}, loc_y::Vector{Float64};
            minedgelinewidth=1.0,
            maxedgelinewidth=10.0,
            edgealpha=.5,
            linetype="straight",
    )
    # edge colors : edgecolor, demandedgecolor, solutionedgecolor
    edgecolor = [RGBA(colorant"lightgray", edgealpha), colorant"orange"]
    edgecolor = vcat(edgecolor, distinguishable_colors(nk(pb), colorant"blue"))
    # create a graph with demand edges
    demand_edge_flag_map = Dict()
    edge_label_map = Dict()
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
            demand_edge_flag_map[src(e), dst(e)] = 2
            amount = min(max_edge_capacity, max(edge_capacities[i], sum([d.amount for d in demds])))
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        else
            demand_edge_flag_map[src(e), dst(e)] = 1
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (edge_capacities[i] - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end

        # check if edge is used in the solution
        for k in 1:nk(pb)
            if has_edge(sol, k, src(e), dst(e))
                # edge used for demand k
                demand_edge_flag_map[src(e), dst(e)] = 2+k
            end
        end
    end
    for d in pb.demands
        if !has_edge(g, d.src, d.dst)
            # add edge and mark as demand edge
            add_edge!(g, d.src, d.dst)
            demand_edge_flag_map[d.src, d.dst] = 2
            push!(edgelinewidth, minedgelinewidth + (maxedgelinewidth - minedgelinewidth) * (d.amount - min_edge_capacity) / (max_edge_capacity - min_edge_capacity))
        end
    end
    edgestrokec = [edgecolor[demand_edge_flag_map[src(e), dst(e)]] for e in edges(g)]
    # node labels
    nodelabel = collect(1:nv(pb))
    # Edge colors
    return gplot(g, loc_x, loc_y, 
                 edgestrokec=edgestrokec, 
                 nodelabel=nodelabel, 
                 edgelinewidth=edgelinewidth,
                 linetype=linetype,
                )

end

