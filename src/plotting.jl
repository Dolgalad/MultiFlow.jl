using Colors
"""
    plot(mf::AbstractMultiFlow, path::String)

A method for plotting multi flow instances
"""
function plot(mf::AbstractMultiFlow; 
              savepath::String="", 
              figsize::Tuple{Measures.Length,Measures.Length}=(16cm,16cm),
              nodelabel=AbstractArray=1:nv(mf.graph),
)
    # demand node colors
    #demand_colors = distinguishable_colors(nk(mf))
    #push!(demand_colors, colorant"lightblue")
    #nodefillc = []
    #for n in 1:nv(mf)
    #    # list of demands originating or terminating at node n
    #    ks = vcat(demands_originating_at(mf, n), demands_terminating_at(mf, n))
    #    println("ks : ", ks, ", ", length(ks))
    #    if length(ks)>0
    #        push!(nodefillc, demand_colors[ks[1]])
    #    else
    #        push!(nodefillc, demand_colors[end])
    #    end
    #end

    plt=gplot(mf.graph,
        nodelabel=nodelabel,
        #nodefillc=nodefillc,
    )
    if !isempty(savepath)
        draw(PNG(savepath, figsize[1], figsize[2]), plt)
    end
end

