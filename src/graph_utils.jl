#using Graphs
"""
    arc_index_matrix(g::AbstractGraph{T})
Get the graphs arc index matrix M.
"""
function arc_index_matrix(g::AbstractGraph{T}) where {T<:Integer}
    am = zeros(Int, Graphs.nv(g), Graphs.nv(g))
    for (i,e) in enumerate(Graphs.edges(g))
        am[src(e), dst(e)] = i
    end
    return am
end
