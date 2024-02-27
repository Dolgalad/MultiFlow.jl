"""
    AbstractSparsifier

Abstract type shared by all Sparsifier models. These are objects who implement the [`sparsify(sprs::AbstractSparsifier, pb::MCF)`](@ref) that returns a `(nk(pb), ne(pb))` 0-1-matrix where if `M[k,a]==1` arc ``a`` is classifier as being used to route demand ``k``
"""
abstract type AbstractSparsifier end

"""
    sparsify(pb::MCF, sprs::AbstractSparsifier)

Produces the arc-demand classification result. This function needs to be defined for each type of sparsifier.
"""
function sparsify(pb::MCF{T,N}, sprs::AbstractSparsifier) where {T,N}
    throw(ErrorException("Not implemented, typeof(sprs) = $(typeof(sprs)), typeof(pb) = $(typeof(pb))"))
end

"""
    SPSparsifier

Shortest Path Sparsifier. For each demand will keep only arcs belonging to the ``k`` shortest paths.
"""
struct SPSparsifier <: AbstractSparsifier
    k::Int64
end

"""
    sparsify(pb::MCF{T,N}, sprs::SPSparsifier) where {T,N}


Sparsify the MCF `pb`.
"""
function sparsify(pb::MCF{T,N}, sprs::SPSparsifier; dstmx::AbstractMatrix=cost_matrix(pb)) where {T,N}
    done_endpoints = Dict{Tuple{T,T}, Int64}()
    I,J = Vector{Int64}[], Vector{Int64}[]
    for (k,demand) in enumerate(pb.demands)
        if (demand.src,demand.dst) in keys(done_endpoints)
            push!(J, J[done_endpoints[demand.src,demand.dst]])
            push!(I, k * ones(Int64, size(J[k],1)))
        else
            # look for k shortest paths
            pths = shortest_paths(pb.graph, demand.src, demand.dst, K=sprs.k, dstmx=dstmx)
            done_endpoints[demand.src,demand.dst] = k
            if sprs.k==1
                push!(J, edge_indices(pths, pb.graph))
                push!(I, k*ones(Int64, size(J[k],1)))
            else
                eidx = vcat([edge_indices(p,pb.graph) for p in pths]...)
                push!(J, eidx)
                push!(I, k*ones(Int64, size(eidx,1)))
            end
        end
    end
    I,J = vcat(I...), vcat(J...)
    return sparse(I,J,ones(Bool,size(I,1)), nk(pb), ne(pb))
end
