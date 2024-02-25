"""
    AbstractSparsifier

Abstract type shared by all Sparsifier models. These are objects who implement the [`sparsify(sprs::AbstractSparsifier, pb::MCF)`](@ref) that returns a `(nk(pb), ne(pb))` 0-1-matrix where if `M[k,a]==1` arc ``a`` is classifier as being used to route demand ``k``
"""
abstract type AbstractSparsifier end

"""
    sparsify(sprs::AbstractSparsifier, pb::MCF)

Produces the arc-demand classification result. This function needs to be defined for each type of sparsifier.
"""
function sparsify(sprs::AbstractSparsifier, pb::MCF)
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
    sparsify(sprs::SPSparsifier, pb::MCF)

Sparsify the MCF `pb`.
"""
function sparsify(sprs::SPSparsifier, pb::MCF)
end
