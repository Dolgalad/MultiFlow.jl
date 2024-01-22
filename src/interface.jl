"""
    AbstractMultiFlow

An abstract type representing a multi flow problem.
"""
abstract type AbstractMultiFlow{T, U} end

"""
    AbstractDemand

An abstract type representing a demand with source and destination type T and demand type U
"""
abstract type AbstractDemand{T, U} end

#AbstractDemandList{T,U} = AbstractArray{AbstractDemand{T,U}, 1}

"""
    Demand

Concrete type representing a demand.
"""
struct Demand{T,U} <: AbstractDemand{T, U}
    src::T
    dst::T
    demand::U
end

"""
    BadDemandError{M}(m)

`Exception` thrown when a MultiFlow instance contains a demand with bad source or demand. 
"""
struct BadDemandError <: Exception end

"""
    MultiFlow

Concrete type representing a multi flow problem.
"""
struct MultiFlow{T, C, E, U}  <: AbstractMultiFlow{T, U}
    graph::AbstractGraph{T}
    demands::Vector{D} where {D<:AbstractDemand{T,U}}
    costs::Vector{C}
    capacities::Vector{E}
end

"""
    nk(m::AbstractMultiFlow)

Returns the number of demands
"""
nk(mf::AbstractMultiFlow) = length(mf.demands)

"""
Get list of demand that originate at node n
"""
function demands_originating_at(mf, n)
    ks = []
    for k in 1:nk(mf)
        if mf.demands[k].src == n
            push!(ks, k)
        end
    end
    return ks
end

"""
Get list of demand that terminate at node n
"""
function demands_terminating_at(mf, n)
    ks = []
    for k in 1:nk(mf)
        if mf.demands[k].dst == n
            push!(ks, k)
        end
    end
    return ks
end

