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
    MultiFlow

Concrete type representing a multi flow problem.
"""
struct MultiFlow{T, U}  <: AbstractMultiFlow{T, U}
    graph::AbstractGraph{T}
    demands::Vector{D} where {D<:AbstractDemand{T,U}}
end

