"""
    AbstractDemand

Abstract type subtyped for creating MCF demands.
"""
abstract type AbstractDemand end

"""
    MCFDemand

Concrete demand type for MCF problems. A demand has a source node `src`, destination node `dst` and an amount `amount`.
"""
struct Demand{T,N} <: AbstractDemand
    src::T
    dst::T
    amount::N
end
