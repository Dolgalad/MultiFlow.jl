using JSON

"""
    SolverStatistics

Container for storing solver statistics.
"""
struct SolverStatistics
    stats::Dict
end

"""
    SolverStatistics()

Initialize empty solver statistics.
"""
function SolverStatistics()
    return SolverStatistics(Dict())
end

"""
    Base.getindex(ss::SolverStatistics, k::String)

Get statistic corresponding to key `k`.
"""
Base.getindex(ss::SolverStatistics, k::String) = ss.stats[k]

"""
    Base.setindex!(ss::SolverStatistics, v::Any, k::String)

Set statistic with key `k` and value `v`.
"""
function Base.setindex!(ss::SolverStatistics, v::Any, k::String)
    ss.stats[k] = v
end

function Base.show(io::IO, ss::SolverStatistics)
    JSON.print(io, ss.stats, 4)
end

"""
    add_JuMP_statistics(ss::SolverStatistics, model::JuMP.Model)

Add statistics of a JuMP model object.

| **Function**         | **Description**                             |
|----------------------|---------------------------------------------|
| solver_name          | Name of the LP solver used                  |
| termination_status   | Status of the solver                        |
| solve_time           | Solve time in seconds                       |
| objective_sense      | Sense of the optimization problem (min/max) |
| objective_value      | Objective value of the problem              |
| objective_bound      | Objective value bound                       |
| relative_gap         | Solution relative gap                       |
| dual_objective_value | Dual objective value of the problem         |
| primal_status        | Status of the primal solution               |
| dual_status          | Status of the dual solution                 |
| node_count           | Branch and bound tree node count            |
| simplex_iterations   | Number of simplex iterations performed      |
| barrier_iterations   | Number of barrier iterations performed      |
| result_count         | Number of results available                 |


"""
function add_JuMP_statistics(ss::SolverStatistics, model::JuMP.Model; verbose::Bool=false)
    jump_statistics_f = [
                         JuMP.solver_name,
                         JuMP.termination_status,
                         JuMP.solve_time,
                         JuMP.objective_sense,
                         JuMP.objective_value,
                         JuMP.objective_bound,
                         JuMP.relative_gap,
                         JuMP.dual_objective_value,
                         JuMP.primal_status,
                         JuMP.dual_status,
                         JuMP.node_count,
                         JuMP.simplex_iterations,
                         JuMP.barrier_iterations,
                         JuMP.result_count,
                        ]
    for f in jump_statistics_f
        try
            ss[String(Symbol(f))] = f(model)
        catch
            if verbose
                @warn "Failed to retrieve statistic " * String(Symbol(f))
            end
        end
    end
end
