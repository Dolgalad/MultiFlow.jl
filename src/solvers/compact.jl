using JuMP
#using CPLEX
using HiGHS

"""
    flow_constraint_rhs(pb::MCF, v::Int64, k::Int64, model::JuMP.Model, max_acceptance::Bool=false)

Compute the flow conservation right hand side value for vertex `v` and demand `k`.
"""
function flow_constraint_rhs(pb::MCF, v::Int64, k::Int64, model::JuMP.Model, max_acceptance::Bool=false)
    if v == pb.demands[k].src
        rhs = 1
        if max_acceptance
            y = model[:y]
            rhs -= y[k]
            #add_to_expression!(rhs, -y[k])
        end
    elseif v == pb.demands[k].dst
        rhs = -1
        if max_acceptance
            y = model[:y]
            rhs += y[k]
            #add_to_expression!(rhs, y[k])
        end
    else
        rhs = 0
    end
    return rhs
end

"""
    add_flow_constraints(pb::MCF, model::JuMP.Model, max_acceptance::Bool=false)

Add flow conservation constraints to the JuMP model.
"""
function add_flow_constraints(pb::MCF, model::JuMP.Model, max_acceptance::Bool=false)
    nnodes = nv(pb)
    narcs = ne(pb)
    nd = nk(pb)

    x = model[:x]
    am = edge_index_matrix(pb.graph)
    for v = 1:nnodes, k = 1:nd
        rhs = flow_constraint_rhs(pb, v, k, model, max_acceptance)
        @constraint(
            model,
            sum(x[k, am[v,a]] for a in outneighbors(pb.graph, v)) - sum(x[k, am[a,v]] for a in inneighbors(pb.graph, v)) == rhs
        )
    end
end

"""
    add_capacity_constraints(pb::MCF, model::JuMP.Model)

Add edge capacity constraints to the `JuMP.Model`.
"""
function add_capacity_constraints(pb::MCF, model::JuMP.Model)
    caps = capacities(pb)
    x = model[:x]
    for a = 1:ne(pb)
        @constraint(model, sum(pb.demands[k].amount * x[k, a] for k = 1:nk(pb)) <= caps[a])
    end
end

"""
    add_objective_value(pb::MCF, model::JuMP.Model, max_acceptance::Bool=false)

Add objective value to the `JuMP.Model`.
"""
function add_objective_value(pb::MCF, model::JuMP.Model, max_acceptance::Bool=false)
    csts = costs(pb)
    da = demand_amounts(pb)
    x = model[:x]
    #obj_expr = sum(csts[a] * pb.demands[k].amount * x[k,a] for a=1:ne(pb), k=1:nk(pb))
    obj_expr = sum((da .* x) .* csts')
    if max_acceptance
        y = model[:y]
        bigM = sum(csts)
        #obj_expr += sum(bigM * pb.demands[k].amount * y[k] for k=1:nk(pb))
        #add_to_expression!(obj_expr, sum(bigM * pb.demands[k].amount * y[k] for k=1:nk(pb)))
        add_to_expression!(obj_expr, bigM * sum(y .* da))

    end
    @objective(
        model,
        Min,
        obj_expr
    )
end

"""
    add_filter_constraints(model::JuMP.Model, filter::Union{Nothing,AbstractMatrix{Bool}}=nothing)

Add filter constraints. If `filter=nothing` does nothing, otherwise `filter` should be a ``|K|\\times |A|`` 0-1-matrix. If `filter[k,a] = 0` a constraint ``x_a^k = 0` is added to the model.
"""
function add_filter_constraints(model::JuMP.Model, filter::Union{Nothing,AbstractMatrix{Bool}}=nothing)
    if !isnothing(filter)
        x = model[:x]
        #@constraint(model, x[filter == 0] == 0)
        for k in 1:size(filter,1)
            for a in 1:size(filter,2)
                if filter[k,a] == 0
                    @constraint(model, x[k,a] == 0)
                end
            end
        end
    end
end

"""
    create_compact_model(pb::MCF)

Create JuMP model corresponding to the provided MCF problem `pb`.
"""
function create_compact_model(pb::MCF; 
                              relaxed::Bool=true, 
                              max_acceptance::Bool=false,
                              verbose::Bool=false,
                              filter::Union{Nothing,AbstractMatrix{Bool}}=nothing,
                              optimizer::DataType=HiGHS.Optimizer,
                              direct::Bool=false
    )
    if direct
        model = JuMP.direct_model(optimizer())
    else
        model = Model(optimizer; add_bridges=false)
    end
    if !verbose
        set_silent(model)
    end
    # arc-flow variables
    if relaxed
        @variable(model, x[1:nk(pb), 1:ne(pb)] >= 0)
    else
        @variable(model, x[1:nk(pb), 1:ne(pb)], Bin)

    end
    if max_acceptance
        # rejection variables
        if relaxed
            @variable(model, y[1:nk(pb)] >= 0)

        else
            @variable(model, y[1:nk(pb)], Bin)
        end
    end
    # add the filter constraints
    add_filter_constraints(model, filter)
    #println("t_fc = $t_fc")
    
    # flow constraints:
    add_flow_constraints(pb, model, max_acceptance)
    #println("t_flc = $t_flc")
    # capacity constraint:

    add_capacity_constraints(pb, model)
    #println("t_cc = $t_cc")
    # objective value

    add_objective_value(pb, model, max_acceptance)
    #println("t_ao = $t_ao")

    # relax integrality constraints if needed
    #if relaxed
    #    t_r = @elapsed relax_integrality(model)
    #    println("t_r = $t_r")
    #end
    return model
end

"""
    solve_compact(pb::MCF)

Solve the compact formulation with state of the art solver. Returns a tuple `(MCFSolution, SolverStatistics)`.

# Example
```jldoctest; setup = :(using Graphs), filter = r"[0-9.]+"
julia> gr = grid((3,3));

julia> pb = MCF(gr, ones(ne(gr)), ones(ne(gr)), [Demand(1,9,1.0), Demand(1,6,1.0)])
MCF(nv = 9, ne = 24, nk = 2)
	Demand{Int64, Float64}(1, 9, 1.0)
	Demand{Int64, Float64}(1, 6, 1.0)

julia> sol, ss = solve_compact(pb);

julia> sol
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 2, 5, 8, 9])
	Demand k = 2
		1.0 on VertexPath{Int64}([1, 4, 5, 6])

julia> ss
{
    "solve_time": 0.00042557716369628906,
    "objective_sense": "MIN_SENSE",
    "dual_objective_value": 7.0,
    "result_count": 1,
    "node_count": -1,
    "objective_value": 7.0,
    "objective_bound": 0.0,
    "termination_status": "OPTIMAL",
    "simplex_iterations": 10,
    "barrier_iterations": 0,
    "dual_status": "FEASIBLE_POINT",
    "primal_status": "FEASIBLE_POINT",
    "solver_name": "HiGHS",
    "relative_gap": null
}

```

"""
function solve_compact(pb::MCF; 
                       max_acceptance::Bool=false,
                       relaxed::Bool=true,
                       filter::Union{Nothing,AbstractMatrix{Bool}}=nothing,
                       optimizer::DataType=HiGHS.Optimizer,
                       timelimit::Union{Nothing,Real}=nothing
    )
    model = create_compact_model(pb, max_acceptance=max_acceptance, filter=filter, relaxed=relaxed, optimizer=optimizer)
    #println("t_model = ", t_model)
    #flush(stdout)
    if !isnothing(timelimit)
        set_time_limit_sec(model, timelimit)
    end
    optimize!(model)
    #println("t_opt : ", t_opt)
    #flush(stdout)
    tstatus = termination_status(model)
    if tstatus==OPTIMAL || tstatus==TIMEOUT
        stats = SolverStatistics()
        add_JuMP_statistics(stats, model)
        sol = solution_from_arc_flow_values(value.(model[:x]), pb)
        #println("t_sol : ", t_sol)
        #flush(stdout)
        return sol, stats
    elseif tstatus==INFEASIBLE
        throw(ErrorException("Infeasible problem"))
    end
    return nothing
end
