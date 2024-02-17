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
        end
    elseif v == pb.demands[k].dst
        rhs = -1
        if max_acceptance
            y = model[:y]
            rhs += y[k]
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
    x = model[:x]
    obj_expr = sum(csts[a] * pb.demands[k].amount * x[k,a] for a=1:ne(pb), k=1:nk(pb))
    if max_acceptance
        y = model[:y]
        bigM = sum(csts)
        obj_expr += sum(bigM * pb.demands[k].amount * y[k] for k=1:nk(pb))
    end
    @objective(
        model,
        Min,
        obj_expr
    )
end

"""
    create_compact_model(pb::MCF)

Create JuMP model corresponding to the provided MCF problem `pb`.
"""
function create_compact_model(pb::MCF; 
                              relaxed::Bool=true, 
                              max_acceptance::Bool=false,
                              verbose::Bool=false
    )
    model = Model(HiGHS.Optimizer)
    if !verbose
        set_silent(model)
    end
    # arc-flow variables
    @variable(model, x[1:nk(pb), 1:ne(pb)], Bin)
    if max_acceptance
        # rejection variables
        @variable(model, y[1:nk(pb)], Bin)
    end
    # add the filter constraints
    #add_filter_Constraints(model, filter)
    
    # flow constraints:
    add_flow_constraints(pb, model, max_acceptance)
    # capacity constraint:

    add_capacity_constraints(pb, model)
    # objective value

    add_objective_value(pb, model, max_acceptance)

    # relax integrality constraints if needed
    if relaxed
        relax_integrality(model)
    end
    return model
end

"""
    solve_compact(pb::MCF)

Solve the compact formulation with state of the art solver.

# Example
```jldoctest; setup = :(using Graphs)
julia> gr = grid((3,3));

julia> pb = MCF(gr, ones(ne(gr)), ones(ne(gr)), [Demand(1,9,1.0), Demand(1,6,1.0)])
MCF(nv = 9, ne = 24, nk = 2)
	Demand{Int64, Float64}(1, 9, 1.0)
	Demand{Int64, Float64}(1, 6, 1.0)

julia> solve_compact(pb)
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 2, 5, 8, 9])
	Demand k = 2
		1.0 on VertexPath{Int64}([1, 4, 5, 6])

```
"""
function solve_compact(pb::MCF)
    model = create_compact_model(pb)
    optimize!(model)
    if termination_status(model)==OPTIMAL
        return solution_from_arc_flow_values(value.(model[:x]), pb)
    end
    return nothing
end
