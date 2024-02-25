"""
    MCFRestrictedMasterProblem

Concrete type representing the MCF Restricted Master Problem data.
"""
struct MCFRestrictedMasterProblem
    model::JuMP.Model
    columns::Vector{Vector{Vector{Int64}}}
end

"""
    MCFRestrictedMasterProblem(pb::MCF, direct::Bool=true)

Initialize the RMP for MCF problem `pb`. If `direct=true` the LP solver is used in direct mode.

# Example
```jldoctest; setup = :(using JuMP),  filter = r"[0-9.]+"
julia> pb = load("../instances/toytests/test1")
MCF(nv = 7, ne = 10, nk = 3)
	Demand{Int64, Int64}(1, 7, 5)
	Demand{Int64, Int64}(2, 6, 5)
	Demand{Int64, Int64}(3, 7, 5)

julia> rmp = MCFRestrictedMasterProblem(pb);

julia> typeof(rmp.model)
Model (alias for GenericModel{Float64})

```
"""
function MCFRestrictedMasterProblem(pb::MCF;
                                    direct::Bool=true,
    )
    if direct
        model = JuMP.direct_model(HiGHS.Optimizer())
    else
        model = Model(HiGHS.Optimizer)
    end
    set_silent(model)
    for k in 1:nk(pb)
        # add rejection variables
        model[Symbol("x$k")] = @variable(model, [1:1], lower_bound=0, base_name="x$k")
        set_name(model[Symbol("x$k")][1], "y$k")
    end
    # capacity constraints
    caps = capacities(pb)
    @constraint(model, capacity[a in 1:ne(pb)], 0 <= caps[a])
    # convexity constraints
    @constraint(model, convexity[k in 1:nk(pb)], model[Symbol("x$k")][1] == 1)
    # objective
    bigM = sum(costs(pb)) * demand_amounts(pb)
    @objective(model, Min, sum(bigM[k] * model[Symbol("x$k")][1] for k in 1:nk(pb)))
    return MCFRestrictedMasterProblem(model, Vector{Vector{Int64}}[[] for _ in 1:nk(pb)])
end

"""
    MCFPricingProblem

MCF Pricing problem data container.
"""
struct MCFPricingProblem
    srcnodes::Vector{Int64}
    dstnodes::Vector{Int64}
    demands::Vector{Demand}
    costs::Vector{Float64}
    capacity_duals::Vector{Float64}
    convexity_duals::Vector{Float64}
end

"""
    MCFPricingProblem

Pricing problem constructor for problem `pb`.

# Example
```jldoctest; setup = :(pb = load("../instances/toytests/test1"))
julia> prp = MCFPricingProblem(pb)
MCFPricingProblem([1, 1, 2, 4, 2, 3, 4, 6, 1, 5], [4, 2, 4, 7, 3, 4, 6, 7, 5, 7], Demand[Demand{Int64, Int64}(1, 7, 5), Demand{Int64, Int64}(2, 6, 5), Demand{Int64, Int64}(3, 7, 5)], [2.0, 3.0, 3.0, 8.0, 4.0, 8.0, 3.0, 3.0, 80.0, 20.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

```
"""
function MCFPricingProblem(pb::MCF)
    return MCFPricingProblem(pb.graph.srcnodes,
                             pb.graph.dstnodes,
                             pb.demands, 
                             costs(pb), 
                             zeros(ne(pb)), 
                             zeros(nk(pb)))
end

"""
    solve!(rmp::MCFRestrictedMasterProblem)

Solve the RMP. Calls `JuMP.optimize!(rmp.model)`. Returns the objective value and duals for the capacity and convexity constraints.

# Example
```jldoctest; setup = :(pb = load("../instances/toytests/test1"); rmp = MCFRestrictedMasterProblem(pb))
julia> obj, sigma, tau = solve!(rmp)
(2010.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [670.0, 670.0, 670.0])

```
"""
function solve!(rmp::MCFRestrictedMasterProblem)
    optimize!(rmp.model)
    capacity_duals = [JuMP.dual(c) for c in rmp.model[:capacity]]
    convexity_duals = [JuMP.dual(c) for c in rmp.model[:convexity]]
    return JuMP.objective_value(rmp.model), capacity_duals, convexity_duals
end

"""
    add_column!(rmp::MCFRestrictedMasterProblem, k::Int64, column::Vector{Int64}, demand_amount::Float64, column_weight::Float64)

Add column for demand `k` to the master problem. `column` is a path ``p`` represented by a sequence of edges, a new variable ``x^k_p`` is added to the master problem, the capacity and convexity constraints are updated to take the new variable into account as well as the objective function.

# Example
```jldoctest addcol; setup = :(using JuMP, Graphs; pb = load("../instances/toytests/test1"); rmp = MCFRestrictedMasterProblem(pb))
julia> num_variables(rmp.model)
3

julia> rmp.model[:capacity]
10-element Vector{ConstraintRef{Model, MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.LessThan{Float64}}, ScalarShape}}:
 capacity[1] : 0 ≤ 6
 capacity[2] : 0 ≤ 12
 capacity[3] : 0 ≤ 12
 capacity[4] : 0 ≤ 5
 capacity[5] : 0 ≤ 11
 capacity[6] : 0 ≤ 20
 capacity[7] : 0 ≤ 10
 capacity[8] : 0 ≤ 20
 capacity[9] : 0 ≤ 10
 capacity[10] : 0 ≤ 10

julia> rmp.model[:convexity]
3-element Vector{ConstraintRef{Model, MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.EqualTo{Float64}}, ScalarShape}}:
 convexity[1] : y1 = 1
 convexity[2] : y2 = 1
 convexity[3] : y3 = 1

julia> objective_function(rmp.model)
670 y1 + 670 y2 + 670 y3

```

Now lets find a path to add to the RMP for demand `k=1` and convert it to column format i.e. Vector of edge indices.

```jldoctest addcol
julia> p = VertexPath(enumerate_paths(dijkstra_shortest_paths(pb.graph, 1), 7))
VertexPath{Int64}([1, 4, 6, 7])

julia> column = edge_indices(p, pb.graph)
3-element Vector{Int64}:
 1
 7
 8

julia> pw = path_weight(p, pb.graph)
8
```

Now we add the column to the RMP : 
```jldoctest addcol
julia> add_column!(rmp, 1, column, pb.demands[1].amount, pw)

julia> rmp.model[:capacity]
10-element Vector{ConstraintRef{Model, MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.LessThan{Float64}}, ScalarShape}}:
 capacity[1] : 5 x1[1] ≤ 6
 capacity[2] : 0 ≤ 12
 capacity[3] : 0 ≤ 12
 capacity[4] : 0 ≤ 5
 capacity[5] : 0 ≤ 11
 capacity[6] : 0 ≤ 20
 capacity[7] : 5 x1[1] ≤ 10
 capacity[8] : 5 x1[1] ≤ 20
 capacity[9] : 0 ≤ 10
 capacity[10] : 0 ≤ 10

julia> rmp.model[:convexity]
3-element Vector{ConstraintRef{Model, MathOptInterface.ConstraintIndex{MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.EqualTo{Float64}}, ScalarShape}}:
 convexity[1] : y1 + x1[1] = 1
 convexity[2] : y2 = 1
 convexity[3] : y3 = 1

julia> objective_function(rmp.model)
670 y1 + 670 y2 + 670 y3 + 40 x1[1]

```


"""
function add_column!(rmp::MCFRestrictedMasterProblem, k::Int64, column::Vector{Int64}, demand_amount::N, column_weight::N) where {N<:Number}
    # add variable for new column
    vref = rmp.model[Symbol("x$k")]
    vname = "x$k[$(size(vref,1))]"
    push!(vref, @variable(rmp.model, lower_bound=0, base_name=vname))
    # add column to list of columns
    push!(rmp.columns[k], column)
    # update the capacity constraints
    for a in column
        set_normalized_coefficient(
                                   rmp.model[:capacity][a],
                                   vref[end],
                                   demand_amount
                                  )
    end
    # update the convexity constraints
    set_normalized_coefficient(
                               rmp.model[:convexity][k],
                               vref[end],
                               1
                              )
    # update the objective
    set_objective_coefficient(rmp.model, vref[end], demand_amount * column_weight)
end

"""
    update_pricing_problem!(prp::MCFPricingProblme, capacity_duals::Vector{Float64}, convexity_duals::Vector{Float64}

Update the pricing problem data. Sets the values of the capacity and convexity constraint duals to the latest values given by solving the RMP.
"""
function update_pricing_problem!(prp::MCFPricingProblem, capacity_duals, convexity_duals)
    prp.convexity_duals .= convexity_duals
    prp.capacity_duals .= capacity_duals
end

"""
    solve!(prp::MCFPricingProblem)

Solve the pricing problem. For each demand `k` search for an ``s_k-t_k``-path on the graph with reduced edge costs ``c_a - \\sigma_a`` where ``\\sigma_a`` is the dual of the capacity constraint corresponding to edge ``a``. A path ``p`` found this way are either added to the master problem if they satisfy the negative reduced cost condition ``b_k * (c_p - \\sigma_a ) - \\tau_k < 0``.

`solve!(prp)` returns a list of columns which satisfy the previous condition.

# Example
```jldoctest; setup = :(pb = load("../instances/toytests/test1"); rmp=MCFRestrictedMasterProblem(pb) ; (obj,sigma,tau)=solve!(rmp))
julia> prp = MCFPricingProblem(pb);

julia> update_pricing_problem!(prp, sigma, tau);

julia> solve!(prp)
3-element Vector{Vector{VertexPath}}:
 [VertexPath{Int64}([1, 4, 6, 7])]
 [VertexPath{Int64}([2, 4, 6])]
 [VertexPath{Int64}([3, 4, 6, 7])]

```
"""
function solve!(prp::MCFPricingProblem)
    columns = [VertexPath[] for _ in 1:size(prp.convexity_duals,1)]
    # create the graph
    g = SimpleWeightedDiGraph(prp.srcnodes, prp.dstnodes, prp.costs .- prp.capacity_duals)
    for (k,demand) in enumerate(prp.demands)
        # solve shortest path problem
        ds = dijkstra_shortest_paths(g, demand.src)
        p = VertexPath(enumerate_paths(ds, demand.dst))
        if demand.amount * path_weight(p,g) - prp.convexity_duals[k] < -1e-8
            push!(columns[k], p)
        end
    end
    return columns
end

"""
    solution_from_rmp(rmp::MCFRestrictedMasterProblem, pb::MCF)

Extract solution to the MCF problem `pb` from the solution of the RMP.
"""
function solution_from_rmp(rmp::MCFRestrictedMasterProblem, pb::MCF)
    nK = length(rmp.columns)
    paths = [[] for _ in 1:nK]
    flows = [[] for _ in 1:nK]
    for k in 1:nK
        vars_k = rmp.model[Symbol("x$k")]
        vars_k_value = value.(vars_k)
        for cidx in findall(>(0), vars_k_value[2:end])
            push!(flows[k], vars_k_value[1+cidx])
            push!(paths[k], path_from_edge_indices(rmp.columns[k][cidx], pb.graph))
        end
    end
    return MCFSolution(paths, flows)
end

"""
    solve_column_generation(pb::MCF;
                                 max_unchanged::Int64=5,
                                 max_iterations::Int64=100,
                                 direct::Bool=true,
                                 return_rmp::Bool=false,
    )


Solve MCF problem using Column Generation. Returns a [`MCFSolution`](@ref) object and  a [`SolverStatistics`](@ref) containing statistics from the solver run. If `return_rmp=true` the second return value is a tuple where the first item is the `JuMP.Model` corresponding to the final RMP and the second item are the solver statistics.

| **Argument** | **Description** |
|---|---|
| max_unchanged | Maximum number of successive iterations without improvement of the objective function |
| max_iterations | Maximum number of CG iterations |
| direct | Use direct solver |
| rmp_solve_callback | Function called after each resolution of the RMP |

# Example
```jldoctest; setup = :(using JuMP ; pb = load("../instances/toytests/test1"))
julia> sol, (rmp, ss) = solve_column_generation(pb);

julia> value.(all_variables(rmp.model))
9-element Vector{Float64}:
  0.0
  0.0
  0.0
  0.0
  1.0
  1.0
  1.0
 -0.0
 -0.0

julia> ss.stats["objective_value"]
150.0

```

You may supply a callback function with the `rmp_solve_callback`. The supplied function must accept a single [`MCFRestrictedMasterProblem`](@ref) argument and will be executed `rmp_solve_callback(rmp)` at each iteration before calling `solve!` on the pricing problem.

```julia
julia> using Plots

julia> pb = load("../instances/toytests/test1");

julia> objvals = []
Any[]

julia> function rmpcallback(rmp)
           push!(objvals, JuMP.objective_value(rmp.model))
       end

julia> solve_column_generation(pb, rmp_solve_callback=rmpcallback);

julia> savefig(plot(1:size(objvals,1), objvals, xlabel="Time", ylabel="Objective"), "callback_toytest1_cg.png");

```

![](callback_toytest1_cg.png)


"""
function solve_column_generation(pb::MCF;
                                 max_unchanged::Int64=5,
                                 max_iterations::Int64=100,
                                 direct::Bool=true,
                                 rmp_solve_callback::Function=(rmp) -> (),
                                 return_rmp::Bool=false
    )
    best_rmpsol = 1e30
    tol = 1e-8
    unchanged = 0
    num_cg_iterations = 0
    # initialize master problem
    master_init_time::Float64 = @elapsed rmp = MCFRestrictedMasterProblem(pb, direct=true)
    # initialize pricing problem
    pricing_init_time::Float64 = @elapsed prp = MCFPricingProblem(pb)
    # master and pricing solve times
    master_solve_time::Float64, pricing_solve_time::Float64 = 0,0
    for i in 1:max_iterations
        # increment counters
        num_cg_iterations += 1
        # reset loop statistics
        n_added_columns = 0
        # solve the restricted master problem
        master_solve_time += @elapsed rmpsol, sigma, tau  = solve!(rmp)
        rmp_solve_callback(rmp)
        if rmpsol >= best_rmpsol - abs(best_rmpsol) * tol
            unchanged += 1
        else
            unchanged = 0

            best_rmpsol = rmpsol
        end

        # update pricing problem 
        update_pricing_problem!(prp, sigma, tau)
        # solve the pricing problem
        pricing_solve_time += @elapsed columns = solve!(prp)
        for k in 1:nk(pb)
            for column in columns[k]
                # add column to master problem
                add_column!(rmp, k, edge_indices(column, pb.graph), Float64.(pb.demands[k].amount), Float64.(path_weight(column, pb.graph)))
                n_added_columns += 1
            end
        end

        if n_added_columns==0 || unchanged>max_unchanged
            break
        end
    end

    stats = SolverStatistics()
    add_JuMP_statistics(stats, rmp.model)
    stats["master_init_time"] = master_init_time
    stats["pricing_init_time"] = pricing_init_time
    stats["master_solve_time"] = master_solve_time
    stats["pricing_solve_time"] = pricing_solve_time
    stats["num_cg_iterations"] = num_cg_iterations
    stats["num_columns"] = sum(size(columns,1) for columns in rmp.columns)

    return solution_from_rmp(rmp, pb), (rmp, stats)
end
