"""
    random_demand_amounts(pb::MCF, n::Int64=nk(pb); factor::Float64=1.)

Select `n` demand amounts randomly by sampling values from ``\\{ b_k \\}_{k\\in K}``.

# Example
```jldoctest randdemands; setup = :(using Random; Random.seed!(123); pb = load("../instances/sndlib/AsnetAm_0_1_1"))
julia> unique(demand_amounts(pb))
4-element Vector{Float64}:
  500.0
   50.0
 1000.0
  100.0

julia> new_amounts = random_demand_amounts(pb, 100);

julia> unique(new_amounts)
4-element Vector{Float64}:
  500.0
 1000.0
  100.0
   50.0

```

The `factor` arguments lets one apply a scaling to the amounts : 
```jldoctest randdemands
julia> new_amounts = random_demand_amounts(pb, 100, factor=2.0);

julia> unique(new_amounts)
4-element Vector{Float64}:
 2000.0
  100.0
  200.0
 1000.0

```
"""
function random_demand_amounts(pb::MCF, n::Int64=nk(pb); factor::N=1) where {N<:Number}
    da = demand_amounts(pb)
    dtype = eltype(da)
    return dtype.(rand(da, n)) * factor
end

#"""
#    random_demand_latencies(inst::UMFData)
#random latencies values.
#"""
#function random_demand_latencies(inst::UMFData; factor=1., ndemands=numdemands(inst))
#    bdw_type = eltype(inst.demand_latencies)
#    bdw_mean = bdw_type(mean(inst.demand_latencies))
#    bdw_std = bdw_type(std(inst.demand_latencies))
#    bdw_min, bdw_max = bdw_type(minimum(inst.demand_latencies)), bdw_type(maximum(inst.demand_latencies))
#    dist = Normal(bdw_mean, bdw_std)
#    if bdw_std != 0
#        dist = truncated(dist, bdw_min, bdw_max)
#    end
#    return bdw_type.(rand(dist, ndemands)) * factor
#
#end

"""
    random_demand_endpoints(pb::MCF{T,N}, n::Int64=nk(pb);
                                 origins_destinations::Tuple{Vector{Int64}, Vector{Int64}}=demand_endpoints(pb),
                                 sample_f::Function=Base.rand,
    ) where {T,N}


Sample new demand endpoints ``s_k, t_k`` randomly from ``\\{ s_k, t_k \\}_{k\\in K}``. The set of origins and destinations from which new endpoints are sampled can be changed by passing a value to the `origins_destinations` arguments. Similarly `sample_f` is a sampling function, the default value samples new endpoints uniformly at random.

# Example
```jldoctest randdemands; setup = :(using Random; Random.seed!(123); pb = load("../instances/sndlib/AsnetAm_0_1_1"))
julia> random_demand_endpoints(pb, 5)
([29, 57, 21, 47, 64], [20, 9, 63, 54, 57])

julia> random_demand_endpoints(pb, 5, origins_destinations=(1:nv(pb), 1:nv(pb)))
([35, 55, 3, 31, 24], [7, 47, 38, 44, 20])

```

"""
function random_demand_endpoints(pb::MCF{T,N}, n::Int64=nk(pb);
                                 origins_destinations::Tuple{AbstractVector{T}, AbstractVector{T}}=demand_endpoints(pb),
                                 sample_f::Function=Base.rand,
    ) where {T,N}
    origins, destinations = origins_destinations
    new_origins = sample_f(origins, n)
    new_targets = sample_f(destinations, n)
    idx = (new_origins .== new_targets)
    while any(idx)
        new_targets[idx] = sample_f(destinations, sum(idx))
        idx = (new_origins .== new_targets)
    end
    return new_origins, new_targets
end

"""
    shake_instance(pb::MCF{T,N}; 
                   nK::Int64=nk(pb),
                   amount_factor::Float64=1., 
                   origins_destinations::Tuple{AbstractVector{T}, AbstractVector{T}}=demand_endpoints(pb),
                   sample_f::Function=Base.rand,
                  ) where {T,N}


Apply demand amount and endpoint perturbation to an MCF instance.

# Example
```jldoctest ; setup = :(using Graphs, Random, GraphPlot; Random.seed!(123); gr=grid((3,3)); (locx,locy)=spring_layout(gr))
julia> pb = MCF(grid((3,3)), ones(12), ones(12), [Demand(1,8,1.), Demand(7,5,.5)]);

julia> shake(pb)
MCF(nv = 9, ne = 24, nk = 2)
	Demand{Int64, Float64}(7, 8, 0.5)
	Demand{Int64, Float64}(7, 5, 0.5)

julia> shake(pb, origins_destinations=(1:nv(pb),1:nv(pb)))
MCF(nv = 9, ne = 24, nk = 2)
	Demand{Int64, Float64}(8, 5, 0.5)
	Demand{Int64, Float64}(6, 1, 0.5)

```

`pb`|`pb1`|`pb2`
:---:|:---:|:---
![](origin_grid_mcf.png)  |  ![](shook_grid_mcf_1.png) |  ![](shook_grid_mcf_2.png)


"""
function shake(pb::MCF{T,N}; 
                        nK::Int64=nk(pb),
                        amount_factor::Float64=1., 
                        #delay_factor=1.0, nK=nk(pb), 
                        origins_destinations::Tuple{AbstractVector{T}, AbstractVector{T}}=demand_endpoints(pb),
                        sample_f::Function=Base.rand,
    ) where {T,N}
    new_amounts = random_demand_amounts(pb, nK; factor=amount_factor)
    #new_lat = random_demand_latencies(inst; factor=latency_factor, ndemands=ndemands)
    new_demand_src, new_demand_trg = random_demand_endpoints(pb, nK, origins_destinations=origins_destinations, sample_f=sample_f)
    # convert features to match the type of amounts
    if eltype(new_amounts) != N
        g = convert_features(pb.graph, eltype(new_amounts))
        return MCF(g, [Demand(s,t,a) for (s,t,a) in zip(new_demand_src, new_demand_trg, new_amounts)])
    else
        return MCF(pb.graph, [Demand(s,t,a) for (s,t,a) in zip(new_demand_src, new_demand_trg, new_amounts)])
    end

end

"""
    non_saturated_path_exists(sol::MCFSolution, pb::MCF; tol::Float64=1e-8)

Check if the solution contains a path with capacity greater than the amount of flow circulating on it.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,2,1.)]);

julia> sol,_ = solve_column_generation(pb);

julia> sol
MCFSolution
	Demand k = 1
		1.0 on VertexPath{Int64}([1, 2])

julia> non_saturated_path_exists(sol, pb)
false

julia> sol.flows[1][1] = .5
0.5

julia> non_saturated_path_exists(sol, pb)
true

```

"""
function non_saturated_path_exists(sol::MCFSolution, pb::MCF; tol::Float64=1e-8)
    leftover_capacity = available_capacity(sol, pb)
    for k in eachindex(sol.paths)
        for p in sol.paths[k]
            if minimum(leftover_capacity[edge_indices(p, pb.graph)]) > tol
                return true
            end
        end
    end
    return false
end

"""
    saturate(pb::MCF{T,N}; 
             tol::Float64=1e-8, 
             max_iter::Int64=10, 
             solve_f::Function=solve_column_generation,
             demand_order_f::Function=identity,
             amount_delta_f::Function=identity
    ) where {T,N}

Saturate an MCF instance. Increase the amount of each demand until reaching the maximum capacity of the paths used to route them. `pb` is solved by calls to `solve_f` until the returned solution does not contain a path with zero minimum available capacity. The maximum number of times the solver is called is `max_iter`. Default behaviour is to take the demands in their original order and increase the amount by ``\\min\\limits_{a \\in p_k} \\text{available_capacity}(a)``, changing the `demand_order_f, amount_delta_f` functions lets users modify this scheme. Setting `demand_order_f=shuffle` will randomize the order in which the demands are increased, and `amount_delta_f=ak->rand()*ak` will increase the amount of demand `k` by a random number sampled from ``\\mathcal{U}([0, a_k])`` where ``a_k`` is the available capacity on the path.

This function return a `Tuple{MCF, MCFSolution}` object.

```jldoctest saturate; setup = :(using Graphs, Random, GraphPlot; Random.seed!(123); spring_layout(grid((3,3))); pb=MCF(grid((3,3)), ones(12), ones(12), [Demand(1,2,.5)]); pb=shake(pb, nK=5, origins_destinations=(1:nv(pb),1:nv(pb))))
julia> pb
MCF(nv = 9, ne = 24, nk = 5)
	Demand{Int64, Float64}(9, 5, 0.5)
	Demand{Int64, Float64}(8, 1, 0.5)
	Demand{Int64, Float64}(8, 6, 0.5)
	Demand{Int64, Float64}(8, 7, 0.5)
	Demand{Int64, Float64}(6, 9, 0.5)

julia> pb_sat, sol_sat = saturate(pb);

julia> pb_sat
MCF(nv = 9, ne = 24, nk = 5)
	Demand{Int64, Float64}(9, 5, 1.0)
	Demand{Int64, Float64}(8, 1, 0.5)
	Demand{Int64, Float64}(8, 6, 1.0)
	Demand{Int64, Float64}(8, 7, 0.5)
	Demand{Int64, Float64}(6, 9, 1.0)
```

`sol`|`sol` available capacities|`sol_sat` available capacities
:---:|:---:|:---
![](grid3x3_nonsat_solution.png)  |  ![](grid3x3_nonsat_capacities.png) |  ![](grid3x3_sat_capacities.png)


```jldoctest saturate
julia> sol,_ = solve_column_generation(pb);

julia> sol_sat,_ = solve_column_generation(pb_sat);

julia> non_saturated_path_exists(sol, pb)
true

julia> non_saturated_path_exists(sol_sat, pb_sat)
false
```

"""
function saturate(pb::MCF{T,N}; 
                  tol::Float64=1e-8, 
                  max_iter::Int64=10, 
                  solve_f::Function=solve_column_generation,
                  demand_order_f::Function=identity,
                  amount_delta_f::Function=identity
    ) where {T,N}
    sol = nothing
    for i in 1:max_iter
        # solve with column generation
        sol,_ = solve_column_generation(pb)
        if !non_saturated_path_exists(sol, pb)
            return pb, sol
        end
        # available capacity
        avail_cap = available_capacity(sol, pb)
        # increase amounts for each demand until reaching the maximum available capacity
        new_demands = Demand{T,N}[]
        for k in demand_order_f(1:nk(pb))
            dem = pb.demands[k]
            amount_delta = 0
            for p in sol.paths[k]
                # available capacity
                d = minimum(avail_cap[edge_indices(p,pb.graph)])
                amount_delta += d > tol ? amount_delta_f(d) : 0
                # update the available capacity
                avail_cap[edge_indices(p,pb.graph)] .-= d
            end
            push!(new_demands, Demand(dem.src, dem.dst, dem.amount+amount_delta))
        end
        pb = MCF(pb.graph, new_demands)
    end
    return pb,sol
end

"""
    generate_example(pb::MCF; 
                     demand_p::Float64=.05, 
                     amount_factor::Float64=1.05,
                     nK::Int64=nk(pb),
                     sample_f::Function=Base.rand,
                     solve_f::Function=solve_column_generation,
                     demand_order_f::Function=identity,
                     amount_delta_f::Function=identity

    )

Generate an example based on instance `pb`. Returns an instance that is the results of applying [`shake`](@ref), [`saturate`](@ref) and then increasing the amount of a randomly chosen set of demands by multiplying its amount by `amout_factor`.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123))
julia> pb = MCF(grid((3,2)), ones(7), ones(7), [Demand(1,2,1.)]);

julia> generate_example(pb)
MCF(nv = 6, ne = 14, nk = 1)
	Demand{Int64, Float64}(1, 2, 1.05)

julia> # change the number of demands

julia> generate_example(pb, nK=10)
MCF(nv = 6, ne = 14, nk = 10)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.05)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)
	Demand{Int64, Float64}(1, 2, 1.0)

```
"""
function generate_example(pb::MCF; 
                          demand_p::Float64=.05, 
                          amount_factor::Float64=1.05,
                          nK::Int64=nk(pb),
                          sample_f::Function=Base.rand,
                          solve_f::Function=solve_column_generation,
                          demand_order_f::Function=identity,
                          amount_delta_f::Function=identity

    )
    # shake
    pb_t = shake(pb, nK=nK, sample_f=sample_f)
    # saturate
    pb_s,sol_s = saturate(pb_t, solve_f=solve_f, demand_order_f=demand_order_f, amount_delta_f=amount_delta_f)
    # increase demand
    nselecteddemands = Int64(max(1, trunc(demand_p * nk(pb_s))))
    demand_idx = sample(1:nk(pb_s), nselecteddemands, replace=false)
    new_amounts = demand_amounts(pb_s)
    new_amounts[demand_idx] *= amount_factor
    new_demands = [Demand(d.src,d.dst,a) for (d,a) in zip(pb_s.demands,new_amounts)]
    return MCF(pb_s.graph, new_demands)
end

"""
    make_dataset(pb::MCF, n::Int64, path::String)

Create a dataset by applying perturbations to a reference instance. Saves resulting instance and solution files to path directory.
"""
function make_dataset(pb::MCF, n::Int64, path::String; 
                      perturbation_f::Function=generate_example, 
                      show_progress::Bool=true, 
                      labeling_f::Union{Nothing,Function}=nothing,
                      overwrite::Bool=true,
    )
    c = 0
    if isdir(path) && overwrite
        rm(path, recursive=true, force=true)
    end
    mkpath(path)
    if show_progress
        progress = ProgressBar(1:n)
    else
        progress = 1:n
    end
    for c in progress
        # save the instance and the solution
        instance_dir = joinpath(path, string(c))
        mkpath(instance_dir)

        # generate an instance
        npb = perturbation_f(pb)
        # save instance
        save(npb, instance_dir, verbose=false)

        # if a labeling function was provided it is called by passing the path of the saved instance
        if !isnothing(labeling_f)
            labeling_f(instance_dir)
        end
    end
end


