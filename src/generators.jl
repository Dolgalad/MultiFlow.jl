"""
    random_demand_bandwidths(pb::MCF, n::Int64=nk(pb); factor::Float64=1.)

Select `n` demand amounts randomly by sampling values from ``\\{ b_k \\}_{k\\in K}``.

# Example
```jldoctest randdemands; setup = :(using Random; Random.seed!(123); pb = load("../instances/sndlib/AsnetAm_0_1_1", capacity_fieldname="bandwidth", demand_amount_fieldname="bandwidth"))
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
function random_demand_amounts(pb::MCF, n::Int64=nk(pb); factor::Float64=1.)
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
```jldoctest randdemands; setup = :(using Random; Random.seed!(123); pb = load("../instances/sndlib/AsnetAm_0_1_1", capacity_fieldname="bandwidth", demand_amount_fieldname="bandwidth"))
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
    return MCF(pb.graph, [Demand(s,t,a) for (s,t,a) in zip(new_demand_src, new_demand_trg, new_amounts)])
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
#
#function saturate_instance(inst::UMFData; tol=1e-3)
#    # solve with column generation
#    (sol, ss) = solveUMF(inst, "CG", "cplex", "./output.txt")
#    ms = ss.stats["ms"]
#    # all columns
#    columns = UMFSolver.allcolumns(ms)
#    # fractional flow per column
#    flows = UMFSolver.getx(ms)
#    # available capacities 
#    available_caps = available_capacities(inst, ms)
#    # new bandwidths
#    new_bdw = copy(inst.bandwidths)
#    # keep list of demands 
#    demand_paths = []
#    for k in eachindex(columns)
#        for p in eachindex(columns[k])
#            push!(demand_paths, (k,p))
#        end
#    end
#    demand_paths = Set(demand_paths)
#
#    #while non_saturated_path_exists(available_caps, columns, tol=tol)
#    while !isempty(demand_paths)
#        (k,p) = rand(demand_paths)
#        if size(columns[k],1)==0
#            println("Poping ", (k,p))
#            pop!(demand_paths, (k,p))
#            continue
#        end
#        #p = rand(1:size(columns[k], 1))
#        # check if path capacity if greater than 0
#        s = path_capacity(columns[k][p], available_caps)
#        if s > tol
#            if s < 1
#                bdw_delta = s
#            else
#                dist = Uniform(0, s)
#                bdw_delta = rand(dist)
#            end
#            new_bdw[k] += bdw_delta
#
#            # update available capacities
#            for a in columns[k][p]
#                available_caps[a] -= bdw_delta
#            end
#        end
#        # pop (k,p) if paths are saturated
#        if path_capacity(columns[k][p], available_caps) < tol
#            #println("poping : ", (k,p))
#            pop!(demand_paths, (k,p))
#        end
#    end
#    return UMFData(
#        inst.name * "_saturated",
#        inst.srcnodes,
#        inst.dstnodes,
#        inst.capacities,
#        inst.costs,
#        inst.latencies,
#        inst.srcdemands,
#        inst.dstdemands,
#        new_bdw,
#        inst.demand_latencies,
#        numdemands(inst),
#        numnodes(inst),
#        numarcs(inst),
#    )
#end
#
#function generate_example(inst::UMFData; demand_p=.05, bdw_factor=1.05)
#    ndemands = numdemands(inst)
#    # shake
#    inst_t = shake_instance(inst)
#    # saturate
#    inst_s = saturate_instance(inst_t)
#
#    # increase demand
#    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
#    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
#    new_bdw = copy(inst_s.bandwidths)
#    new_bdw[demand_idx] *= bdw_factor
#    return UMFData(
#        inst_s.name * "_generated",
#        inst_s.srcnodes,
#        inst_s.dstnodes,
#        inst_s.capacities,
#        inst_s.costs,
#        inst_s.latencies,
#        inst_s.srcdemands,
#        inst_s.dstdemands,
#        new_bdw,
#        inst_s.demand_latencies,
#        numdemands(inst),
#        numnodes(inst),
#        numarcs(inst),
#    )
#
#end
#
#"""Generates examples, changes the number of demands
#"""
#function generate_example_2(inst::UMFData; demand_p=.05, bdw_factor=1.05, demand_delta_p=0.1)
#    delta_range = range(floor(Int64, -demand_delta_p * nk(inst)), floor(Int64, demand_delta_p*nk(inst)))
#    ndemands = numdemands(inst) + rand(delta_range)
#    # shake
#    inst_t = shake_instance(inst, ndemands=ndemands)
#    # saturate
#    inst_s = saturate_instance(inst_t)
#
#    # increase demand
#    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
#    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
#    new_bdw = copy(inst_s.bandwidths)
#    new_bdw[demand_idx] *= bdw_factor
#    return UMFData(
#        inst_s.name * "_generated",
#        inst_s.srcnodes,
#        inst_s.dstnodes,
#        inst_s.capacities,
#        inst_s.costs,
#        inst_s.latencies,
#        inst_s.srcdemands,
#        inst_s.dstdemands,
#        new_bdw,
#        inst_s.demand_latencies,
#        numdemands(inst_t),
#        numnodes(inst_t),
#        numarcs(inst_t),
#    )
#
#end
#
#"""Generates examples, changes the number of demands
#"""
#function generate_example_3(inst::UMFData; demand_p=.05, bdw_factor=1.05, demand_delta_p=0.1)
#    delta_range = range(floor(Int64, -demand_delta_p * nk(inst)), floor(Int64, demand_delta_p*nk(inst)))
#    ndemands = numdemands(inst) + rand(delta_range)
#    # shake
#    inst_t = shake_instance_2(inst, ndemands=ndemands)
#    # saturate
#    inst_s = saturate_instance(inst_t)
#
#    # increase demand
#    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
#    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
#    new_bdw = copy(inst_s.bandwidths)
#    new_bdw[demand_idx] *= bdw_factor
#    return UMFData(
#        inst_s.name * "_generated",
#        inst_s.srcnodes,
#        inst_s.dstnodes,
#        inst_s.capacities,
#        inst_s.costs,
#        inst_s.latencies,
#        inst_s.srcdemands,
#        inst_s.dstdemands,
#        new_bdw,
#        inst_s.demand_latencies,
#        numdemands(inst_t),
#        numnodes(inst_t),
#        numarcs(inst_t),
#    )
#
#end
#
#"""Generates examples, changes the number of demands
#"""
#function generate_example_4(inst::UMFData; demand_p=.05, bdw_factor=1.05, ndemands=numdemands(inst))
#    # shake
#    inst_t = shake_instance_2(inst, ndemands=ndemands)
#    # saturate
#    inst_s = saturate_instance(inst_t)
#
#    # increase demand
#    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
#    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
#    new_bdw = copy(inst_s.bandwidths)
#    new_bdw[demand_idx] *= bdw_factor
#    return UMFData(
#        inst_s.name * "_generated",
#        inst_s.srcnodes,
#        inst_s.dstnodes,
#        inst_s.capacities,
#        inst_s.costs,
#        inst_s.latencies,
#        inst_s.srcdemands,
#        inst_s.dstdemands,
#        new_bdw,
#        inst_s.demand_latencies,
#        numdemands(inst_t),
#        numnodes(inst_t),
#        numarcs(inst_t),
#    )
#
#end
#
#
#function generate_example_with_aggregated_demands(inst::UMFData; demand_p=.05, bdw_factor=1.05)
#    # shake
#    inst_t = shake_instance_with_aggregated_demands(inst)
#    # saturate
#    inst_s = saturate_instance(inst_t)
#
#    # increase demand
#    ndemands = numdemands(inst_s)
#    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
#    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
#    new_bdw = copy(inst_s.bandwidths)
#    new_bdw[demand_idx] *= bdw_factor
#    return UMFData(
#        inst_s.name * "_generated",
#        inst_s.srcnodes,
#        inst_s.dstnodes,
#        inst_s.capacities,
#        inst_s.costs,
#        inst_s.srcdemands,
#        inst_s.dstdemands,
#        new_bdw,
#        numdemands(inst_s),
#        numnodes(inst_s),
#        numarcs(inst_s),
#    )
#
#end
#
#
