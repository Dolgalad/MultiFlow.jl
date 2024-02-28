"""
    make_batchable(gl::Vector{GNNGraph})

Make GNNGraphs batchable, pads the `targets` fields with zeros thus ensuring that those fields may be concatenanted along their last dimension. Adds the `target_mask` field so that `g.targets[g.target_mask]` returns the original target values.

# Example
```jldoctest makebatch; setup = :(using GraphNeuralNetworks, JuMP, Graphs)
julia> pb1 = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> _,(mod1,_) = solve_compact(pb1, max_acceptance=true, return_model=true);

julia> y1 = value.(mod1[:x]) .> 0 ;

julia> gnn1 = to_gnngraph(pb1, y1)
GNNGraph:
  num_nodes: 7
  num_edges: 14
  ndata:
	mask = 7-element Vector{Bool}
  edata:
	e = 3×14 Matrix{Float64}
	demand_amounts_mask = 14-element BitVector
	mask = 14-element Vector{Bool}
	demand_to_source_mask = 14-element Vector{Bool}
	target_to_demand_mask = 14-element Vector{Bool}
  gdata:
	targets = 8×3×1 BitArray{3}
	K = 3
	E = 8

julia> pb2 = MCF(grid((2,3)), rand(7), rand(7), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> _,(mod2,_) = solve_compact(pb2, max_acceptance=true, return_model=true);

julia> y2 = value.(mod2[:x]) .> 0 ;

julia> gnn2 = to_gnngraph(pb2, y2)
GNNGraph:
  num_nodes: 9
  num_edges: 20
  ndata:
	mask = 9-element Vector{Bool}
  edata:
	e = 3×20 Matrix{Float64}
	demand_amounts_mask = 20-element BitVector
	mask = 20-element Vector{Bool}
	demand_to_source_mask = 20-element Vector{Bool}
	target_to_demand_mask = 20-element Vector{Bool}
  gdata:
	targets = 14×3×1 BitArray{3}
	K = 3
	E = 14

```

If we try to create a batch of `gnn1` and `gnn2` a `DimensionMismatch` error is raised : 
```jldoctest makebatch
julia> batch([gnn1, gnn2])
ERROR: DimensionMismatch("mismatch in dimension 1 (expected 8 got 14)")
[...]
```

```jldoctest makebatch
julia> gl = make_batchable(GNNGraph[gnn1,gnn2])
2-element Vector{GNNGraph}:
 GNNGraph(7, 14) with mask: 7-element, (e: 3×14, demand_amounts_mask: 14-element, mask: 14-element, demand_to_source_mask: 14-element, target_to_demand_mask: 14-element), (targets: 14×3×1, K: 0-dimensional, target_mask: 14×3×1, E: 0-dimensional) data
 GNNGraph(9, 20) with mask: 9-element, (e: 3×20, demand_amounts_mask: 20-element, mask: 20-element, demand_to_source_mask: 20-element, target_to_demand_mask: 20-element), (targets: 14×3×1, K: 0-dimensional, target_mask: 14×3×1, E: 0-dimensional) data

julia> batch_g = batch(gl)
GNNGraph:
  num_nodes: 16
  num_edges: 34
  num_graphs: 2
  ndata:
	mask = 16-element Vector{Bool}
  edata:
	e = 3×34 Matrix{Float64}
	demand_amounts_mask = 34-element BitVector
	mask = 34-element Vector{Bool}
	demand_to_source_mask = 34-element Vector{Bool}
	target_to_demand_mask = 34-element Vector{Bool}
  gdata:
	targets = 14×3×2 Array{Bool, 3}
	K = 2-element Vector{Int64}
	target_mask = 14×3×2 Array{Bool, 3}
	E = 2-element Vector{Int64}
```

Retrieving the labels 
```jldoctest makebatch
julia> gl[1].targets[gl[1].target_mask] == vec(y1)
true

julia> gl[2].targets[gl[2].target_mask] == vec(y2)
true

julia> batch_g.targets[batch_g.target_mask] == vcat(vec(y1), vec(y2))
true
```
"""
function make_batchable(gl::Vector{GNNGraph})
    max_k = maximum(g.K for g in gl)
    max_ne = maximum(size(g.targets,1) for g in gl)
    new_graphs = GNNGraph[]
    for g in gl
        nedges = size(g.targets,1)
        new_targets = zeros(Bool, (max_ne, max_k))
        new_targets[1:nedges, 1:size(g.targets, 2)] .= g.targets
        target_mask = zeros(Bool, (max_ne, max_k))
        target_mask[1:nedges, 1:size(g.targets, 2)] .= 1
        ng = GNNGraph(g, gdata=(;K=g.K, E=g.E, targets=new_targets, target_mask=target_mask))
        push!(new_graphs, ng)
    end
    new_graphs
end
"""
    load_dataset(dataset_dir::String; 
                      scale_instances::Bool=true, 
                      batchable::Bool=true, 
                      edge_dir::Symbol=:double,
                      feature_type::DataType=Float32,
                      show_progress::Bool=false
    )

Load dataset from directory. If instance directories contain a `labels.jld2` file it is added to the `targets` field of the GNNGraph object. If `scale_instances=true` instances are scaled by calling [`scale`](@ref) before being added, if `batchable=true` a call to [`make_batchable`](@ref) is executed on the list of graphs. 

# Example
Create a simple dataset : 
```jldoctest loaddataset; setup = :(using Graphs, Random, JLD2, JuMP, GraphNeuralNetworks; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> # labeling function

julia> function labeling_f(path::String)
           pb = MultiFlows.load(path, edge_dir=:double)
           _,(model,ss) = solve_compact(pb, return_model=true, max_acceptance=true)
           labels = value.(model[:x]) .> 0
           labels_path = joinpath(path, "labels.jld2")
           jldsave(labels_path; labels)
       end
labeling_f (generic function with 1 method)

julia> # perturbation function

julia> function perturbation_f(pb::MCF)
           generate_example(pb, nK=nk(pb)+rand([-1,0,1]))
       end
perturbation_f (generic function with 1 method)

julia> make_dataset(pb, 5, "small_dataset", 
                    overwrite=true,
                    perturbation_f=perturbation_f,
                    labeling_f=labeling_f,
                    show_progress=false
                   )

```

Now load the instances. By default `load_dataset` ensures that the graph features are batchable.
```jldoctest loaddataset
julia> d = load_dataset("small_dataset")
5-element Vector{GNNGraph}:
 GNNGraph(7, 14) with mask: 7-element, (e: 3×14, demand_amounts_mask: 14-element, mask: 14-element, demand_to_source_mask: 14-element, target_to_demand_mask: 14-element), (targets: 8×4×1, K: 0-dimensional, target_mask: 8×4×1, E: 0-dimensional) data
 GNNGraph(6, 12) with mask: 6-element, (e: 3×12, demand_amounts_mask: 12-element, mask: 12-element, demand_to_source_mask: 12-element, target_to_demand_mask: 12-element), (targets: 8×4×1, K: 0-dimensional, target_mask: 8×4×1, E: 0-dimensional) data
 GNNGraph(6, 12) with mask: 6-element, (e: 3×12, demand_amounts_mask: 12-element, mask: 12-element, demand_to_source_mask: 12-element, target_to_demand_mask: 12-element), (targets: 8×4×1, K: 0-dimensional, target_mask: 8×4×1, E: 0-dimensional) data
 GNNGraph(6, 12) with mask: 6-element, (e: 3×12, demand_amounts_mask: 12-element, mask: 12-element, demand_to_source_mask: 12-element, target_to_demand_mask: 12-element), (targets: 8×4×1, K: 0-dimensional, target_mask: 8×4×1, E: 0-dimensional) data
 GNNGraph(8, 16) with mask: 8-element, (e: 3×16, demand_amounts_mask: 16-element, mask: 16-element, demand_to_source_mask: 16-element, target_to_demand_mask: 16-element), (targets: 8×4×1, K: 0-dimensional, target_mask: 8×4×1, E: 0-dimensional) data

julia> batch(d)
GNNGraph:
  num_nodes: 33
  num_edges: 66
  num_graphs: 5
  ndata:
	mask = 33-element Vector{Bool}
  edata:
	e = 3×66 Matrix{Float32}
	demand_amounts_mask = 66-element BitVector
	mask = 66-element Vector{Bool}
	demand_to_source_mask = 66-element Vector{Bool}
	target_to_demand_mask = 66-element Vector{Bool}
  gdata:
	targets = 8×4×5 Array{Bool, 3}
	K = 5-element Vector{Int64}
	target_mask = 8×4×5 Array{Bool, 3}
	E = 5-element Vector{Int64}

```
"""
function load_dataset(dataset_dir::String; 
                      scale_instances::Bool=true, 
                      batchable::Bool=true, 
                      edge_dir::Symbol=:double,
                      feature_type::DataType=Float32,
                      show_progress::Bool=false
    )
    graphs = GNNGraph[]
    if show_progress
        bar = ProgressBar(readdir(dataset_dir, join=true))
        set_description(bar, "Loading data $(dataset_dir)")
    else
        bar = readdir(dataset_dir, join=true)
    end
    for f in bar
        # must be a directory containing a link.csv and service.csv 
        if !is_instance_dir(f)
            continue
        end

        inst = MultiFlows.load(f, edge_dir=edge_dir)
        #println([nv(inst), ne(inst), nk(inst)])
        # scale cost and capacities
        if scale_instances
            inst = scale(inst)
        end
        # check if a solution file exists
        solution_file_path = joinpath(f, "labels.jld2")
        if isfile(solution_file_path)
            y = JLD2.load(solution_file_path, "labels")
            g = to_gnngraph(inst, y, feature_type=feature_type)
        else
            g = to_gnngraph(inst, feature_type=feature_type)
        end
        
        push!(graphs, g)
    end
    if batchable
        graphs = make_batchable(graphs)
    end
    return graphs
end

#function load_solved_dataset(dataset_dir, solution_dir; scale_instances=true, batchable=true, edge_dir=:double, solver_type="CplexAll")
#    dataset_name = split(dataset_dir, "/")[end-2]
#    println("dataset name : ", dataset_name)
#    graphs = []
#    bar = ProgressBar(filter(f->contains(f,dataset_name) && contains(f,solver_type), readdir(solution_dir)))
#    set_description(bar, "Loading from $(dataset_dir)")
#    # got over solution files
#    for f in bar
#        if contains(f, dataset_name) && contains(f, solver_type)
#            # number
#            d = split(f, "_")
#            instance_number = replace(d[end],solver_type*".csv"=>"")
#            # load instance
#            inst = UMFData(joinpath(dataset_dir, instance_number), edge_dir=edge_dir)
#            # scale cost and capacities
#            if scale_instances
#                inst = scale(inst)
#            end
#            # load solution from csv file
#            try
#                y = load_csv_solution(joinpath(solution_dir, f), inst)
#
#                g = UMFSolver.to_gnngraph(inst, y, feature_type=Float32)
#                
#                push!(graphs, g)
#            catch
#            end
#
#        end
#    end
#    if batchable
#        graphs = make_batchable(graphs)
#    end
#    return graphs
#
#end
#
#"""
#Load single instance
#"""
#function load_instance(inst_path; scale_instance=true)
#    inst = UMFData(inst_path)
#    # scale cost and capacities
#    if scale_instance
#        inst = UMFSolver.scale(inst)
#    end
#    # check if a solution file exists
#    solution_file_path = joinpath(inst_path, "sol.jld")
#    if isfile(solution_file_path)
#        sol = UMFSolver.load_solution(solution_file_path)
#    else
#        ssol, stats = solveUMF(inst_path, "CG", "highs", "./output.txt")
#        sol = ssol.x
#    end
#    y = (sol .> 0)
#    return g = UMFSolver.to_gnngraph(inst, y, feature_type=Float32)
#end


