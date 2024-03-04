"""
    AugmentedGraph

Representation of an MCF instance as a directed graph. Demands are represented as vertices of the graph connected to the source and destination vertices. `node_mask,edge_mask` are boolean valued vectors respectively indicating the positions of the original graph vertices and edges. `.!node_mask, .!edge_mask` are used to retrieve vertices and edges corresponding to the demands.
"""
struct AugmentedGraph
    #graph::SimpleDiGraph
    edge_src::Vector{Int64}
    edge_dst::Vector{Int64}
    edge_features::Matrix{Float64}
    node_mask::Vector{Bool}
    edge_mask::Vector{Bool}
end


"""
    augmented_graph(pb::MCF)

Converts a [`MCF`](@ref) object to [`AugmentedGraph`](@ref).

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)]);

julia> g = augmented_graph(pb);

julia> pb.graph.srcnodes == g.edge_src[g.edge_mask]
true

julia> pb.graph.dstnodes == g.edge_dst[g.edge_mask]
true

julia> costs(pb) == g.edge_features[1,g.edge_mask]
true

julia> capacities(pb) == g.edge_features[2,g.edge_mask]
true

```
"""
function augmented_graph(pb::MCF)
    srcdemands,dstdemands = demand_endpoints(pb)
    demandamounts = demand_amounts(pb)
    # create the DemandGraph
    n,m,k = nv(pb),ne(pb),nk(pb)
    redge_features_size = (3, m)
    redge_features = zeros(Float64, redge_features_size)
    node_mask = zeros(Bool, n+k)
    node_mask[1:n] .= 1
    edge_mask = zeros(Bool, m+2*k)
    edge_mask[1:m] .= 1
    
    redge_features[1,:] = costs(pb)
    redge_features[2,:] = capacities(pb)
    #redge_features[3,:] = inst.latencies

    # demand nodes -> demand source
    demand_nodes = collect(1:k) .+ n
    dnto_src, dnto_dst = demand_nodes, srcdemands
    dnto_features = zeros(Float64, (3, k))
    #dnto_features[3,:] .= inst.demand_latencies
    dnto_features[3,:] .= demandamounts
    # demand target -> demand node
    dntt_src, dntt_dst = dstdemands, demand_nodes
    dntt_features = zeros(Float64, (3, k))
    #dntt_features[3,:] .= -inst.demand_latencies
    dntt_features[3,:] .= -demandamounts # TODO : no need for negative value

    edge_src = vcat(pb.graph.srcnodes, dnto_src, dntt_src)
    edge_dst = vcat(pb.graph.dstnodes, dnto_dst, dntt_dst)

    edge_features = hcat(redge_features, dnto_features, dntt_features)

    return AugmentedGraph(edge_src, edge_dst, edge_features, node_mask, edge_mask)
end

"""
    get_instance(g::AugmentedGraph)

Convert an [`AugmentedGraph`](@ref) object to the corresponding [`MCF`](@ref) object.

# Example
```jldoctest; setup = :(using Random, Graphs; Random.seed!(123))
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)])
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> g = augmented_graph(pb);

julia> pb1 = get_instance(g)
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> pb1 == pb
true
```
"""
function get_instance(ag::AugmentedGraph)
    edge_src, edge_dst = [],[]
    demand_src, demand_dst = [],[]
    costs, capacities = [],[]
    amounts = []
    amount_map = Dict() # keep dictionary of demand amounts

    nrealnodes = sum(ag.node_mask)
    nrealedges = sum(ag.edge_mask)
    ndemands = size(ag.node_mask,1) - nrealnodes

    srcnodes = ag.edge_src[1:nrealedges]
    dstnodes = ag.edge_dst[1:nrealedges]
    costs = ag.edge_features[1,1:nrealedges]
    capacities = ag.edge_features[2,1:nrealedges]
    latencies = ag.edge_features[3,1:nrealedges]

    demandsrc = ag.edge_dst[nrealedges+1:nrealedges+ndemands]
    demanddst = ag.edge_src[nrealedges+1+ndemands:end]
    demandamounts = ag.edge_features[3,nrealedges+1:nrealedges+ndemands]
    #demand_latencies = ag.edge_features[3,nrealedges+1:nrealedges+ndemands]
    # edges
    edge_list = Edge.([(s,t) for (s,t) in zip(srcnodes,dstnodes)])
    # demands
    demands = [Demand(s,t,a) for (s,t,a) in zip(demandsrc,demanddst,demandamounts)]
    return MCF(SimpleDiGraph(edge_list), costs, capacities, demands)
end


"""
    get_instance(g::GNNGraph)

Transforms a `GNNGraph` to a [`MCF`](@ref) instance. Works with batched `GNNGraph` objects.

# Example
```jldoctest; setup = :(using Graphs)
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)]);

julia> g = to_gnngraph(pb);

julia> pb1 = get_instance(g)
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> pb == pb1
true
```

If `g` is composed of multiple graphs : 
```jldoctest; setup = :(using Graphs, GraphNeuralNetworks; pb1 = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)]); pb2 = MCF(grid((3,3)), rand(12), rand(12), [Demand(6,9,2.)]); g = batch([to_gnngraph(pb1), to_gnngraph(pb2)]))
julia> g.num_graphs
2

julia> get_instance(g)
2-element Vector{MCF}:
 MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

 MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(6, 9, 2.0)

"""
function get_instance(g::GNNGraph)
    if g.num_graphs > 1
        return MCF[get_instance(bg) for bg in unbatch(g)]
    end
    s,t = edge_index(g)
    nK = sum(.!g.ndata.mask)
    demandsrc = t[ne(g)-2*nK+1:ne(g)-nK]
    demanddst = s[ne(g)-nK+1:end]
    demandamounts = Float64.(demand_amounts(g))
    demands = [Demand(ss,tt,a) for (ss,tt,a) in zip(demandsrc,demanddst,demandamounts)]

    edge_list = Edge.([(s,t) for (s,t) in zip(s[g.edata.mask],t[g.edata.mask])])
    csts = Float64.(g.e[1,g.edata.mask])
    caps = Float64.(g.e[2,g.edata.mask])
    return MCF(SimpleDiGraph(edge_list), csts, caps, demands)

end

"""
    add_stacked_index(g::GNNGraph)

Utility function that adds the `demand_stacked_idx, edge_stacked_idx` index vectors to the `GNNGraph`. Given an instance with demands ``K`` and edges ``A`` the classifier computes the scores ``s_a^k`` for each pair ``(a,k)``. Initial implementations of the model stacked the edge and demand vectors at each forward call of the model leading to CPU operations and slowing down computation at train time. This was especially true when batching multiple instances since we want to avoid computing scores for pairs ``(a_i, k_j)`` where ``i, j`` denote the indexes of two different instances in the batch. 

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb1 = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)]); g = to_gnngraph(pb); g = GNNGraph(g, gdata=(;K=g.K, E=g.E))
julia> g
GNNGraph:
  num_nodes: 10
  num_edges: 26
  ndata:
	mask = 10-element Vector{Bool}
  edata:
	e = 3×26 Matrix{Float64}
	demand_amounts_mask = 26-element BitVector
	mask = 26-element Vector{Bool}
	demand_to_source_mask = 26-element Vector{Bool}
	target_to_demand_mask = 26-element Vector{Bool}
  gdata:
	K = 1
	E = 24

julia> g = add_stacked_index(g)
GNNGraph:
  num_nodes: 10
  num_edges: 26
  ndata:
	mask = 10-element Vector{Bool}
  edata:
	e = 3×26 Matrix{Float64}
	demand_amounts_mask = 26-element BitVector
	mask = 26-element Vector{Bool}
	demand_to_source_mask = 26-element Vector{Bool}
	target_to_demand_mask = 26-element Vector{Bool}
  gdata:
	edge_stacked_idx = 24×1 Matrix{Int64}
	K = 1
	demand_stacked_idx = 24×1 Matrix{Int64}
	E = 24

julia> hcat(g.edge_stacked_idx, g.demand_stacked_idx)
24×2 Matrix{Int64}:
  1  1
  2  1
  3  1
  4  1
  5  1
  6  1
  7  1
  8  1
  9  1
 10  1
 11  1
 12  1
 13  1
 14  1
 15  1
 16  1
 17  1
 18  1
 19  1
 20  1
 21  1
 22  1
 23  1
 24  1

"""
function add_stacked_index(g::GNNGraph)
    # create the demand and edge stacking indexes
    egind = graph_indicator(g, edges=true)
    regind = egind[g.edata.mask]
    niedges = sum(g.edata.mask[egind .== 1]) # same number of edges in each graph
    ngind = graph_indicator(g)
    dgind = ngind[.!g.ndata.mask]
    dind = 1:size(dgind,1)
    reind = 1:size(regind,1)
    demand_stacked_idx = reduce(vcat,[repeat(dind[dgind .== i], inner=niedges) for i=1:g.num_graphs])
    edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], g.K[i]) for i=1:g.num_graphs])
    nstack = size(demand_stacked_idx,1)
    if nstack % g.num_graphs == 0
        _demand_stacked_idx = zeros(Int64, Int64(nstack / g.num_graphs), g.num_graphs)
        _demand_stacked_idx[:] .= demand_stacked_idx
        _edge_stacked_idx = zeros(Int64, Int64(nstack / g.num_graphs), g.num_graphs)
        _edge_stacked_idx[:] .= edge_stacked_idx
    else
        _demand_stacked_idx = zeros(Int64, 1 + trunc(Int64, nstack / g.num_graphs), g.num_graphs)
        _demand_stacked_idx[1:nstack] .= demand_stacked_idx
        _edge_stacked_idx = zeros(Int64, 1 + trunc(Int64, nstack / g.num_graphs), g.num_graphs)
        _edge_stacked_idx[1:nstack] .= edge_stacked_idx
    end

    GNNGraph(g, 
                 ndata=g.ndata, 
                 edata=g.edata, 
                 gdata=(;
                        g.gdata..., 
                        demand_stacked_idx=_demand_stacked_idx,
                        edge_stacked_idx=_edge_stacked_idx
                       )
                )
end




"""
    to_gnngraph(pb::MCF; feature_type::DataType=eltype(costs(pb)))

Convert a [`MCF`](@ref) instance to GNNGraph object.

# Example
```jldoctest; setup = :(using Random,Graphs,MultiFlows.ML; Random.seed!(123);)
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)])
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> to_gnngraph(pb)
GNNGraph:
  num_nodes: 10
  num_edges: 26
  ndata:
	mask = 10-element Vector{Bool}
  edata:
	e = 3×26 Matrix{Float64}
	demand_amounts_mask = 26-element BitVector
	mask = 26-element Vector{Bool}
	demand_to_source_mask = 26-element Vector{Bool}
	target_to_demand_mask = 26-element Vector{Bool}
  gdata:
	edge_stacked_idx = 24×1 Matrix{Int64}
	K = 1
	demand_stacked_idx = 24×1 Matrix{Int64}
	E = 24

```
"""
function to_gnngraph(pb::MCF; feature_type::DataType=eltype(costs(pb)))
    # start by creating the augmented graph
    ag = augmented_graph(pb)
    
    nK,nV,nE = nk(pb),nv(pb),ne(pb)
    demand_amounts_mask = .!ag.edge_mask
    demand_amounts_mask[findall(==(1), demand_amounts_mask)[nK+1:end]] .= 0

    # demand to source edge mask
    demand_to_source_mask = zeros(Bool, nE+2*nK)
    target_to_demand_mask = zeros(Bool, nE+2*nK)

    demand_to_source_mask[nE+1:nE+nK] .= 1
    target_to_demand_mask[nE+nK+1:end] .= 1

    g = GNNGraph(ag.edge_src, ag.edge_dst,
               ndata=(;mask=ag.node_mask),
               edata=(;e=feature_type.(ag.edge_features), 
		       mask=ag.edge_mask,
		       demand_amounts_mask=demand_amounts_mask,
		       demand_to_source_mask=demand_to_source_mask,
		       target_to_demand_mask=target_to_demand_mask
		     ),
               gdata=(;K=nK,
                       E=nE,
		      )
           )
    return add_stacked_index(g)
end


"""
    to_gnngraph(pb::MCF, y::AbstractMatrix; feature_type::DataType=eltype(costs(pb)))

Convert a UMFData instance to GNNGraph object with target labels.

# Example
```jldoctest; setup = :(using Random,Graphs,MultiFlows.ML; Random.seed!(123);)
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)]);

julia> y = rand(Bool, ne(pb), nk(pb));

julia> to_gnngraph(pb, y)
GNNGraph:
  num_nodes: 10
  num_edges: 26
  ndata:
	mask = 10-element Vector{Bool}
  edata:
	e = 3×26 Matrix{Float64}
	demand_amounts_mask = 26-element BitVector
	mask = 26-element Vector{Bool}
	demand_to_source_mask = 26-element Vector{Bool}
	target_to_demand_mask = 26-element Vector{Bool}
  gdata:
	edge_stacked_idx = 24×1 Matrix{Int64}
	targets = 24×1 Matrix{Bool}
	K = 1
	demand_stacked_idx = 24×1 Matrix{Int64}
	E = 24


```

"""
function to_gnngraph(pb::MCF, y::AbstractMatrix{Bool}; feature_type::DataType=eltype(costs(pb)))
    # create GNNGraph without labels
    gnn = to_gnngraph(pb, feature_type=feature_type)
    gnn = add_stacked_index(gnn)
    return GNNGraph(gnn, 
                    ndata=gnn.ndata, 
                    edata=gnn.edata, 
                    gdata=(;gnn.gdata..., targets=y)
                   )
end

"""
    MultiFlows.demand_endpoints(g::GNNGraph)

Get the demand endpoints (source and destination vertices) from a GNNGraph.

# Example
```jldoctest; setup = :(using Graphs,MultiFlows.ML)
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.), Demand(2,8,2.), Demand(3,7,3.)])
MCF(nv = 9, ne = 24, nk = 3)
	Demand{Int64, Float64}(1, 9, 1.0)
	Demand{Int64, Float64}(2, 8, 2.0)
	Demand{Int64, Float64}(3, 7, 3.0)

julia> gnn = to_gnngraph(pb);

julia> demand_endpoints(gnn)
([1, 2, 3], [9, 8, 7])

```

"""
function MultiFlows.demand_endpoints(g::GNNGraph)
    s,t = edge_index(g)

    # offset due to demand nodes
    gie = graph_indicator(g, edges=true)
    if g.num_graphs==1
        ds_t,dt_t = t[g.demand_to_source_mask], s[g.target_to_demand_mask]
    else
        offsets = cumsum(g.K) .- g.K
        ds_t,dt_t = (t .- offsets[gie])[g.demand_to_source_mask], (s .- offsets[gie])[g.target_to_demand_mask]
    end
    return ds_t,dt_t
end

"""
    MultiFlows.demand_amounts(g::GNNGraph)

Get the demand amounts from a GNNGraph.

# Example
```jldoctest; setup = :(using Graphs,MultiFlows.ML;pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.), Demand(2,8,2.), Demand(3,7,3.)]); gnn = to_gnngraph(pb))
julia> demand_amounts(gnn)
3-element Vector{Float64}:
 1.0
 2.0
 3.0

```

"""
function MultiFlows.demand_amounts(g::GNNGraph)
    return g.e[3, g.edata.demand_amounts_mask]
end


"""
    aggregate_demand_labels(g::GNNGraph)

Combine the labels for demands that share endpoints. If demands ``k_1, k_2`` are such that ``s_{k_1}=s_{k_2}`` and ``t_{k_1} = t_{k_2}`` then their respective labels ``l_1, l_2`` are set to ``l_1 = l_2 = l_1 \\& l_2``.

# Example
```jldoctest; setup = :(using Random,Graphs,MultiFlows.ML; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> y = rand(Bool, ne(pb), nk(pb))
8×3 Matrix{Bool}:
 1  0  0
 1  0  1
 1  1  0
 0  0  1
 1  1  1
 0  0  1
 1  1  1
 0  0  1

julia> gnn = aggregate_demand_labels(to_gnngraph(pb, y));

julia> gnn.targets
8×3×1 Array{Bool, 3}:
[:, :, 1] =
 1  1  0
 1  1  1
 1  1  0
 0  0  1
 1  1  1
 0  0  1
 1  1  1
 0  0  1

```

"""
function aggregate_demand_labels(g::GNNGraph)
    ndemands = sum(.!g.ndata.mask)
    ds,dt = demand_endpoints(g)
    new_labels = Dict()
    for k in 1:ndemands
        if haskey(new_labels, (ds[k], dt[k]))
            new_labels[(ds[k], dt[k])] .|= g.targets[:,k]
        else
            new_labels[(ds[k], dt[k])] = g.targets[:,k]
        end
    end
    for k in 1:ndemands
        g.targets[:,k] .= new_labels[(ds[k], dt[k])]
    end
    return g
end

"""
    AugmentedGNNGraph

A simple container for the GNNGraph representation of a MCF problem. Usefull for specializing the `batch` method for our use case.
"""
struct AugmentedGNNGraph
    g::GNNGraph
end

"""
    aggregate_demand_labels(ag::AugmentedGNNGraph)

Specialization of the [`aggregate_demand_labels`](@ref) function for `AugmentedGNNGraph` objects.

# Example
```jldoctest; setup = :(using Random,Graphs,MultiFlows.ML; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> y = rand(Bool, ne(pb), nk(pb));

julia> gnn = aggregate_demand_labels(AugmentedGNNGraph(to_gnngraph(pb, y)));

julia> gnn.g.targets
8×3×1 Array{Bool, 3}:
[:, :, 1] =
 1  1  0
 1  1  1
 1  1  0
 0  0  1
 1  1  1
 0  0  1
 1  1  1
 0  0  1

```

"""
function aggregate_demand_labels(ag::AugmentedGNNGraph)
    return AugmentedGNNGraph(aggregate_demand_labels(ag.g))
end

"""
    get_instance(ag::AugmentedGNNGraph)

Specialization of the [`get_instance`](@ref) function for `AugmentedGNNGraph` objects.

# Example
```jldoctest; setup = :(using Random, Graphs; Random.seed!(123))
julia> pb = MCF(grid((3,3)), rand(12), rand(12), [Demand(1,9,1.)])
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> g = AugmentedGNNGraph(to_gnngraph(pb));

julia> pb1 = get_instance(g)
MCF(nv = 9, ne = 24, nk = 1)
	Demand{Int64, Float64}(1, 9, 1.0)

julia> pb1 == pb
true
```

"""
function get_instance(ag::AugmentedGNNGraph)
    return get_instance(ag.g)
end

