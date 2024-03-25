"""
Current best version of the arc-demand classifier model.
"""

abstract type AbstractClassifierModel end

"""
    M8MPLayer

Message passing layer. Based on _GraphNeuralNetworks.jl_ `MEGNetConv` layer with vertex and edge feature normalization and dropout layers. Messages are passed forward through the edges, backward and the results averaged.
"""
struct M8MPLayer
    forward_conv::MEGNetConv
    drop_n::Dropout
    drop_e::Dropout
    bn_n::BatchNorm
    bn_e::BatchNorm
    rev::Union{Function,M3EdgeReverseLayer}
    backward_conv::MEGNetConv
    _n::Int64
end

Flux.@functor M8MPLayer

"""
    M8MPLayer(n::Int64; drop_p::Float64=.1)

`M8MPLayer` constructor. This layer has the same input and output dimension `n`.

# Example
```jldoctest
julia> m = M8MPLayer(2)
M8MPLayer(GraphNeuralNetworks.MEGNetConv{Flux.Chain{Vector{Any}}, Flux.Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), Dropout(0.1), Dropout(0.1), BatchNorm(2), BatchNorm(2), M3EdgeReverseLayer(), GraphNeuralNetworks.MEGNetConv{Flux.Chain{Vector{Any}}, Flux.Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), 2)
```
"""
function M8MPLayer(n::Int64; drop_p::Float64=.1, reverse::Bool=true)
    return M8MPLayer(
                     MyMEGNetConv3(n), 
		     Flux.Dropout(drop_p), 
		     Flux.Dropout(drop_p),
		     Flux.BatchNorm(n), 
		     Flux.BatchNorm(n), 
		     reverse ? M3EdgeReverseLayer() : identity, 
		     MyMEGNetConv3(n),
                     n
                    )
end

"""
    (l::M8MPLayer)(g, x, e)

`M8MPLayer` forward call.

# Example
```jldoctest; setup = :(using GraphNeuralNetworks, Random ; Random.seed!(123))
julia> g = GNNGraph([1,2,3], [2,3,1]);

julia> x, e = rand(Float32, 4, 3), rand(Float32, 5, 3);

julia> m = M8MPLayer(2);

julia> x, e = m(g, x, e)
(Float32[0.048501432 0.011552513 0.4102404; 0.52121377 0.8908787 0.52566236; … ; -0.019039169 -0.003820137 -0.024450637; 0.008280508 0.0007322127 0.014470182], Float32[0.82208836 0.58055997 0.8296713; 0.044817984 0.4181347 0.83622855; … ; 0.26154602 0.20365497 -0.019983638; -0.28453535 -0.042258546 -0.058157448])

julia> size(x), size(e)
((6, 3), (7, 3))

```
"""
function (l::M8MPLayer)(g, x, e)
    # forward message passing
    xf,ef = l.forward_conv(g, x[end-l._n+1:end,:], e[end-l._n+1:end,:])
    xf = l.drop_n(xf)
    xf = l.bn_n(xf)
    ef = l.drop_e(ef)
    ef = l.bn_e(ef)
    # backward message passing
    ng = l.rev(g)
    xb,eb = l.backward_conv(ng, x[end-l._n+1:end,:], e[end-l._n+1:end,:])
    xb = l.drop_n(xb)
    xb = l.bn_n(xb)
    eb = l.drop_e(eb)
    eb = l.bn_e(eb)

    on, oe = vcat(x, 0.5f0 .* (xf+xb)), vcat(e, 0.5f0 * (ef+eb))
 
    return on,oe
end

"""
    M8ClassifierModel

Classifier model.
"""
struct M8ClassifierModel <: AbstractClassifierModel
    node_embeddings::Flux.Embedding
    edge_encoder::Flux.Chain
    demand_encoder::Flux.Chain
    graph_conv::Vector{M8MPLayer}
    demand_mlp::Flux.Chain
    edge_mlp::Flux.Chain
    scoring::Flux.Bilinear
    _device::Function
end

Flux.@functor M8ClassifierModel

"""
    M8ClassifierModel(node_feature_dim::Int64, 
                      edge_feature_dim::Int64, 
                      n_layers::Int64, 
                      nnodes::Int64; 
                      drop_p::Float64=0.1,
                      device::Function=CUDA.functional() ? Flux.gpu : Flux.cpu,
    )

`M8ClassifierModel` constructor with node embedding dimension `node_feature_dim`, input edge features `edge_feature_dim` (value is 3 for basic MCF instances), number of message passing layers `n_layers` and number of nodes in the graph `nnodes`. `drop_p` is the dropout probability used in the message passing layers. 

Current implementation of the edge-demand scores requires passing the device (GPU, CPU) to the constructor. If a GPU is available it is used.

This model design works on datasets constructed from a single graph. It relies on a node embedding dictionary and thus we need to know in advance the number of nodes in the graph. 
"""
function M8ClassifierModel(node_feature_dim::Int64, 
			   edge_feature_dim::Int64, 
			   n_layers::Int64, 
			   nnodes::Int64; 
                           drop_p::Float64=0.1,
			   device::Function=CUDA.functional() ? Flux.gpu : Flux.cpu,
                           reverse::Bool=true
			   )
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = MLP(edge_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    demand_encoder = MLP(2*node_feature_dim+1, node_feature_dim, node_feature_dim, drop_p=drop_p)
    graph_conv      = [M8MPLayer(node_feature_dim, drop_p=drop_p, reverse=reverse) for _ in 1:n_layers]

    demand_mlp = MLP(3*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)
    edge_mlp = MLP(2*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((node_feature_dim, node_feature_dim) => 1)

    M8ClassifierModel(node_embeddings, edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring, device)
end

"""
    compute_graph_embeddings(model::M8ClassifierModel, g::GNNGraph)

Compute the graph embeddings, returns a matrix for the vertex embeddings and another for the edge embeddings.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)])
MCF(nv = 4, ne = 8, nk = 3)
	Demand{Int64, Float64}(1, 4, 1.0)
	Demand{Int64, Float64}(1, 4, 1.0)
	Demand{Int64, Float64}(3, 2, 1.0)

julia> g = to_gnngraph(pb, feature_type=Float32);

julia> model = M8ClassifierModel(4, 3, 2, nv(pb));

julia> node_codes, edge_codes = compute_graph_embeddings(model, g)
(Float32[-1.3416092 -0.07616795 … -0.10367019 -0.204979; 0.41216165 -0.65490055 … -0.018500613 -0.036579825; … ; -0.09281337 -0.040796205 … -0.016434366 -0.009337337; -0.018684518 -0.018848153 … 0.0015017446 0.009444744], Float32[-0.00523452 -0.028606605 … -0.1676333 -0.1676333; 0.0027749126 0.0024765532 … 0.10864657 0.10864657; … ; 0.022817744 0.02496451 … 0.020531857 0.0015449864; 0.067348234 0.079991825 … 0.02253621 0.007676319])

julia> size(node_codes), size(edge_codes)
((12, 7), (12, 14))

julia> size(node_codes,2) == nv(pb)+nk(pb)
true

julia> size(edge_codes,2) == ne(pb)+2*nk(pb)
true

```
"""
function compute_graph_embeddings(model::M8ClassifierModel, g::GNNGraph)
    if g.num_graphs==1
        nnodes = sum(g.ndata.mask)
        ndemands = g.num_nodes - nnodes
    else
        nnodes = sum(g.ndata.mask .& (graph_indicator(g).==1))
        ndemands = sum((.!g.ndata.mask) .& (graph_indicator(g).==1))
    end

    # first encode the edge features
    edge_features = model.edge_encoder(g.e)

    # dimension of node embeddings
    node_feature_dim = size(edge_features, 1)
  
    # stack node embeddings and demands
    # batch size support
    node_embedding_idx = repeat(1:nnodes, g.num_graphs)

    # initial node embeddings
    node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    allK = sum(g.K)

    full_amounts = demand_amounts(g)
    all_amounts = reshape(full_amounts, 1, allK)
    dcodes = make_demand_codes(node_embeddings_0, g)
    demand_embeddings_0 = reduce(vcat, [dcodes , all_amounts])
    demand_embeddings_0 = model.demand_encoder(demand_embeddings_0)

    full_node_embeddings = hcat(node_embeddings_0, demand_embeddings_0)

    # apply the graph convolution
    encoded_nodes,_encoded_edges = full_node_embeddings, edge_features
    #encoded_nodes,_encoded_edges = model.graph_conv[1](g, full_node_embeddings, edge_features)
    if length(model.graph_conv)>0
        for gnnlayer in model.graph_conv
            encoded_nodes, _encoded_edges = gnnlayer(g, encoded_nodes, _encoded_edges)
        end
    end

    #println("types : ", [typeof(encoded_nodes), typeof(_encoded_edges)])

    return encoded_nodes, _encoded_edges

end

"""
    make_demand_codes(model::M8ClassifierModel, g::GNNGraph)

Make the demand codes.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); g = to_gnngraph(pb, feature_type=Float32); model = M8ClassifierModel(4,3,2,nv(pb)))
julia> demand_codes = make_demand_codes(model, g);

julia> size(demand_codes)
(36, 3)

julia> size(demand_codes,2) == nk(pb)
true

```

"""
function make_demand_codes(model::M8ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

"""
    (model::M8ClassifierModel)(g::GNNGraph)

[`M8ClassifierModel`](@ref) forward call. Returns the flattened edge-demand score matrix.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); g = to_gnngraph(pb, feature_type=Float32); model = M8ClassifierModel(4,3,2,nv(pb)))
julia> y = model(g);

julia> size(y)
(1, 24)

julia> size(y,2) == ne(pb) * nk(pb)
true

```

"""
function (model::M8ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)

    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)

    ndemands = g.num_nodes - nnodes

    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(concat_nodes, g, encoded_nodes, encoded_nodes)

    # separate the encoded demands
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
  
    # compute scores
    scores = compute_edge_demand_scores(model, 
                                                  model.edge_mlp(encoded_edges[:,g.edata.mask]), 
                                                  model.demand_mlp(encoded_demands),
                                                  g)
    return scores
end

"""
    (model::M8ClassifierModel)(pb::MCF)

[`M8ClassifierModel`](@ref) forward call for MCF problem `pb`. Returns a `(ne(pb), nk(pb))` matrix.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); model = M8ClassifierModel(4,3,2,nv(pb)))
julia> y = model(pb);

julia> size(y)
(8, 3)

julia> size(y,1) == ne(pb)
true

julia> size(y,2) == nk(pb)
true

```

"""
function (model::M8ClassifierModel)(pb::MCF)
    gnn = to_gnngraph(scale(pb), feature_type=Float32) |> model._device
    pred = Flux.cpu(model(gnn))
    return reshape(pred, ne(pb), nk(pb))
end
