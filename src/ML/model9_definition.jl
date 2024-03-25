"""
Variant of [`M8ClassifierModel`](@ref) that does not depend on the graph size (number of vertices).
"""

"""
    M9ClassifierModel

Classifier model.
"""
struct M9ClassifierModel
    node_encoder::Flux.Chain
    edge_encoder::Flux.Chain
    demand_encoder::Flux.Chain
    graph_conv::Vector{M8MPLayer}
    demand_mlp::Flux.Chain
    edge_mlp::Flux.Chain
    scoring::Flux.Bilinear
    _device::Function
end

Flux.@functor M9ClassifierModel

"""
    M9ClassifierModel(node_feature_dim::Int64, 
                      edge_feature_dim::Int64, 
		      embedding_dim::Int64,
                      n_layers::Int64, 
                      nnodes::Int64; 
                      drop_p::Float64=0.1,
                      device::Function=CUDA.functional() ? Flux.gpu : Flux.cpu,
    )

`M9ClassifierModel` constructor 
"""
function M9ClassifierModel(node_feature_dim::Int64, 
			   edge_feature_dim::Int64, 
			   embedding_dim::Int64,
			   n_layers::Int64; 
                           drop_p::Float64=0.1,
			   device::Function=CUDA.functional() ? Flux.gpu : Flux.cpu,
                           reverse::Bool=true
			   )
    #node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    node_encoder = MLP(node_feature_dim, embedding_dim, embedding_dim, drop_p=drop_p)
    edge_encoder = MLP(edge_feature_dim, embedding_dim, embedding_dim, drop_p=drop_p)
    demand_encoder = MLP(2*embedding_dim+1, embedding_dim, embedding_dim, drop_p=drop_p)
    graph_conv      = [M8MPLayer(embedding_dim, drop_p=drop_p, reverse=reverse) for _ in 1:n_layers]

    demand_mlp = MLP(3*((n_layers+1)*embedding_dim), embedding_dim, embedding_dim, drop_p=drop_p)
    edge_mlp = MLP(2*((n_layers+1)*embedding_dim), embedding_dim, embedding_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((embedding_dim, embedding_dim) => 1)

    M9ClassifierModel(node_encoder, edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring, device)
end

"""
    compute_graph_embeddings(model::M9ClassifierModel, g::GNNGraph)

Compute the graph embeddings, returns a matrix for the vertex embeddings and another for the edge embeddings.

"""
function compute_graph_embeddings(model::M9ClassifierModel, g::GNNGraph)
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
    #node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    node_embeddings_0 = model.node_encoder(g.ndata.x[:, g.ndata.mask])
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


    return encoded_nodes, _encoded_edges

end

"""
    make_demand_codes(model::M9ClassifierModel, g::GNNGraph)

Make the demand codes.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); g = to_gnngraph(pb, feature_type=Float32); model = M9ClassifierModel(4,3,2,nv(pb)))
julia> demand_codes = make_demand_codes(model, g);

julia> size(demand_codes)
(36, 3)

julia> size(demand_codes,2) == nk(pb)
true

```

"""
function make_demand_codes(model::M9ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

"""
    (model::M9ClassifierModel)(g::GNNGraph)

[`M9ClassifierModel`](@ref) forward call. Returns the flattened edge-demand score matrix.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); g = to_gnngraph(pb, feature_type=Float32); model = M9ClassifierModel(4,3,2,nv(pb)))
julia> y = model(g);

julia> size(y)
(1, 24)

julia> size(y,2) == ne(pb) * nk(pb)
true

```

"""
function (model::M9ClassifierModel)(g::GNNGraph)
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
    (model::M9ClassifierModel)(pb::MCF)

[`M9ClassifierModel`](@ref) forward call for MCF problem `pb`. Returns a `(ne(pb), nk(pb))` matrix.

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); model = M9ClassifierModel(4,3,2,nv(pb)))
julia> y = model(pb);

julia> size(y)
(8, 3)

julia> size(y,1) == ne(pb)
true

julia> size(y,2) == nk(pb)
true

```

"""
function (model::M9ClassifierModel)(pb::MCF)
    gnn = to_gnngraph(scale(pb), feature_type=Float32) #|> model._device
    S,T = edge_index(gnn)
    gg = SimpleDiGraph(Edge.([(s,t) for (s,t) in zip(S[gnn.edata.mask], T[gnn.edata.mask])]))
    pos_x, pos_y = spring_layout(gg)
    nf = zeros(Float32, 2, nv(gnn))
    nf[1,gnn.ndata.mask] .= pos_x
    nf[2,gnn.ndata.mask] .= pos_y
    # add node features
    gnn = GNNGraph(gnn, ndata=(;gnn.ndata..., x=nf)) |> model._device
    
    pred = Flux.cpu(model(gnn))
    return reshape(pred, ne(pb), nk(pb))
end
