"""
    M9MLSparsifier

Sparsifier using the [`M9ClassifierModel`](@ref) predictor. 
"""
struct M9MLSparsifier <: AbstractSparsifier
    model::Union{Nothing,M9ClassifierModel}
end

"""
    M9MLSparsifier

`M9MLSparsifier constructor from model state defined in file `model_path`. `model_path` file should be a JLD2 file containing the `model_state` field.

# Example
```jldoctest; setup = :(using JLD2, Flux)
julia> model = M9ClassifierModel(2, 2, 2, 2);

julia> model_state = Flux.state(model);

julia> jldsave("checkpoint.jld2"; model_state);

julia> M9MLSparsifier("checkpoint.jld2")
M9MLSparsifier(M9ClassifierModel(Embedding(2 => 2), Chain([Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Chain([Dense(5 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), MultiFlows.ML.M9MPLayer[M9MPLayer(GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), Dropout(0.1), Dropout(0.1), BatchNorm(2), BatchNorm(2), M3EdgeReverseLayer(), GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), 2), M9MPLayer(GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), Dropout(0.1), Dropout(0.1), BatchNorm(2), BatchNorm(2), M3EdgeReverseLayer(), GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), 2)], Chain([Dense(18 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Chain([Dense(12 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Bilinear(2 => 1), Flux.cpu))
```
"""
function M9MLSparsifier(model_path::String)
    # create the model
    model_state = JLD2.load(model_path, "model_state")
    nhidden = size(model_state.scoring.weight, 2)
    nlayers = length(model_state.graph_conv)
    edge_feature_dim = size(model_state.edge_encoder.layers[1].weight, 2)
    node_feature_dim = size(model_state.node_encoder.layers[1].weight, 2)

    model = M9ClassifierModel(node_feature_dim, edge_feature_dim, nhidden, nlayers)
    # TODO : watch the _device parameter
    Flux.loadmodel!(model, model_state)
    if CUDA.functional()
        return M9MLSparsifier(Flux.gpu(model))
    end
    return M9MLSparsifier(model)
end


"""
    MultiFlows.sparsify(pb::MCF, sprs::M9MLSparsifier)

Sparsify the problem with prediction from a [`M9ClassifierModel`](@ref).

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> model = M9ClassifierModel(64, 3, 4, nv(pb));

julia> sprs = sparsify(pb, M9MLSparsifier(model))
8×3 Matrix{Bool}:
 0  0  1
 1  1  1
 1  1  1
 1  1  1
 0  0  1
 1  1  1
 1  1  0
 0  0  1

```
"""
function MultiFlows.sparsify(pb::MCF, sprs::M9MLSparsifier)
    # get GNNGraph representation of the instance
    #g = to_gnngraph(pb)
    # predict
    scores = sprs.model(pb)
    # TODO: post processing step
    return m8_post_processing(sigmoid(scores), pb)
end
