"""
    M8MLSparsifier

Sparsifier using the [`M8ClassifierModel`](@ref) predictor. 
"""
struct M8MLSparsifier <: AbstractSparsifier
    model::Union{Nothing,M8ClassifierModel}
end

"""
    M8MLSparsifier

`M8MLSparsifier constructor from model state defined in file `model_path`. `model_path` file should be a JLD2 file containing the `model_state` field.

# Example
```jldoctest; setup = :(using JLD2, Flux)
julia> model = M8ClassifierModel(2, 2, 2, 2);

julia> model_state = Flux.state(model);

julia> jldsave("checkpoint.jld2"; model_state);

julia> M8MLSparsifier("checkpoint.jld2")
M8MLSparsifier(M8ClassifierModel(Embedding(2 => 2), Chain([Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Chain([Dense(5 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), MultiFlows.ML.M8MPLayer[M8MPLayer(GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), Dropout(0.1), Dropout(0.1), BatchNorm(2), BatchNorm(2), M3EdgeReverseLayer(), GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), 2), M8MPLayer(GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), Dropout(0.1), Dropout(0.1), BatchNorm(2), BatchNorm(2), M3EdgeReverseLayer(), GraphNeuralNetworks.MEGNetConv{Chain{Vector{Any}}, Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean), 2)], Chain([Dense(18 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Chain([Dense(12 => 2, relu), Dropout(0.1), Dense(2 => 2, relu), Dropout(0.1), Dense(2 => 2), Dropout(0.1)]), Bilinear(2 => 1), Flux.cpu))
```
"""
function M8MLSparsifier(model_path::String)
    # create the model
    model_state = JLD2.load(model_path, "model_state")
    nhidden = size(model_state.node_embeddings.weight, 1)
    nnodes = size(model_state.node_embeddings.weight, 2)
    nlayers = length(model_state.graph_conv)
    edge_feature_dim = size(model_state.edge_encoder.layers[1].weight, 2)
    model = M8ClassifierModel(nhidden, edge_feature_dim, nlayers, nnodes)
    # TODO : watch the _device parameter
    Flux.loadmodel!(model, model_state)
    if CUDA.functional()
        return M8MLSparsifier(Flux.gpu(model))
    end
    return M8MLSparsifier(model)
end

"""
    m8_post_processing(scores::AbstractMatrix, pb::MCF)

Post-processing step of the prediction of a [`M8ClassifierModel`](@ref). Ensures that the predicted set of edges forms a connected graph containing the source and destination vertices for each demand.

| **Predicted scores** | **Selected** | **Post-processed** |
| :------------------: | :----------: | :----------------: |
| ![](grid3x3_scores.png)  |  ![](grid3x3_selected.png) |  ![](grid3x3_post_processed.png) |



"""
function m8_post_processing(scores::AbstractMatrix, pb::MCF; threshold::Float64=0.5, keep_proportion::Float64=0.)
    filter = zeros(Bool, ne(pb), nk(pb))
    am = edge_index_matrix(pb.graph)
    for k in 1:nk(pb)
        filter[:,k] = m8_post_processing(scores[:,k], pb.demands[k], am, pb.graph, threshold=threshold, keep_proportion=keep_proportion)
    end
    return filter
end

function m8_post_processing(scores::AbstractVector, demand::Demand, arc_matrix::AbstractMatrix, fg::FeatureDiGraph; threshold::Float64=0.5, keep_proportion::Float64=0.0)
    nE = size(scores,1)
    if keep_proportion>0.0
        to_keep = trunc(Int64, nE * keep_proportion)
        sort_idx = sortperm(scores, rev=true)
        idx = @views scores .>= scores[sort_idx[to_keep]]
    else
        idx = @views scores .>= threshold
    end

    if any(idx)
        # check if there is at least one selected edge leaving the demand source
        if isnothing(findfirst(==(demand.src), fg.srcnodes[idx]))
            temp_scores = -Inf .* ones(length(scores))
            insidx = findall(==(demand.src), fg.srcnodes)
            temp_scores[insidx] .= scores[insidx]
            idx[argmax(temp_scores)] = 1
        end
        # check if there is at least one selected edge entering the demand destination
        if isnothing(findfirst(==(demand.dst), fg.dstnodes[idx]))
            temp_scores = -Inf .* ones(length(scores))
            insidx = findall(==(demand.dst), fg.dstnodes)
            temp_scores[insidx] .= scores[insidx]
            idx[argmax(temp_scores)] = 1
        end

        # create a graph
        g = SimpleWeightedDiGraph(fg.srcnodes[idx], fg.dstnodes[idx], vec(scores[idx]))
        # vertex queue
        q, r = Set(vertices(g)), Set()
        # first vertex in queue
        tloop = @elapsed begin
        while !isempty(q)
            u = first(q)
            inn, outn = inneighbors(g,u), outneighbors(g,u)
            # if node has no outneighbors and no inneighbors skip it
            if isempty(inn) && isempty(outn)
                # it might be necessary to come back to this vertex in the future
                pop!(q)
            elseif !isempty(inn) && !isempty(outn)
                # it is never necessary to come back to this vertex
                push!(r, pop!(q))
            elseif isempty(inn) && u==demand.src
                push!(r, pop!(q))
            elseif isempty(outn) && u==demand.dst
                push!(r, pop!(q))
            elseif isempty(inn)
                # add v with highest score (v,u)
                # all candidate in neighbors
                candidates = fg.srcnodes[findall(==(u), fg.dstnodes)]
                candidate_scores = scores[[arc_matrix[v,u] for v in candidates]]
                (s,vi) = findmax(candidate_scores)
                v = candidates[vi]
                idx[arc_matrix[v,u]] = 1

                if !has_vertex(g, v)
                    add_vertices!(g, v-nv(g))
                end
                if s==0.0
                    add_edge!(g, v, u, 1e-8)
                else
                    add_edge!(g, v, u, s) 
                end
                if !(v in r)
                    push!(q, v)
                end
            else
                # add v with highest score (u,v)
                candidates = fg.dstnodes[findall(==(u), fg.srcnodes)]
                candidate_scores = scores[[arc_matrix[u,v] for v in candidates]]
                (s,vi) = findmax(candidate_scores)
                v = candidates[vi]

                idx[arc_matrix[u,v]] = 1
                if !has_vertex(g, v)
                    add_vertices!(g, v-nv(g))
                end
                if s==0.0
                    add_edge!(g, u, v, 1e-8)
                else
                    add_edge!(g, u, v, s) 
                end

                if !(v in r)
                    push!(q, v)
                end
            end
        end
    end
        p = shortest_paths(g, demand.src, demand.dst)
        if isempty(p)
            p = shortest_paths(fg, demand.src, demand.dst)
            idx[edge_indices(p, fg)] .= 1
        end
        return idx
    else
        idx[edge_indices(shortest_paths(fg, demand.src, demand.dst), fg)] .= 1
        return idx
    end
end


"""
    MultiFlows.sparsify(pb::MCF, sprs::M8MLSparsifier)

Sparsify the problem with prediction from a [`M8ClassifierModel`](@ref).

# Example
```jldoctest; setup = :(using Graphs, Random; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> model = M8ClassifierModel(64, 3, 4, nv(pb));

julia> sprs = sparsify(pb, M8MLSparsifier(model))
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
function MultiFlows.sparsify(pb::MCF, sprs::M8MLSparsifier)
    # get GNNGraph representation of the instance
    #g = to_gnngraph(pb)
    # predict
    scores = sprs.model(pb)
    # TODO: post processing step
    return m8_post_processing(sigmoid(scores), pb)
end
