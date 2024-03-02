"""
    accuracy(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the accuracy of the predicted values `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.accuracy([1,0,1,0], [1,1,0,0])
0.5
```

"""
function accuracy(y_pred::AbstractVector, y_true::AbstractVector)
    sum(y_pred .== y_true)/size(y_pred,1)
end

"""
    tpcount(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the True Positive count of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.tpcount([1,0,1,0], [1,1,0,0])
1
```

"""
function tpcount(y_pred::AbstractVector, y_true::AbstractVector)
    sum( Bool.(y_pred .== y_true) .& Bool.(y_true)  )
end

"""
    fpcount(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the False Positive count of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.fpcount([1,0,1,0], [1,1,0,0])
1
```

"""
function fpcount(y_pred::AbstractVector, y_true::AbstractVector)
    sum( Bool.(y_pred .!= y_true) .& Bool.(y_pred)  )
end

"""
    tncount(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the True Negative count of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.tncount([1,0,1,0], [1,1,0,0])
1
```

"""
function tncount(y_pred::AbstractVector, y_true::AbstractVector)
    sum( Bool.(y_pred .== y_true) .& .!Bool.(y_true)  )
end

"""
    fncount(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the False Negative count of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.fncount([1,0,1,0], [1,1,0,0])
1
```

"""
function fncount(y_pred::AbstractVector, y_true::AbstractVector)
    sum( Bool.(y_pred .!= y_true) .& .!Bool.(y_pred)  )
end

"""
    precision(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the precision ([Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)) of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.precision([1,0,1,0], [1,1,0,0])
0.5
```

"""
function precision(y_pred::AbstractVector, y_true::AbstractVector)
    sum(y_pred)>0 ? tpcount(y_pred, y_true)/sum(y_pred) : 0
end

"""
    recall(y_pred::AbstractVector, y_true::AbstractVector)

Evaluate the recall ([Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)) of prediction `y_pred` and true values `y_true`.

# Example
```jldoctest
julia> ML.recall([1,0,1,0], [1,1,0,0])
0.5
```

"""
function recall(y_pred::AbstractVector, y_true::AbstractVector)       
    sum(y_true)>0 ? tpcount(y_pred, y_true)/sum(y_true) : 0
end

"""
    f_beta_score(y_pred::AbstractVector, y_true::AbstractVector, β::Real=1.0)

Evaluate the F-score ([F-score](https://en.wikipedia.org/wiki/F-score)) of prediction `y_pred` and true values `y_true` with weight `β`.

# Example
```jldoctest
julia> f_beta_score([1,0,1,0], [1,1,0,0])
0.5
```

"""
function f_beta_score(y_pred::AbstractVector, y_true::AbstractVector, beta::Real=1.) 
    pr,rc = precision(y_pred,y_true), recall(y_pred,y_true)
    if pr==0 || rc==0
        return 0
    end
    sum(1 .+ (beta^2)) * (pr*rc)/(((beta^2) * pr) + rc)
end

"""
    graph_reduction(y_pred::AbstractArray)

Evaluate the graph reduction i.e. portion of edges not selected by the model.

# Example
```jldoctest
julia> graph_reduction([1,0,1,0])
0.5
```

"""
function graph_reduction(y_pred::AbstractArray)
    sum(y_pred .== 0) / prod(size(y_pred))
end

"""
    check_device(model)

Utility function for checking the device of the model.
"""
function check_device(model)
    if Flux.params(model.scoring)[1] isa CuArray
        return Flux.gpu
    end
    return Flux.cpu
end

"""
    metrics(pred::AbstractVector, labels::AbstractVector)

Evaluates metrics between prediction `pred` and true labels `labels`.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123))
julia> pred = rand([0,1], 10);

julia> labels = rand([0,1], 10);

julia> metrics(pred, labels)
Dict{String, Float64} with 5 entries:
  "rec"  => 0.8
  "gr"   => 0.4
  "prec" => 0.666667
  "f1"   => 0.727273
  "acc"  => 0.7

```
"""
function metrics(pred::AbstractVector, labs::AbstractVector)
    # debug
    #pred, lab = Int64.(cpu(pred .> 0)),Int64.(cpu(labs))
    pred, lab = Int64.(pred .> 0),Int64.(labs)

    #pred, lab = Int64.(cpu((vec(model(g)) .> 0))),Int64.(cpu(vec(g.targets[g.target_mask])))

    acc = accuracy(pred, lab)
    rec = recall(pred, lab)
    prec = precision(pred, lab)
    f1 = f_beta_score(pred, lab)
    gr = graph_reduction(pred)
    #nd = node_embedding_distance(g, model)
    #return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr, "nd"=>nd)
    return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr)

end


"""
    metrics(g::GNNGraph, model)

Evaluate metrics of the prediction of `model` on graph `g`. Returns a dictionary containing the values of the accuracy, recall, precision, F-1 score and the graph reduction.

```jldoctest; setup = :(using Graphs, JuMP, Random; Random.seed!(123))
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> _,(mod,_) = solve_compact(pb, max_acceptance=true,return_model=true);

julia> y = value.(mod[:x]) .> 0 ; 

julia> g = to_gnngraph(pb, y, feature_type=Float32);

julia> model = M8ClassifierModel(4, 3, 2, nv(pb));

julia> metrics(g, model)
Dict{String, Float64} with 5 entries:
  "rec"  => 0.625
  "gr"   => 0.25
  "prec" => 0.277778
  "f1"   => 0.384615
  "acc"  => 0.333333

```

"""
function metrics(g::GNNGraph, model)
    #function metrics(g::GNNGraph, model; device=CUDA.functional() ? Flux.gpu : Flux.cpu)
    # TODO : watch for error on GPU
    dev = check_device(model)
    # debug
    if haskey(g.gdata, :target_mask)
        pred, labs = Int64.(cpu((vec(model(g |> dev)) .> 0))),Int64.(cpu(vec(g.targets[g.target_mask])))
    else
        pred, labs = Int64.(cpu((vec(model(g |> dev)) .> 0))),Int64.(cpu(vec(g.targets)))
    end
    return metrics(pred, labs)
end


"""
    metrics(loader::DataLoader, model)

Evaluate metrics for the prediction of `model` on all samples in a DataLoader object `loader`.

# Example
```jldoctest; setup = :(using Graphs,Random,GraphNeuralNetworks; Random.seed!(123); pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]); model = M8ClassifierModel(4, 3, 2, nv(pb)))
julia> using Flux: DataLoader

julia> loader = DataLoader([to_gnngraph(pb, rand(Bool, ne(pb), nk(pb)), feature_type=Float32)
                                for _ in 1:10], 
                           collate=true
                          )
10-element DataLoader(::Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}, collate=Val{true}())
  with first element:
  GraphNeuralNetworks.GNNGraphs.GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}

julia> metrics(loader, model)
Dict{String, Float64} with 5 entries:
  "rec"  => 0.728095
  "gr"   => 0.25
  "prec" => 0.438889
  "f1"   => 0.53621
  "acc"  => 0.458333

```
"""
function metrics(loader::DataLoader, model)
    dev = check_device(model)
    # debug
    #mtrcs = [metrics(add_stacked_index(g), model) for g in loader]
    mtrcs = [metrics(g, model) for g in loader]

    vs = mean(hcat([collect(values(m)) for m in mtrcs]...), dims=2)
    return Dict(k .=> v for (k,v) in zip(keys(mtrcs[1]), vs))
end

"""
    node_distance_matrix(codes::AbstractMatrix; dist_func::Function=(x,y) -> norm(x-y))

Computes distances between node embeddings, returns matrix. Used for plots.
"""
function node_distance_matrix(codes::AbstractMatrix; dist_func::Function=(x,y) -> norm(x-y))
    nnodes = size(codes, 2)
    m = zeros(nnodes, nnodes)
    for i in 1:nnodes
        for j in 1:nnodes
            if i!=j
                m[i,j] = dist_func(codes[:,i], codes[:,j])
            end
        end
    end
    return m
end 


"""
    node_embedding_distance(g::GNNGraph, model)

Mean embedding distance.
"""
function node_embedding_distance(g::GNNGraph, model)
    node_embeddings, edge_embeddings = compute_graph_embeddings(model, g)
    return norm(apply_edges((xi,xj)->xi.-xj, g, node_embeddings, node_embeddings))
end
