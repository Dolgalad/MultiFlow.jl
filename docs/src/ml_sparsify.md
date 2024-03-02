# Machine Learning

Machine learning based sparsification.

```@eval
ENV["GKSwstype"] = "100"
ENV["LD_LIBRARY_PATH"] = ""
push!(LOAD_PATH, "../../..")
using Graphs, Plots, MultiFlows, MultiFlows.ML, StatsBase
using Random
Random.seed!(123)
gr = Graphs.grid((10,10))
demand_od = [sample(1:nv(gr), 2, replace=false) for _ in 1:10]
demands = [Demand(s,t,rand()) for (s,t) in demand_od]
pb = MCF(gr, rand(ne(gr)), rand(ne(gr)), demands)
g = to_gnngraph(pb, feature_type=Float32)
model = M8ClassifierModel(4, 3, 2, nv(pb))
plts = make_plots(model, g)
for k in keys(plts)
    savefig(plts[k], "model_plot_$(String(k)).png")
end

# M8 sparsifying plots
using MultiFlows, MultiFlows.ML
using GraphPlot
using Flux
using Graphs
using Colors
using Compose
using SimpleWeightedGraphs
using Random
Random.seed!(123)
gr = Graphs.grid((3,3))
loc_x, loc_y = spring_layout(gr)
pb = MCF(gr, ones(12), ones(12), [Demand(1,9,1.0)])
model = M8ClassifierModel(64, 3, 4, nv(pb))
sprs = M8MLSparsifier(model)
# plot the score intensity
pred = model(pb)
scores = sigmoid(pred)[:,1]
# normalize
com = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=100)
otherdir = vcat(scores[Int64(ne(pb)/2)+1:end], scores[1:Int64(ne(pb)/2)])
H = vec(maximum(hcat(scores, otherdir), dims=2))
edgestrokec = com[trunc.(Int64, H * 99) .+ 1]
nodelabel=["s","","","","","","","","t"]
draw(PNG("grid3x3_scores.png", 16cm, 16cm), gplot(pb.graph, loc_x, loc_y, nodelabel=nodelabel, edgestrokec=edgestrokec))
# selected subgraph
selected = (scores .>= 0.5)
draw(PNG("grid3x3_selected.png", 16cm, 16cm), gplot(pb.graph, loc_x, loc_y, nodelabel=nodelabel, edgelinewidth=selected))
# post processed subgraph
tpp = @elapsed filter = m8_post_processing(sigmoid(pred) , pb)
cols=[colorant"lightgray", colorant"blue"]
edgestrokec = [cols[1] for _ in 1:size(scores,1)]
edgestrokec[filter[:,1] .!= selected] .= cols[2]
draw(PNG("grid3x3_post_processed.png", 16cm, 16cm), gplot(pb.graph, loc_x, loc_y, nodelabel=nodelabel, edgelinewidth=filter[:,1], edgestrokec=edgestrokec))
nothing
```

## Index

```@index
Pages = ["ml_sparsify.md"]
```

## Instance encoding
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/augmented_graph.jl"]

```

## Datasets
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/dataset.jl"]

```

## Evaluation
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/metrics.jl"]

```

## Plots
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/plots.jl"]

```

## History
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/history.jl"]

```

## Training
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/training.jl"]

```

## Layers and utilities
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/classifier_utils.jl", "ML/layer_utils.jl"]

```

## Classifier model

```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/model8_definition.jl", "ML/model8_sparsify.jl"]

```
