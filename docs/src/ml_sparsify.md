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


## Layers and utilities
```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/classifier_utils.jl", "ML/layer_utils.jl"]

```

## Classifier model

```@autodocs
Modules = [MultiFlows.ML]
Pages = ["ML/model8_definition.jl"]

```

