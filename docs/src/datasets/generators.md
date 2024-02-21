# Instance generation

Functions and routines for creating new instances by applying perturbations to a reference problem. These functions are used for creating training and testing sets for training Sparsifying models.


```@eval
ENV["GKSwstype"] = "100"
push!(LOAD_PATH, "../../..")
using Graphs, Random, Compose, MultiFlows, GraphPlot
Random.seed!(123)
gr = grid((3,3))
loc_x, loc_y = spring_layout(gr)
pb = MCF(gr, ones(12), ones(12), [Demand(1,8,1.),Demand(7,5,.5)])
pb1 = shake(pb)
pb2 = shake(pb, origins_destinations=(1:nv(pb), 1:nv(pb)))
draw(PNG("origin_grid_mcf.png",16cm,16cm), mcfplot(pb, loc_x, loc_y))
draw(PNG("shook_grid_mcf_1.png",16cm,16cm), mcfplot(pb1, loc_x, loc_y))
draw(PNG("shook_grid_mcf_2.png",16cm,16cm), mcfplot(pb2, loc_x, loc_y))
nothing
```

## Index

```@index
Pages = ["generators.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["generators.jl"]

```

