# Instance generation

Functions and routines for creating new instances by applying perturbations to a reference problem. These functions are used for creating training and testing sets for training Sparsifying models.


```@eval
ENV["GKSwstype"] = "100"
push!(LOAD_PATH, "../../..")
using Graphs, Random, Compose, MultiFlows, GraphPlot, Colors
Random.seed!(123)
gr = grid((3,3))
loc_x, loc_y = spring_layout(gr)
pb = MCF(gr, ones(12), ones(12), [Demand(1,8,1.),Demand(7,5,.5)])
pb1 = shake(pb)
pb2 = shake(pb, origins_destinations=(1:nv(pb), 1:nv(pb)))
draw(PNG("origin_grid_mcf.png",16cm,16cm), mcfplot(pb, loc_x, loc_y))
draw(PNG("shook_grid_mcf_1.png",16cm,16cm), mcfplot(pb1, loc_x, loc_y))
draw(PNG("shook_grid_mcf_2.png",16cm,16cm), mcfplot(pb2, loc_x, loc_y))

# saturating instance
Random.seed!(123)
loc_x1, loc_y1 = spring_layout(gr)
pb = MCF(gr, ones(12), ones(12), [Demand(1,2,.5)])
pb = shake(pb, nK=5, origins_destinations=(1:nv(pb),1:nv(pb)))
sol,_ = solve_column_generation(pb)
draw(PNG("grid3x3_nonsat_solution.png", 16cm, 16cm), mcfsolplot(sol, pb, loc_x, loc_y, minedgelinewidth=1, maxedgelinewidth=1))
avail = available_capacity(sol, pb)
otherdir = vcat(avail[Int64(ne(pb)/2)+1:end], avail[1:Int64(ne(pb)/2)])
H = vec(minimum(hcat(avail, otherdir), dims=2))
com = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=100)
edgestrokec = com[trunc.(Int64, H * 99) .+ 1]
draw(PNG("grid3x3_nonsat_capacities.png", 16cm, 16cm), gplot(pb.graph, loc_x, loc_y, edgestrokec=edgestrokec, arrowlengthfrac=0, nodelabel=collect(1:nv(pb))))
pb_sat,sol_sat = MultiFlows.saturate(pb)
avail = available_capacity(sol_sat, pb_sat)
otherdir = vcat(avail[Int64(ne(pb_sat)/2)+1:end], avail[1:Int64(ne(pb_sat)/2)])
H = vec(minimum(hcat(avail, otherdir), dims=2))
edgestrokec = com[trunc.(Int64, H * 99) .+ 1]
draw(PNG("grid3x3_sat_capacities.png", 16cm, 16cm), gplot(pb_sat.graph, loc_x, loc_y, edgestrokec=edgestrokec, arrowlengthfrac=0, nodelabel=collect(1:nv(pb))))
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

