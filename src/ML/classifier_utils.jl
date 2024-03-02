"""
    compute_edge_demand_scores(model, 
                               edge_codes::AbstractMatrix, 
                               demand_codes::AbstractMatrix, 
                               g::GNNGraph;
    )

Compute the edge-demand scores with classifier model `model`, edge codes `edge_codes`, demand codes `demand_codes` and on graph `g`. Returns a `(1, ne(pb) * nk(pb))` size matrix that should be reshaped `reshape(scores, ne(pb), nk(pb))` so that value `s[a,k]` corresponds to the score of the edge-demand pair ``(a,k)``.

# Example
Create a simple instance and model with edge and demand encoding space of 2
```jldoctest computescores; setup = :(using Graphs, LinearAlgebra)
julia> pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]);

julia> model = M8ClassifierModel(2, 3, 1, nv(pb));

julia> model.scoring.weight[1,:,:] .= diagm(ones(2)); # for simplicity

julia> gnn = to_gnngraph(pb, feature_type=Float32);
```

Create some edge and demand codes and compute scores 

```jldoctest computescores
julia> edge_codes = reshape(1:2*(ne(pb)+2*nk(pb)), 2, ne(pb)+2*nk(pb))
2×14 reshape(::UnitRange{Int64}, 2, 14) with eltype Int64:
 1  3  5  7   9  11  13  15  17  19  21  23  25  27
 2  4  6  8  10  12  14  16  18  20  22  24  26  28

julia> demand_codes = reshape(1:2*nk(pb), 2, nk(pb))
2×3 reshape(::UnitRange{Int64}, 2, 3) with eltype Int64:
 1  3  5
 2  4  6

julia> s = compute_edge_demand_scores(model, edge_codes, demand_codes, gnn);

julia> s = reshape(s, ne(pb), nk(pb))
8×3 Matrix{Float32}:
  5.0   11.0   17.0
 11.0   25.0   39.0
 17.0   39.0   61.0
 23.0   53.0   83.0
 29.0   67.0  105.0
 35.0   81.0  127.0
 41.0   95.0  149.0
 47.0  109.0  171.0

julia> s[3, 2] == dot(edge_codes[:,3], demand_codes[:,2])
true
```
"""
function compute_edge_demand_scores(model, 
				    edge_codes::AbstractMatrix, 
				    demand_codes::AbstractMatrix, 
				    g::GNNGraph;
				    )
    if !haskey(g.gdata, :demand_stacked_idx)
        ngind = graph_indicator(g)
        ninodes = sum(g.ndata.mask[ngind .== 1])
        nidemands = sum(.!g.ndata.mask[ngind .== 1])

        egind = graph_indicator(g, edges=true)
        regind = egind[g.edata.mask]
        niedges = sum(g.edata.mask[egind .== 1])

        # demand node graph indicator
        dgind = ngind[.!g.ndata.mask]

        # stack repeated demand codes
        dind = collect(1:size(dgind,1)) |> model._device
        demand_stacked_idx = reduce(vcat,[repeat(dind[dgind .== i], inner=niedges) for i=1:g.num_graphs])
        demand_stacked = getobs(demand_codes,demand_stacked_idx)


        # stacked repeated edge codes
        reind = collect(1:size(regind,1)) |> model._device
        #edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], nidemands) for i=1:g.num_graphs])
        edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], g.K[i]) for i=1:g.num_graphs])

        edge_stacked = getobs(edge_codes, edge_stacked_idx)
    else
        # TODO: debug
        demand_stacked = obsview(demand_codes,g.demand_stacked_idx[g.demand_stacked_idx .> 0])
        edge_stacked = obsview(edge_codes, g.edge_stacked_idx[g.edge_stacked_idx .> 0])
    end


    # if scoring layer is bilinear
    if model.scoring isa Flux.Bilinear
        scores= model.scoring(edge_stacked, demand_stacked)
    else
        scores= model.scoring(vcat(edge_stacked, demand_stacked))
    end
    return scores
end

"""
    make_demand_codes(node_codes::AbstractMatrix, g::GNNGraph)

Takes the node codes and computes the demand codes by concatening the codes of ``s_k`` and ``t_k``.

# Example
```jldoctest; setup = :(using Graphs; pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]) ; gnn = to_gnngraph(pb) )
julia> pb.demands
3-element Vector{Demand{Int64, Float64}}:
 Demand{Int64, Float64}(1, 4, 1.0)
 Demand{Int64, Float64}(1, 4, 1.0)
 Demand{Int64, Float64}(3, 2, 1.0)

julia> node_codes = reshape(1:2*nv(pb), 2, nv(pb))
2×4 reshape(::UnitRange{Int64}, 2, 4) with eltype Int64:
 1  3  5  7
 2  4  6  8

julia> make_demand_codes(node_codes, gnn)
4×3 Matrix{Int64}:
 1  1  5
 2  2  6
 7  7  3
 8  8  4

```
"""
function make_demand_codes(node_codes::AbstractMatrix, g::GNNGraph)
    ds,dt = demand_endpoints(g)
    return vcat(view(node_codes,:,ds), view(node_codes,:,dt))
end

"""
    make_demand_codes(node_codes::AbstractMatrix, demand_node_codes::AbstractMatrix, g::GNNGraph)

Takes node codes and demand node codes and concatenates codes for ``s_k``, ``t_k`` and the demand node code for ``k``.

# Example
```jldoctest; setup = :(using Graphs; pb = MCF(grid((2,2)), rand(4), rand(4), [Demand(1,4,1.), Demand(1,4,1.), Demand(3,2,1.)]) ; gnn = to_gnngraph(pb); node_codes = reshape(1:2*nv(pb), 2, nv(pb)) )
julia> demand_node_codes = reshape(1:3*nk(pb), 3, nk(pb))
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> make_demand_codes(node_codes, demand_node_codes, gnn)
7×3 Matrix{Int64}:
 1  1  5
 2  2  6
 7  7  3
 8  8  4
 1  4  7
 2  5  8
 3  6  9

```

"""
function make_demand_codes(node_codes::AbstractMatrix, demand_node_codes::AbstractMatrix, g::GNNGraph)
    ds,dt = demand_endpoints(g)
    return vcat(view(node_codes,:,ds), view(node_codes,:,dt), demand_node_codes)
end

"""
    concat_nodes(xi::AbstractMatrix, xj::AbstractMatrix, e::Nothing)

Utility function for concatenating node features.
"""
function concat_nodes(xi::AbstractMatrix, xj::AbstractMatrix, e::Nothing)
    return vcat(xi, xj)
end

