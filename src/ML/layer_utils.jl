"""
    M3EdgeReverseLayer <: GraphNeuralNetworks.GNNLayer

Utility layer, reverses the directions of the graph edges and returns new GNNGraph.
"""
struct M3EdgeReverseLayer <: GraphNeuralNetworks.GNNLayer
end

"""
    (l::M3EdgeReverseLayer)(g::GNNGraph)

Definition of the forward call for the [`M3EdgeReverseLayer`](@ref). Features are not changed.

# Example
```jldoctest; setup = :(using GraphNeuralNetworks)
julia> g = GNNGraph([1,2,3],[2,3,1],ndata=(;x=rand(2,3)),edata=(;e=rand(4,3)))
GNNGraph:
  num_nodes: 3
  num_edges: 3
  ndata:
	x = 2×3 Matrix{Float64}
  edata:
	e = 4×3 Matrix{Float64}

julia> edge_index(g)
([1, 2, 3], [2, 3, 1])

julia> gr = M3EdgeReverseLayer()(g);

julia> edge_index(gr)
([2, 3, 1], [1, 2, 3])

```
"""
function (l::M3EdgeReverseLayer)(g::GNNGraph)
    s, t = edge_index(g)
    return GNNGraph(t, s, ndata=g.ndata, edata=g.edata)

    return GNNGraph(t, s, ndata=(;x=g.ndata.x), edata=(;e=g.edata.e))
end

"""
    MLP(in_dim, out_dim, hidden_dim; drop_p=0.1)

Build a Multi-layer perceptron with input dimension `in_dim`, output dimension `out_dim` and hidden dimension `hidden_dim`. `drop_p` is the dropout probability.

# Example
```jldoctest
julia> m = MLP(2, 3, 4)
Chain([
  Dense(2 => 4, relu),                  # 12 parameters
  Dropout(0.1),
  Dense(4 => 4, relu),                  # 20 parameters
  Dropout(0.1),
  Dense(4 => 3),                        # 15 parameters
  Dropout(0.1),
])                  # Total: 6 arrays, 47 parameters, 620 bytes.

```
"""
function MLP(in_dim, out_dim, hidden_dim; drop_p=0.1)
    return Chain([
                  Flux.Dense(in_dim=>hidden_dim, relu),
		  Flux.Dropout(drop_p),
                  Flux.Dense(hidden_dim=>hidden_dim, relu),
		  Flux.Dropout(drop_p),
                  Flux.Dense(hidden_dim=>out_dim),
		  Flux.Dropout(drop_p),
                 ])
end

"""
    MyMEGNetConv3(n::Int64; drop_p::Float64=0.1)

MEGNetConv version 3. Implementation of a message passing layer based on _GraphNeuralNetworks.jl_ `MEGNetConv` layer with a multi-layer perceptron for edge and vertex encoding.

# Example
```jldoctest
julia> MyMEGNetConv3(2)
GraphNeuralNetworks.MEGNetConv{Flux.Chain{Vector{Any}}, Flux.Chain{Vector{Any}}, typeof(Statistics.mean)}(Chain([Dense(6 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Chain([Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 4, relu), Dropout(0.1), BatchNorm(4), Dense(4 => 2)]), Statistics.mean)

```
"""
function MyMEGNetConv3(n::Int64; drop_p::Float64=0.1)
    phie = Chain([Flux.Dense(3*n=>2*n, relu),
		  Flux.Dropout(drop_p),
                  Flux.BatchNorm(2*n),
                  Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n)
		  ])
    phiv = Chain([Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
                  Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n)
		  ])
    return MEGNetConv(phie, phiv)
end

