module ML

using MultiFlows
import MultiFlows: demand_endpoints, demand_amounts, sparsify

using Graphs
using Flux
using Flux: DataLoader
using CUDA
using GraphNeuralNetworks
using MLUtils
using ProgressBars
using JLD2
using ChainRules, ChainRulesCore
using Statistics
using Plots
using GraphPlot
using LinearAlgebra
using SimpleWeightedGraphs

#using HDF5
#using SparseArrays
#using Graphs
#using SimpleWeightedGraphs

export augmented_graph,
       get_instance,
       to_gnngraph,
       AugmentedGNNGraph,
       aggregate_demand_labels,
       # dataset
       load_dataset,
       make_batchable,
       load_instance,
       add_stacked_index,
       # metrics
       accuracy,
       precision,
       recall,
       f_beta_score,
       graph_reduction,
       metrics,
       # plots
       make_plots,
       # history
       last_metrics,
       last_value,
       update!,
       # training
       train_model,
       # layers
       M3EdgeReverseLayer,
       MLP,
       MyMEGNetConv3,
       # utilities
       make_demand_codes,
       compute_edge_demand_scores,
       # M8ClassifierModel
       M8MPLayer,
       M8ClassifierModel,
       compute_graph_embeddings,
       # ML sparsifier
       m8_post_processing,
       M8MLSparsifier,
       # M9ClassifierModel
       M9ClassifierModel,
       M9MLSparsifier


# Machine learning
include("./metrics.jl")
include("./history.jl")
include("./training.jl")
include("./classifier_utils.jl")
include("./plots.jl")
#include("./mem_utils.jl")
#include("./filter.jl")
include("./augmented_graph.jl")
#include("./instance_generation.jl")
include("./dataset.jl")
include("./layer_utils.jl")
include("./model8_definition.jl")
include("./model8_sparsify.jl")
include("./model9_definition.jl")
include("./model9_sparsify.jl")

end # module
