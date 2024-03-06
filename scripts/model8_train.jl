"""Train M8ClassifierModel
Arguments : 
    - train dataset path
    - test dataset path
    - batch size
"""

#ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["GKSwstype"] = "nul"

using MultiFlows, MultiFlows.ML
using ProgressBars
using Plots
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader
using Flux.Losses: logitbinarycrossentropy, binary_focal_loss, mean, logsoftmax
using DataFrames, CSV
using TensorBoardLogger
using Logging
using LinearAlgebra
using JLD2

using Random: seed!

# set the seed
seed!(2023)

CUDA.allowscalar(false)
device = CUDA.functional() ? Flux.gpu : Flux.cpu;

println("CUDA.functional: ", CUDA.functional())

## working directory : environment variable 
#if !("TRAIN_WORKDIR" in keys(ENV))
#    println("The TRAIN_WORKDIR environment variable is not set. Exiting.")
#    exit()
#end
#workdir = ENV["TRAIN_WORKDIR"]
if "MULTIFLOWS_WORKDIR" in keys(ENV)
    workdir = ENV["MULTIFLOWS_WORKDIR"]
else
    workdir = "."
end

# training parameters
epochs = 10000
if length(ARGS)>=2
    bs = parse(Int64, ARGS[2])
else
    bs = 32
end
lr = 1.0e-6
nhidden = 64
nlayers = 4
tversky_beta = 0.1
es_patience = 1000
solve_interval = 10
_reverse = false

model_name_prefix = "temp_model8"

#if length(ARGS)>=1
#    dataset_path = joinpath(ARGS[1], "train")
#    dataset_name = basename(ARGS[1])
#    test_dataset_path = joinpath(ARGS[1], "test")
#else
#    dataset_path = joinpath(workdir, "datasets", "Oxford_0_1_1", "train")
#    dataset_name = "Oxford_0_1_1"
#    test_dataset_path = joinpath(workdir, "datasets", "Oxford_0_1_1", "test")
#end

dataset_name = "AsnetAm_0_1_1_cgcolumns_rnk0.1_train"
dataset_path = joinpath(workdir, "datasets", dataset_name)

if !isdir(dataset_path)
    throw(ErrorException("Dataset path '$(dataset_path)' not found"))
end

# model name and save path
model_name = model_name_prefix * "_l"*string(nlayers)*"_lr"*string(lr)*"_h"*string(nhidden)*"_bs"*string(bs)*"_e"*string(epochs)*"_tversky"*string(tversky_beta)*"_"*string(_reverse ? "rev" : "norev")
save_path = joinpath(workdir, "models", dataset_name, model_name)
if isdir(save_path)
    rm(save_path, force=true, recursive=true)
end
mkpath(save_path)

# tensorboard logging directory
tb_log_dir = joinpath(workdir, "tb_logs", dataset_name, model_name)

# testing solve output dir
test_solve_output_dir = joinpath(workdir, "solve_outputs", dataset_name, model_name)
if isdir(test_solve_output_dir)
    rm(test_solve_output_dir, force=true, recursive=true)
end
mkpath(test_solve_output_dir)


# print summary
println("Training summary")
println("\tWorkdir                     = ", workdir)
println("\tepochs                      = ", epochs)
println("\tbatch size                  = ", bs)
println("\tlearning rate (effective)   = ", lr, " (", bs*lr, ")")
println("\thidden dimension            = ", nhidden)
println("\tGCN layers                  = ", nlayers)
println("\ttversky beta                = ", tversky_beta)
println("\tearly stopping patience     = ", es_patience)
println("\ttrain dataset               = ", dataset_path)
println("\tdataset name                = ", dataset_name)
println("\tmodel name                  = ", model_name)
println("\tcheckpoint directory        = ", save_path)
println("\tTensorBoard directory       = ", tb_log_dir)
println("\ttest solve output directory = ", test_solve_output_dir)


# run once for compilation

##solveUMF(inst,"CG","highs","./output.txt")
##solveUMF(inst,"CG","highs","./output.txt","","clssp model5_test_checkpoint.bson 1")
#for inst_dir in readdir(test_dataset_path, join=true)
#    if UMFSolver.is_instance_path(inst_dir)
#        local inst = UMFSolver.scale(UMFData(inst_dir))
#        s1,ss1 = solveUMF(inst,"CG","cplex",joinpath(output_dir_default, basename(inst_dir))*".json")
#    end
#end

print("Loading dataset...")
# TODO: not enough memory on my machine for whole dataset
all_graphs = load_dataset(dataset_path, transform_f=aggregate_demand_labels, show_progress=true, max_n=1000)
#all_graphs = map(aggregate_demand_labels, all_graphs)

nnodes = sum(all_graphs[1].g.ndata.mask)
println("nnodes : ", nnodes)

train_graphs, val_graphs = MLUtils.splitobs(all_graphs, at=0.9)

train_loader = DataLoader(train_graphs,
                batchsize=bs, shuffle=true, collate=true)
val_loader = DataLoader(val_graphs,
               batchsize=bs, shuffle=false, collate=true)
println("done")

function solve_dataset(graphs, solve_f)
    objective_values, solve_times = [], []
    bar = ProgressBar(graphs)
    set_description(bar, "Solving validation set: ")
    for (i,g) in enumerate(bar)
        _,ss = solve_f(g)
        push!(objective_values, ss["objective_value"])
        push!(solve_times, ss["solve_time"])
    end
    return objective_values, solve_times
end

# model
model = M8ClassifierModel(nhidden, 3, nlayers, nnodes, reverse=_reverse) |> device

# optimizer
opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam(bs* lr))

# run once for compilation
pb = get_instance(val_graphs[1])
solve_column_generation(pb)
solve_column_generation(pb, pricing_filter=ones(Bool, ne(pb), nk(pb)))
sprs = M8MLSparsifier(device(model))
solve_column_generation(pb, pricing_filter=sparsify(pb, sprs))

# solve the instances in the validation set
default_solve_vals, default_solve_times = solve_dataset(val_graphs, 
              g->solve_column_generation(get_instance(g))
             )

# debug
function loss(pred::AbstractVector, labs::AbstractVector)
    return Flux.Losses.tversky_loss(pred, labs; beta=tversky_beta)
end

function get_labs(g)
    g = g |> device
    return vec(g.targets[g.target_mask])
end
function loss(loader, m)
    r = mean(loss(sigmoid(vec(model(ag |> device))), get_labs(ag)) for ag in loader)
    return r
end




# tensorboard callback
logger = TBLogger(tb_log_dir, tb_overwrite)


function TBCallback(epoch, history)
    train_metrics = last_metrics(history, prefix="train")
    val_metrics = last_metrics(history, prefix="val")

    with_logger(logger) do
        @info "train" train_metrics... log_step_increment=0
        @info "val" val_metrics...
        @info "plot" make_plots(model, val_loader.data[1].g)... log_step_increment=0

    end

    #@info "train" train_metrics
    @info "val" val_metrics

end

# save model checkpoint callback
function save_model_checkpoint(epoch, history)
    _model = model |> Flux.cpu
    model_state = Flux.state(_model)
    jldsave(joinpath(save_path, "checkpoint_e$(epoch).jld2"); model_state)
end


# use model to solve test instances
function solve_test_dataset(epoch, history)
    if epoch % solve_interval == 0 || epoch==1
        println("Testing solving validation set")
        # create a directory for this epochs test solve outputs for K=0
     	checkpoint_path = joinpath(save_path, "checkpoint_e$epoch.jld2")
        sparsifier = M8MLSparsifier(checkpoint_path)
        solve_vals, solve_times = solve_dataset(val_graphs,
                      g->solve_column_generation(get_instance(g),
                                                      pricing_filter=sparsify(get_instance(g),
                                                                              sparsifier)
                                                     ))
	# update training history, add mean optimality criterion and speedup value
        update!(history, Dict(
			      "opt"=>mean((solve_vals .- default_solve_vals) ./ default_solve_vals), 
			      "spd"=>mean((solve_times .- default_solve_times) ./ default_solve_times)
			      ), 
			 prefix="val")
    end
end

# Early stopping
function es_metric(epoch, history)
    return last_value(history, "val_loss")
end
es = Flux.early_stopping(es_metric, es_patience, min_dist=1.0f-8, init_score=1.0f8);

# start training
train_model(model,opt,loss,train_loader,val_loader,
                      callbacks=[save_model_checkpoint,solve_test_dataset,TBCallback],
                      early_stopping=es,
                      epochs=epochs,
                      #steps_per_epoch=30,
                     )

