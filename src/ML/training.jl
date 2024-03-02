"""
    train_model(model,
                optimizer,
                loss,
                train_dataloader,
                val_dataloader;
                device=CUDA.functional() ? Flux.gpu : Flux.cpu,
                epochs=100,
                batchsize=1,
                callbacks=[],
                early_stopping=nothing,
    )

Train `model` using optimizer `optimizer`, minimizing loss function `loss`. 

"""
function train_model(model,
                     optimizer,
                     loss,
                     train_dataloader,
                     val_dataloader;
                     device=CUDA.functional() ? Flux.gpu : Flux.cpu,
                     epochs=100,
                     batchsize=1,
                     callbacks=[],
                     early_stopping=nothing,
                     reclaim=false,
                     steps_per_epoch=nothing
		     )

    # model parameters
    ps = Flux.params(model)

    # training history, this object is passed as second argument to each callback
    history = Dict("train_loss"=>[],
                   "val_loss"=>[]
                  )

    
    println("starting training")

    state = Flux.setup(optimizer, model)

    for epoch in 1:epochs
        # epoch progress bar
        bar = ProgressBar(train_dataloader)
        set_description(bar, "Epoch $epoch")

        epoch_losses = Real[]
        epoch_metrics = Dict{String,Number}[]
        for (i,g) in enumerate(bar)
            # send batch to device
            #g = add_stacked_index(g) |> device
            g = g |> device

            # compute gradient
            batch_loss, grad = Flux.withgradient(model) do m
                pred = sigmoid(vec(m(g)))
                labs = vec(g.targets[g.target_mask])
                ignore_derivatives() do
                    push!(epoch_metrics, metrics(pred, labs))
                end
                loss(pred, labs)
            end

            Flux.update!(state, model, grad[1])

            push!(epoch_losses, batch_loss)
            if !isnothing(steps_per_epoch)
                if i >= steps_per_epoch
                    break
                end
            end
        end

        # update training and validation history
        update!(history, "train_loss", mean(vcat(epoch_losses...)))
        update!(history, epoch_metrics, prefix="train")
        
        val_loss = loss(val_dataloader, model)
        update!(history, "val_loss", val_loss)
        val_metrics = metrics(val_dataloader, model)


        update!(history, val_metrics, prefix="val")


        # call callbacks
        for callback in callbacks
            callback(epoch, history)
        end
        # early stopping
        if !isnothing(early_stopping)
            if early_stopping(epoch, history)
                println("Early stopping")
                break
            end
        end
        if reclaim
            # reclaim memory at end of batch
            GC.gc()
	    if CUDA.functional()
	        CUDA.reclaim()
	    end
        end
    end
end 


