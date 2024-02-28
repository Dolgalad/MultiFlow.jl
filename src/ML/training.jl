function train_model(model,
                     optimizer,
                     loss,
                     train_dataloader,
                     test_dataloader;
                     device=CUDA.functional() ? Flux.gpu : Flux.cpu,
                     epochs=100,
                     batchsize=1,
                     callbacks=[],
                     early_stopping=nothing,
		     )

    # model parameters
    ps = Flux.params(model)

    # training history, this object is passed as second argument to each callback
    history = Dict("train_loss"=>[],
                   "test_loss"=>[]
                  )

    
    println("starting training")

    state = Flux.setup(optimizer, model)

    for epoch in 1:epochs
        # epoch progress bar
        bar = ProgressBar(train_dataloader)
        set_description(bar, "Epoch $epoch")

        epoch_losses = []
        epoch_metrics = []

        for g in bar
            # send batch to device
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
        end
        

        # update training and testing history
        update!(history, "train_loss", mean(vcat(epoch_losses...)))
        update!(history, epoch_metrics, prefix="train")
        
        test_loss = loss(test_dataloader, model)
        update!(history, "test_loss", test_loss)
        test_metrics = metrics(test_dataloader, model)


        update!(history, test_metrics, prefix="test")


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
        # reclaim memory at end of batch
        GC.gc()
	if CUDA.functional()
	    CUDA.reclaim()
	end
    end
end 


