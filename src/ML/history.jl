"""
    update!(history::Dict, key::String, value::Number)

Update history inplace by adding `value` in the `key` field.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123); hist = Dict(); for i in 1:5 update!(hist,"loss",rand()) end)
julia> hist
Dict{Any, Any} with 1 entry:
  "loss" => [0.521214, 0.586807, 0.890879, 0.190907, 0.525662]

julia> update!(hist, "loss", rand())
6-element Vector{Float64}:
 0.521213795535383
 0.5868067574533484
 0.8908786980927811
 0.19090669902576285
 0.5256623915420473
 0.3905882754313441

```
"""
function update!(history::Dict, key::String, value::Number)
    if haskey(history, key)
        push!(history[key], value)
    else
        history[key] = [value]
    end
end

"""
    update!(history::Dict, values::Dict; prefix::String="train")

Update history inplace by adding `values` to dictionary.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123); hist = Dict(); for i in 1:5 update!(hist,Dict("loss"=>rand(), "acc"=>rand())) end)
julia> hist
Dict{Any, Any} with 2 entries:
  "train_loss" => [0.521214, 0.890879, 0.525662, 0.044818, 0.58056]
  "train_acc"  => [0.586807, 0.190907, 0.390588, 0.933353, 0.327238]

julia> update!(hist, Dict("loss"=>rand(), "acc"=>rand()))

julia> hist
Dict{Any, Any} with 2 entries:
  "train_loss" => [0.521214, 0.890879, 0.525662, 0.044818, 0.58056, 0.526996]
  "train_acc"  => [0.586807, 0.190907, 0.390588, 0.933353, 0.327238, 0.836229]

julia> update!(hist, Dict("loss"=>rand(), "acc"=>rand()), prefix="test")

julia> hist
Dict{Any, Any} with 4 entries:
  "train_loss" => [0.521214, 0.890879, 0.525662, 0.044818, 0.58056, 0.526996]
  "train_acc"  => [0.586807, 0.190907, 0.390588, 0.933353, 0.327238, 0.836229]
  "test_acc"   => [0.465202]
  "test_loss"  => [0.0409061]

```
"""
function update!(history::Dict, values::Dict; prefix::String="train")
    for k in keys(values)
        nk = prefix*"_"*k
        update!(history, nk, values[k])
    end
end

"""
    update!(history::Dict, values::Vector{Dict{String,Number}}; prefix::String="train")

Update history with vector of dictionaries.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123); hist = Dict() )
julia> update!(hist, [Dict("loss"=>rand(), "acc"=>rand()), 
                      Dict("loss"=>rand(), "acc"=>rand())])

julia> hist
Dict{Any, Any} with 2 entries:
  "train_loss" => [0.706046]
  "train_acc"  => [0.388857]

```

"""
function update!(history::Dict, values::Vector; prefix::String="train")
    for k in keys(values[1])
        nk = prefix*"_"*k
        update!(history, nk, mean([em[k] for em in values]))
    end
end

"""
    last_metrics(history::Dict; prefix::String="train")

Get the last set of metrics with prefix `prefix`.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123); hist=Dict(); for i in 1:5 update!(hist,Dict("loss"=>rand(), "acc"=>rand())) end ; for i in 1:5 update!(hist,Dict("loss"=>rand(), "acc"=>rand()), prefix="test") end )
julia> last_metrics(hist)
(loss = 0.5805599818745412, acc = 0.32723787925628356)

julia> last_metrics(hist, prefix="test")
(acc = 0.29536650475479964, loss = 0.6644684787269287)

```

"""
function last_metrics(history::Dict; prefix::String="train")
    return (; (Symbol(replace(k, prefix*"_"=>"")) => history[k][end] for (k,v) in history if startswith(k, prefix*"_"))...)
end

"""
    last_value(history::Dict, key::String)

Get the last `key` value.

# Example
```jldoctest; setup = :(using Random; Random.seed!(123); hist = Dict() ; for i in 1:5 update!(hist,Dict("loss"=>rand(), "acc"=>rand())) end )
julia> last_value(hist, "train_loss")
0.5805599818745412

```

"""
function last_value(history::Dict, key::String)
    return history[key][end]
end

