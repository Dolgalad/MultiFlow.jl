"""
Routines for producing TikZ outputs : 
  - graphs
  - bar plots
"""

"""
    TikZEnvironment

Object representing a TikZ environment.
"""
struct TikZEnvironment
    name::String
    arguments::String
    content::Vector
end

"""
    indentation(n::Int64)

Concatenate `n` indentation characters `c`.
"""
indentation(n::Int64, c::String="\t") = join([c for _ in 1:n])

"""
    begin_str(env::TikZEnvironment)

`TikZEnvironment` begin tag.
"""
function begin_str(env::TikZEnvironment)
    return raw"\begin{" * env.name * raw"}" * (isempty(env.arguments) ? "" : "[" * env.arguments * "]")
end

function Base.show(io::IO, env::TikZEnvironment, indent::Int64=0)
    println(io, indentation(indent) * begin_str(env))
    for item in env.content
        if item isa String
            println(io, indentation(indent+1) * item)
        else
            Base.show(io, item, indent+1)
        end

    end
    println(io, indentation(indent) * raw"\end{" * env.name * raw"}")
end

"""
    tex(env::TikZEnvironment)

Create the Tex content string.
"""
function tex(env::TikZEnvironment)
    buf = IOBuffer()
    println(buf, env)
    return String(take!(buf))
end

"""
    tikz_node(i, posx, posy, arguments, label)

Create the TikZ node line.
"""
function tikz_node(i, posx, posy, arguments, label)
    return raw"\node" * (isempty(arguments) ? "" : raw"[" * arguments * raw"]" ) * " (" * string(i) * raw") at (" * string(posx) * raw"," * string(posy) * raw") {" * string(label) * raw"}; "
end

"""
    tikz_edge(u, v, arguments)

Create the TikZ edge line.
"""
function tikz_edge(u, v, arguments)
    return raw"\path (" * string(u) * raw") edge" * (isempty(arguments) ? "" : raw"[" * arguments * raw"]") * raw" (" * string(v) * raw");"
end

"""
    tikz_graph(g::AbstractGraph{T}, loc_x::Vector{Float64}, loc_y::Vector{Float64}; 
               nodelabel::Union{Nothing,Vector}=nothing,
               arguments::String="",
               nodeargument::Union{Nothing,Vector{String}}=nothing,
               edgeargument::Union{Nothing,Vector{String}}=nothing
    ) where {T}

TikZ representation of graph `g`.

"""
function tikz_graph(g::AbstractGraph{T}, loc_x::Vector{Float64}, loc_y::Vector{Float64}; 
                    nodelabel::Union{Nothing,Vector}=nothing,
                    arguments::String="",
                    nodeargument::Union{Nothing,Vector{String}}=nothing,
                    edgeargument::Union{Nothing,Vector{String}}=nothing
    ) where {T}
    if isnothing(nodelabel)
        nodelabel = collect(1:nv(g))
    end
    if isnothing(nodeargument)
        nodeargument = ["" for _ in 1:nv(g)]
    end
    if isnothing(edgeargument)
        edgeargument = ["" for _ in 1:ne(g)]
    end
    # create vertices
    vertex_l = Union{String,TikZEnvironment}[tikz_node(i,loc_x[i], loc_y[i], nodeargument[i], nodelabel[i]) for i in 1:nv(g)]
    # edges
    edge_l = [tikz_edge(src(e), dst(e), edgeargument[i]) for (i,e) in enumerate(edges(g))]
    edge_scope = TikZEnvironment("scope", "on background layer", edge_l)
    push!(vertex_l, edge_scope)
    
    return TikZEnvironment("tikzpicture", arguments, vertex_l)
end
