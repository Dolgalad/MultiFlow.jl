push!(LOAD_PATH,"./src/")
push!(LOAD_PATH,"../src/")

using Documenter, MultiFlows


function get_title(markdown_file_path::AbstractString)
    first_line = open(markdown_file_path) do io
        readline(io)
    end
    return String(chop(first_line; head=2, tail=0))
end

pages_files = [
    "First steps" => [
        "index.md",
    ],
    "Core API" => [
        "core_functions/feature_graph.md",
    ],
    "Multi-Commodity Flows" => [
        "mcf/mcf.md",
        "mcf/io.md",
        "mcf/plot.md",
        "mcf/solver.md",
    ]
]

pages = [
    section_name => [
        get_title(joinpath(normpath(@__FILE__, ".."), "src", file)) => file for
        file in section_files
    ] for (section_name, section_files) in pages_files
]

DocMeta.setdocmeta!(MultiFlows, :DocTestSetup, :(using MultiFlows); recursive=true) 
makedocs(
           sitename="MultiFlows.jl", 
           format = Documenter.HTML(prettyurls = false),
           pages=[
               section_name => [
                   get_title(joinpath(normpath(@__FILE__, ".."), "src", file)) => file for
                   file in section_files
               ] for (section_name, section_files) in pages_files
           ],
           modules=[MultiFlows],

        )

deploydocs(
           repo = "github.com/Dolgalad/MultiFlows.jl.git",
           devbranch = "docs",
          )
