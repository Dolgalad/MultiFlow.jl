push!(LOAD_PATH,"./src/")
push!(LOAD_PATH,"../src/")

using Documenter, MultiFlows

# run dataset description page generation script
include("./scripts/base_instance_descriptions.jl")

cp(
    normpath(@__FILE__, "../../README.md"),
    normpath(@__FILE__, "../src/index.md");
    force=true,
)

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
        "core_functions/path.md",
    ],
    "Multi-Commodity Flows" => [
        "mcf/mcf.md",
        "mcf/io.md",
        "mcf/plot.md",
    ],
    "Solvers" => [
        "solvers/solver_statistics.md",
        "solvers/heuristic_solver.md",
        "solvers/compact_solver.md",
        "solvers/column_generation.md",
    ],
    "Datasets" => [
        "datasets/description.md",
        "datasets/generators.md",
    ],
    "Sparsification" => [
        "sparsify.md",
    ],
    "Installation and development" => [
        "development.md"
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
           format = Documenter.HTML(
                                    prettyurls = false,
                                    assets = [
                                              "assets/extra_styles.css"
                                             ],
                                   ),
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
           devbranch = "main",
          )
