"""
This script produces the base instance descriptions and creates a `docs/src/datasets/description.md` file in the current directory. Images and other assets are created in the `docs/src/assets` directory.
"""

using MultiFlows
using Random
using Latexify
using DataFrames
using GraphPlot
using Compose

println("base_instance_description.jl : pwd = ", pwd())

header_path = "./docs/src/assets/dataset_description_header.md"
target_path = "./docs/src/datasets/description.md"
# copy header to target destination
cp(header_path, target_path, force=true)

instance_dir = "./instances/sndlib"

instance_data = DataFrame(name=[], nv=[], ne=[], nk=[], img=[])

for instance_path in readdir(instance_dir, join=true)
    if is_instance_dir(instance_path)
        instance_name = basename(instance_path)
        pb = MultiFlows.load(instance_path, edge_dir=:double)
        if !isdir("./docs/src/assets/img/base_instances")
            mkpath("./docs/src/assets/img/base_instances")
        end
        draw(PNG("./docs/src/assets/img/base_instances/$(instance_name).png", 6cm, 6cm), mcfplot(pb, 
                                                                                  nodelabel=nothing,
                                                                                  minedgelinewidth=.1,
                                                                                  maxedgelinewidth=.1,
                                                                                  arrowlengthfrac=0.0,
                                                                                 )
            )

        row = DataFrame(name=replace(instance_name, "_"=>raw"\_"), nv=nv(pb), ne=ne(pb), nk=nk(pb), img="![](../assets/img/base_instances/$(instance_name).png)")

        append!(instance_data, row)
    end
end

rename!(instance_data, ["Name", "# Vertices","# Edges","# Demands","Plot"])

# append base instance description to target file
open(target_path, "a") do f
    println(f, "\n## Base instances\n")
    println(f, "| " * join(names(instance_data), " | ") * " |")
    println(f, "| " * join(["----" for _ in 1:length(names(instance_data))], " | ") * " | ")
    for row in eachrow(instance_data)
        println(f, "| " * join(row, " | ") * " | ")
    end
    println(f, "")
end

# Dataset descriptions : name of the base instance, number of instances, description
