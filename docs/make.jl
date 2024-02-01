using Documenter, MultiFlows

#makedocs(sitename="MultiFlows.jl documentation")
makedocs(sitename="MultiFlows.jl documentation", format = Documenter.HTML(prettyurls = false))

deploydocs(
           repo = "github.com/Dolgalad/MultiFlows.jl.git",
           devbranch = "docs",
          )
