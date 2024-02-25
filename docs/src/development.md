# Installation and Development

## Installation

To install and start using _MultiFlows.jl_ first clone the repository move into the `MultiFlows.jl` directory and launch Julia REPL : 
```bash
git clone https://github.com/Dolgalad/MultiFlows.jl.git
cd MultiFlows.jl
julia --project
```

```julia-pkg
(MultiFlows) pkg> instantiate
[...]

julia> using MultiFlows
```

## Development
The `dev` branch is used for development purposes, new functionality should first be added to this branch. 

### Building documentation
First enter the Julia REPL environment : 
```bash
julia --project
```

Activate the `docs` sub-project : 
```julia
(MultiFlows) pkg> activate ./docs
[...]

(MultiFlows) pkg> dev .
[...]

julia> include("docs/make.jl")
[...]
```

or simply execute the following while in the project root directory : 
```bash
julia --project=./docs docs/make.jl
```

In another terminal start a local web server to serve the contents of the `docs/build` directory : 
```bash
julia --project=./docs -e 'using LiveServer ; serve(dir="docs/build")'
Server ; serve(dir="./docs/build")'
✓ LiveServer listening on http://localhost:8000/ ...
  (use CTRL+C to shut down)

```
