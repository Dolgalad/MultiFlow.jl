# Reading/writing MCFs

## Index

```@index
Pages = ["mcf_file.md"]
```

## MCF file formats
### Reading/writing CSV files
MCF instances are stored as two CSV files, a `link.csv` file containing edge data and a `service.csv` file containing the demand data. 

Example of a `link.csv` file: 
```csv
|# LinkId|srcNodeId|dstNodeId|capacity |cost|latency|
|--------|---------|---------|---------|----|-------|
|1       |1        |4        |6        |2   |0      |
|2       |1        |2        |12       |3   |0      |
|3       |2        |4        |12       |3   |0      |
|4       |4        |7        |5        |8   |0      |
|5       |2        |3        |11       |4   |0      |
|6       |3        |4        |20       |8   |0      |
|7       |4        |6        |10       |3   |0      |
|8       |6        |7        |20       |3   |0      |
|9       |1        |5        |10       |80  |0      |
|10      |5        |7        |10       |20  |0      |
```

Example of a `service.csv` file: 
```csv
|# DemandId|srcNodeId|dstNodeId|amount   |latency|
|----------|---------|---------|---------|-------|
|1         |1        |7        |5        |0      |
|2         |2        |6        |5        |0      |
|3         |3        |7        |5        |0      |
```

If `dirname` is the path of a directory containing both files as described in the example above we may load an MCF instance : 
```julia
julia> pb = load(dirname)
MCF(nv = 7, ne = 10, nk = 3)
	Demand{Int64, Int64}(1, 7, 5)
	Demand{Int64, Int64}(2, 6, 5)
	Demand{Int64, Int64}(3, 7, 5)
```
## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["mcf_file.jl"]

```

