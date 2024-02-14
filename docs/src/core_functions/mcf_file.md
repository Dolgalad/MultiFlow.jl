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
|srcNodeId|dstNodeId|cost|bandwidth|latency|
|---------|---------|----|---------|-------|
|1        |2        |5.0 |10000.0  |123.63 |
|1        |3        |5.0 |10000.0  |190.15 |
|1        |7        |2.5 |20000.0  |280.39 |
|3        |8        |1.0 |50000.0  |87.5   |
|3        |56       |5.0 |10000.0  |78.5   |
|4        |57       |5.0 |10000.0  |43.77  |
|4        |5        |10.0|5000.0   |229.06 |
|4        |8        |5.0 |10000.0  |87.38  |
|5        |8        |1.0 |50000.0  |201.64 |
[...]
```

Example of a `service.csv` file: 
```csv
|srcNodeId|dstNodeId|bandwidth         |latency           |
|---------|---------|------------------|------------------|
|61       |62       |1000.0            |1115.8246731722002|
|48       |57       |6227.536188450593 |904.1495945723898 |
|54       |5        |4137.119197583509 |93.29530333700012 |
|54       |26       |100.0             |397.68708243310357|
|47       |54       |31286.750204673575|626.5855474790925 |
|20       |21       |9999.999999999998 |327.34899755777883|
|47       |57       |1233.0580630388602|592.3648469918586 |
|62       |26       |4045.6446097688377|771.1194450643927 |
|26       |64       |8299.670586179074 |644.2239128380035 |
[...]
```
## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["mcf_file.jl"]

```

