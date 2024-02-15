# Multi-Commodity Flow

_MultiFlows.jl_ offers the `MCF` interface for creating Multi-Commodity Flow problems. An `MCF` instance is defined by a capacitated graph and a set of demands. The objective is to find a set of paths that satisfy the demands without exceeding edge capacities.

## Formulation

The problem is defined on a directed graph ``G = (V, A)`` with edge costs ``r_a \in \mathbb{R}^+`` and capacities ``c_a \in \mathbb{R}^+``. We are given a set of demands ``K`` with each demand having an origin and destination ``s_k, t_k \in V`` and an amount ``b_k \in \mathbb{R}^+``. Denoting by ``x_a^k \in [0, 1]`` the amount of flow for demand ``k`` circulating through edge ``a`` the problem can be written : 

``math
\begin{aligned}
\min\limits_{x} & \sum\limits_{k \in K} \sum\limits_{a\in A} b_k r_a x_a^k & & (1) \\
\text{s.t.} &\sum\limits_{a \in \delta^+(v)} x_a^k - \sum\limits_{a\in\delta^-(v)} x_a^k = \left\{ \begin{array}{rl} 1 & \text{if } v=s_k \\ -1 & \text{if } v=t_k \\ 0 & \text{otherwise} \end{array} \right. & \forall k\in K, v\in V \quad & (2)\\
&\sum\limits_{k\in K} b_k x_a^k \leq c_a & \forall a \in A \quad & (3)\\
&x_a^k \in [0, 1] & \forall k\in K, a\in A \quad & (4)
\end{aligned}
``

``\delta^-(u), \delta^+(u)`` respectively denote the set of incomming and outgoing edges at vertex ``u``. Constraint ``(2)`` are flow conservation constraints and ``(3)`` the edge capacity constraints. When replacing ``(4)`` by integer constraints ``x_a^k \in \{0, 1\}`` we refer to this problem as the **Unsplittable Multi-Commodity Flow** problem, demands must then by fully routed on a single path.

## Index

```@index
Pages = ["mcf.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["demand.jl", "mcf.jl", "solution.jl"]

```

