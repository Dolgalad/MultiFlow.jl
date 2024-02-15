# Multi-Commodity Flow

_MultiFlows.jl_ offers the `MCF` interface for creating Multi-Commodity Flow problems. An `MCF` instance is defined by a capacitated graph and a set of demands. The objective is to find a set of paths that satisfy the demands without exceeding edge capacities.

## Formulation

### Arc-flow ormulation

The problem is defined on a directed graph ``G = (V, A)`` with edge costs ``r_a \in \mathbb{R}^+`` and capacities ``c_a \in \mathbb{R}^+``. We are given a set of demands ``K`` with each demand having an origin and destination ``s_k, t_k \in V`` and an amount ``b_k \in \mathbb{R}^+``. Denoting by ``x_a^k \in [0, 1]`` the amount of flow for demand ``k`` circulating through edge ``a`` the problem can be written : 

```math
\begin{aligned}
\min\limits & \sum\limits_{k \in K} \sum\limits_{a\in A} b_k r_a x_a^k & & (1) \\
\text{s.t.} &\sum\limits_{a \in \delta^+(v)} x_a^k - \sum\limits_{a\in\delta^-(v)} x_a^k = \left\{ \begin{array}{rl} 1 & \text{if } v=s_k \\ -1 & \text{if } v=t_k \\ 0 & \text{otherwise} \end{array} \right. & \forall k\in K, v\in V \quad & (2)\\
&\sum\limits_{k\in K} b_k x_a^k \leq c_a & \forall a \in A \quad & (3)\\
&x_a^k \in [0, 1] & \forall k\in K, a\in A \quad & (4)
\end{aligned}
```

``\delta^-(u), \delta^+(u)`` respectively denote the set of incomming and outgoing edges at vertex ``u``. Constraint ``(2)`` are flow conservation constraints and ``(3)`` the edge capacity constraints. When replacing ``(4)`` by integer constraints ``x_a^k \in \{0, 1\}`` we refer to this problem as the **Unsplittable Multi-Commodity Flow** problem, demands must then by fully routed on a single path.

### Path-flow formulation
For each ``k \in K`` we defined ``P^k`` the set of ``s_k-t_k``-paths composed of edges with capacity geater than ``b_k``. For each ``p \in P^k`` we denote ``x_p^k`` the amount of flow circulating on ``p`` for demand ``k``. Let ``P_a^k \subset P^k`` the set of paths that pass through edge ``a``. The **Path-flow** formulation of the problem is then : 

```math
\begin{aligned}
\min\limits & \sum\limits_{k \in K} \sum\limits_{p\in P^k} \sum\limits_{a\in p} b_k r_a x_p^k & & (5) \\
\text{s.t.} &\sum\limits_{p\in P^k} x_p^k  = 1 & \forall k\in K & (6)\\
&\sum\limits_{k\in K} \sum\limits_{p\in P^k_a} b_k x_p^k \leq c_a & \forall a \in A  & (7)\\
&x_p^k \in [0, 1] & \forall k\in K, p\in P^k & (8)
\end{aligned}

Constraint ``(6)`` ensures that the totality of ``b_k`` is routed and ``(7)`` that the edge capacities are not exceeded.
```

## Index

```@index
Pages = ["mcf.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["demand.jl", "mcf.jl", "solution.jl"]

```

