# Multi-Commodity Flow

_MultiFlows.jl_ offers the `MCF` interface for creating Multi-Commodity Flow problems. An `MCF` instance is defined by a capacitated graph and a set of demands. The objective is to find a set of paths that satisfy the demands without exceeding edge capacities.

## Arc-flow formulation

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

## Path-flow formulation
For each ``k \in K`` we defined ``P^k`` the set of ``s_k``-``t_k``-paths composed of edges with capacity geater than ``b_k``. For each ``p \in P^k`` we denote ``x_p^k`` the amount of flow circulating on ``p`` for demand ``k``. Let ``P_a^k \subset P^k`` the set of paths that pass through edge ``a``. The **Path-flow** formulation of the problem is then : 

```math
\begin{aligned}
\min\limits & \sum\limits_{k \in K} \sum\limits_{p\in P^k} \sum\limits_{a\in p} b_k r_a x_p^k & & (5) \\
\text{s.t.} &\sum\limits_{p\in P^k} x_p^k  = 1 & \forall k\in K \quad & (6)\\
&\sum\limits_{k\in K} \sum\limits_{p\in P^k_a} b_k x_p^k \leq c_a & \forall a \in A \quad & (7)\\
&x_p^k \in [0, 1] & \forall k\in K, p\in P^k \quad & (8)
\end{aligned}
```

Constraint ``(6)`` ensures that the totality of ``b_k`` is routed and ``(7)`` that the edge capacities are not exceeded.

## Max-Acceptance
The **Max-Acceptance** formulation allows for some of the demands to not be fully routed at the cost of paying a prohibitive price ``M``. The ``M`` parameter must be chosen to be greater then the longest path in the graph. By default we take ``M = \sum_{a\in A} c_a``. For each demand ``k`` and variable ``y_k \in [0,1]`` is introduced to the model representing the portion of the demand that was not routed.

```math
\begin{aligned}
\min\limits & \sum\limits_{k \in K} \sum\limits_{p\in P^k} \sum\limits_{a\in p} b_k r_a x_p^k + \sum\limits_{k\in K} M b_k y_k  & & (9) \\
\text{s.t.} &\sum\limits_{p\in P^k} x_p^k  + y_k = 1 & \forall k\in K \quad & (10)\\
&\sum\limits_{k\in K} \sum\limits_{p\in P^k_a} b_k x_p^k \leq c_a & \forall a \in A \quad & (11)\\
&x_p^k \in [0, 1] & \forall k\in K, p\in P^k \quad & (12) \\
&y_k \in [0,1] & \forall k\in K \quad & (13)
\end{aligned}
```
Constraint ``(10)``, named the **convexity constraint** ensures that the totality of the demand is routed or not.

## Index

```@index
Pages = ["mcf.md"]
```

## Full docs

```@autodocs
Modules = [MultiFlows]
Pages = ["demand.jl", "mcf.jl", "solution.jl"]

```

