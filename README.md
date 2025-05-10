# course-project-01-EPO

# Project #1 — Cost‑Optimised IIoT Task Scheduling with Deadline Guarantees in a Hierarchical Fog–Cloud Architecture

---

## 1  Introduction & Motivation

Industrial‑IoT (IIoT) applications such as predictive maintenance, real‑time quality inspection and digital twins generate heterogeneous computational tasks with strict response‑time guarantees. A hierarchical "fog–cloud" architecture—where near‑device fog nodes handle latency‑sensitive work and public cloud VMs offer elastic capacity—can satisfy these requirements if two intertwined scheduling decisions are solved simultaneously:

1. **Placement** – decide whether a task runs on a fog node or in the cloud.
2. **Timing** – decide when the task starts so that every deadline is met.

Because the joint search space is combinatorial and the workload varies over time, we adopt an advanced meta‑heuristic—the **Enhanced Puma Optimizer (EPO)**—as the core engine. EPO’s adaptive exploration/exploitation strategy provides the robustness needed to minimise total cost (compute + data transfer + energy) while guaranteeing that every task finishes before its deadline.

---

## 2  Formal Problem Statement

**Inputs**

| Symbol      | Meaning                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------- |
| *G = (V,E)* | DAG workflow. Each task *v ∈ V* has workload *w\_v* (CPU‑cycles) and edges carry data *d\_{v→u}*.  |
| **F**       | Set of fog nodes; each has speed *S\_f* (MIPS), price *c\_f* (\$/MI) and intrinsic latency *ℓ\_f*. |
| **C**       | Set of cloud VMs; each has speed *S\_c*, price *c\_c* and round‑trip latency *ℓ\_c*.               |
| *Δ\_v*      | Absolute deadline of task *v*.                                                                     |
| **B**       | Bandwidth matrix between every pair of nodes.                                                      |

**Decision variables**  For each task *v*: device id *x\_v ∈ F∪C* and start‑time slot *t\_v*.

**Objective** – minimise total monetary cost while forbidding deadline misses.  A large penalty *M* enforces deadlines:

$$
\min\,F\;=\;\sum_{v\in V}\Big[c_{x_v}\,\tfrac{w_v}{S_{x_v}} + p_{x_v}\Big]\;\;+
\sum_{(v,u)\in E}\beta\,d_{v\to u}\;\;+
M\,\sum_{v\in V}\max\big(0,\,(t_v+ w_v/S_{x_v})-\Delta_v\big).
$$

Energy or CO₂ cost may be appended as extra weighted terms if "green" optimisation is required.

---

## 3  Enhanced CEIS Scheduler

### 3.1 Chromosome Encoding

A candidate schedule ("chromosome") is a fixed‑length integer array **χ = ⟨(x\_v, t\_v) : v∈V⟩**, i.e. a device ID and start‑slot for each task.

### 3.2 Fitness Function

The fitness implements the objective above: linear cost terms plus a large death‑penalty *M* for any deadline violation.

### 3.3 Oppositional Initialization

Half of the initial population is generated at random; the other half is its opposition‑based mirror. This widens early exploration without extra evaluations.

### 3.4 Adaptive Exploration/Exploitation

EPO continually measures diversity and convergence speed; each "puma" (individual) decides at every iteration whether to explore (random jump or social foraging) or exploit (ambush or sprint operators).

### 3.5 Deadline‑Aware Repair

After each operator, infeasible individuals are pushed to the nearest feasible start slot or migrated to a faster node. If repair remains impossible, the penalty term preserves selection pressure.

### 3.6 Elite Local Hill‑Climb

The **top‑k** individuals undergo a simple two‑task swap followed by repair. This low‑overhead step shaves the last few milliseconds from makespan and improves deadline‑hit rate.

### 3.7 Complexity Analysis

With population size *N* and workflow size |V|, each generation costs **O(N·|V|)** for operator work and fitness evaluation. Sorting elites adds **O(N log N)**.

---

## 4  Hierarchical Fog–Cloud Optimisation (HFCO)

An outer optimiser decides where and when to deploy extra fog nodes so that the inner CEIS sees a latency‑friendly substrate. Candidate sites are encoded as a bit‑vector and evolved with the same EPO operators, minimising a weighted sum of deployment cost and average end‑to‑end latency. Once a placement is fixed for the present time‑window, CEIS schedules the current workload.

---

## 5  Detailed Pseudo‑Code

### Algorithm 1 – Enhanced\_EPO‑CEIS

```text
Input : Task DAG G, node sets F∪C, deadlines Δ
Output: Cost‑minimal, deadline‑feasible schedule

Phase‑0  Oppositional Initialization
 1  Encode chromosome χ = ⟨(xv,tv) : v∈V⟩
 2  Generate Npop/2 random chromosomes
 3  Create opposition counterparts; keep best Npop

Phase‑1  Evolutionary Loop
 4  while iter < MaxIter and not stagnated do
 5     for each puma χᵢ in population P do
 6         /* Adaptive phase decision */
 7         phase ← Exploration if diversityScore(χᵢ) > intensificationScore(χᵢ)
 8         if phase == Exploration then
 9             χ′ ← RandomJump(χᵢ)   or   SocialForage(χᵢ,P)
10         else  /* Exploitation */
11             χ′ ← Ambush(χᵢ,best)  or  Sprint(χᵢ)
12         end if
13         χ′ ← DeadlineAwareRepair(χ′)
14         if Fitness(χ′) < Fitness(χᵢ) then χᵢ ← χ′
15     end for
16     /* Elite local hill‑climb */
17     for χᵉ in top‑k(P) do
18         χʰ ← SwapTwoTasks(χᵉ); χʰ ← Repair(χʰ)
19         if Fitness(χʰ) < Fitness(χᵉ) then replace
20     end for
21     iter ← iter + 1
22  end while
23  return global_best
```

### Algorithm 2 – HFCO\_Fog\_Placement

```text
Input : Candidate sites S, forecast demand Λ, budget Bmax
Output: Deployed fog set F* and inner‑layer schedule

 1  Encode placement ζ as bit‑vector over S
 2  Initialise population with greedy‑coverage and random variants
 3  while stop criterion not met do
 4      evolve ζ with EPO operators
 5      objective   F = w₁·avgLatency(Λ,ζ) + w₂·DeployCost(ζ)
 6  end while
 7  Deploy best ζ*  → update F
 8  Call Enhanced_EPO‑CEIS(G, F∪C, Δ)
```

---

## 6  Parameter Recommendations

| Parameter                   | Default                  | Comment                                       |
| --------------------------- | ------------------------ | --------------------------------------------- |
| Population (*Npop*)         | 30–50                    | Good trade‑off between diversity and runtime. |
| MaxIter                     | 200                      | Increase to 300 when HFCO outer loop active.  |
| Phase weights (PF₁,PF₂,PF₃) | 0.5, 0.5, 0.3            | Derived by flock‑centroidal‑control tuning.   |
| Exploration depth (U,L,a)   | 0.2, 0.7, 2              | Controls jump radius.                         |
| Penalty *M*                 | 10 × (max per‑task cost) | Guarantees deadline dominance.                |

---

## 7  Evaluation Plan

* **Simulator:** CloudSim 6.0 (with iFogSim 2 extensions).
* **Datasets:** Bosch Production Line IIoT traces plus synthetic DAGs (100–1000 tasks).
* **Baselines:** GA, PSO, NSGA‑II, EPSO, Min‑Min, FirstFit.
* **Metrics:** deadline‑hit rate, total cost, average latency, energy (Wh), utilisation.
* **Statistics:** Student t‑test (p < 0.05) across 30 independent runs.

Phase 1 deliverables: CEIS code + graphs of makespan, cost vs. workload.

Phase 2 deliverables: HFCO code + end‑to‑end evaluation under varied city‑scale topologies; sensitivity analysis for fog density and bandwidth price.

---

## 8  Proposed Enhancements & Research Extensions

* **Pareto‑front EPO:** natively trade off cost vs. latency.
* **Opposition‑Based Learning in all generations:** refresh diversity when stagnation detected.
* **Hybrid Tabu‑Search refinement** on final elite schedule.
* **Container‑aware model:** include cold‑start penalties for micro‑service functions.
* **Carbon‑aware scheduling:** re‑rank nodes by real‑time CO₂ intensity to cut emissions.
* **Reinforcement‑guided phase weights:** let the phase‑change mechanism adapt to workload history.

---

## 9  Indicative Timeline

| Month | Milestone                                       |
| ----- | ----------------------------------------------- |
| 1     | Literature study, CloudSim environment ready    |
| 2     | CEIS encoding + EPO integration, unit tests     |
| 3     | Phase‑1 experiments, draft report               |
| 4     | HFCO design & integration                       |
| 5     | Full evaluation, statistical analysis           |
| 6     | Thesis writing, GitHub finalisation, demo video |

---

## 10  Expected Contributions

1. **EPO‑driven discrete scheduler** that minimises cost and guarantees 100 % deadline adherence.
2. **Joint HFCO + CEIS framework** that co‑optimises fog placement and task mapping.
3. **Open‑sourced CloudSim/iFogSim artefacts** enabling reproducible IIoT scheduling research.
4. Empirical evidence that EPO outperforms state‑of‑the‑art meta‑heuristics on deadline‑constrained cost minimisation.

---

This document is ready for inclusion in the project report or as a README in the accompanying GitHub repository.

---

## 9  Variable Glossary (README & Algorithms)

| Variable        | Type           | Appears in        | Meaning                                                         |
| --------------- | -------------- | ----------------- | --------------------------------------------------------------- |
| **G=(V,E)**     | DAG            | Problem Statement | Workflow of IIoT tasks; *V*: tasks, *E*: data‑dependency edges. |
| **v, u**        | Integer / node | Problem Statement | Individual task (node) identifiers.                             |
| **w₍ᵥ₎**        | Float (MI)     | Problem Statement | Required CPU millions‑of‑instructions for task *v*.             |
| **d₍ᵥ→u₎**      | Float (MB)     | Problem Statement | Data that must be sent from *v* to *u*.                         |
| **Δ₍ᵥ₎**        | Float (s)      | Problem Statement | Hard absolute deadline by which *v* must finish.                |
| **𝑭, 𝑪**      | Set            | Problem Statement | Fog node set / Cloud VM set.                                    |
| **S₍f₎, S₍c₎**  | Float (MIPS)   | Problem Statement | Processing speed of a fog node *f* or cloud VM *c*.             |
| **c₍f₎, c₍c₎**  | Float (\$/MI)  | Fitness Eq.       | Monetary cost per MI executed on fog / cloud.                   |
| **ℓ₍f₎, ℓ₍c₎**  | Float (ms)     | System Model      | Round‑trip network latency to fog / cloud.                      |
| **B**           | Matrix (Mbps)  | System Model      | Available bandwidth between all nodes.                          |
| **x₍ᵥ₎**        | Integer        | Chromosome        | Index of the device chosen for *v*.                             |
| **t₍ᵥ₎**        | Integer (slot) | Chromosome        | Start time‑slot assigned to *v*.                                |
| **χ**           | Array          | Algorithms        | A complete schedule chromosome.                                 |
| **P**           | Population     | Algorithms        | Current set of pumas (schedules).                               |
| **Npop**        | Integer        | Params            | Population size.                                                |
| **MaxIter**     | Integer        | Params            | Maximum iterations/generations.                                 |
| **PF₁,PF₂,PF₃** | Float          | PO internals      | Weights that tune phase‑change sensitivity.                     |
| **M**           | Float          | Fitness Eq.       | Large penalty for each ms of deadline violation.                |
| **Λ**           | Demand matrix  | HFCO              | Expected task‑arrival rates per zone.                           |
| **ζ**           | Bit‑vector     | HFCO              | Fog deployment decision at candidate sites.                     |
| **F₁, F₂**      | Float          | HFCO fitness      | Weighted latency and deployment‑cost objectives.                |
| **TaskSizeSet** | List\[int]     | Evaluation        | {100,200,…,1000,1500,2000,2500,3000}.                           |

---

## 10  Additional Detailed Pseudo‑Code

### 10.1 Fitness Computation

```text
function Fitness(χ):
    cost  ← 0
    pen   ← 0
    for v in V:
        dev ← χ.device[v]; start ← χ.slot[v]
        dur ← w_v / S_dev
        finish ← start + dur
        cost += c_dev * dur
        if finish > Δ_v: pen += M * (finish - Δ_v)
    for (v,u) in E:
        bw  ← B[χ.device[v]][χ.device[u]]
        cost += β * d_vu / bw   // transfer cost
    return cost + pen
```

### 10.2 Deadline‑Aware Repair

```text
function DeadlineAwareRepair(χ):
    for v in topological_sort(V):
        dev  ← χ.device[v]
        start← χ.slot[v]
        while start + w_v/S_dev > Δ_v and start > 0:
            start -= 1                      // move earlier
        if start < 0:                       // still infeasible
            dev ← fastest_available_node
            start ← earliest_free_slot(dev)
        χ.slot[v] ← start; χ.device[v] ← dev
    return χ
```

### 10.3 Social Forage Operator (Discrete)

```text
function SocialForage(χ_i, P):
    μ ← mean_chromosome(P)
    χ_new ← χ_i
    for gene g in 1..|χ_i|:
        if rand() < ρ:
            χ_new[g] ← round(χ_i[g] + rand() * (μ[g] - χ_i[g]))
    return χ_new
```

### 10.4 Local Swap Hill‑Climb

```text
function SwapTwoTasks(χ):
    (i,j) ← pick_two_tasks_without_dependency()
    swap χ.device[i], χ.device[j]
    swap χ.slot[i],   χ.slot[j]
    return χ
```

---

## 11  Evaluation Dataset Specification

* **Dataset:** *Alibaba Cluster Trace 2018* (public, 12‑day production‑cluster logs).
* **Extraction:**

  * Filter job records whose workflow DAG can be fully reconstructed (≈ 2.6 M jobs).
  * Generate synthetic IIoT workflows by grouping consecutive tasks from the same job ID, retaining original CPU & memory usage as *w₍ᵥ₎*.
* **Workload Scales:**

  * **Small range:** 100 tasks → 1000 tasks in steps of 100.
  * **Large range:** 1000 tasks → 3000 tasks in steps of 500.
  * For each scale pick 10 random samples (with replacement) to create 130 distinct DAGs.
* **Replay:** feeds arrive‑times into CloudSim exactly as in trace; EPO‑CEIS scheduling makes on‑line decisions.
* **Reported Metrics:** Cost, Makespan, Deadline‑Hit‑Rate, Average Fog/Cloud Utilisation, p‑value vs. baselines.

These additions clarify every variable in the README, supply lower‑level algorithmic details, and lock‑in the Alibaba Cluster dataset with the specified workload sizes for reproducible evaluation.
