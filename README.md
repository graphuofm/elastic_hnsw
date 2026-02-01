# ElasticHNSW

Optimized HNSW graph maintenance for high-churn vector databases. ElasticHNSW extends [hnswlib](https://github.com/nmslib/hnswlib) with **Lazy Bridging** and **Adaptive M** to preserve search quality under continuous insert/delete workloads — the kind of workloads seen in LLM agent memory systems, recommendation engines, and streaming analytics.

## Problem

Standard HNSW handles static datasets well, but degrades significantly under high churn (frequent insertions and deletions). When nodes are deleted, the graph loses connectivity — search latency increases and recall drops. Naive solutions like periodic full rebuilds are expensive and cause service disruptions.

## Approach

ElasticHNSW introduces two lightweight mechanisms implemented directly in hnswlib's C++ core (`hnswalg.h`):

### Lazy Bridging

When a node is deleted, its neighbors may become disconnected. Before marking a node as deleted, ElasticHNSW inspects the node's neighbor list and selectively adds "bridge" edges between nearby neighbors to preserve local connectivity.

- Collects valid (non-deleted) neighbors of the target node
- Sorts neighbors by distance to the deleted node
- For the closest neighbors (up to 6), evaluates candidate bridge edges using a distance heuristic: only bridge if `dist(n1, n2) ≤ 1.5 × (dist(del, n1) + dist(del, n2))`
- Adds bidirectional edges for qualifying pairs

### Adaptive M

Standard HNSW enforces a hard cap (`maxM0_`) on neighbor list size. During bridging, this cap can prevent necessary repair edges from being added. Adaptive M allows neighbor lists to temporarily accept bridge edges up to the `maxM0_` limit without overflow, keeping the graph connected while staying within the pre-allocated memory layout.

### Threshold-Triggered Rebuild (τ)

At the Python experiment layer, ElasticHNSW monitors query latency and triggers a full index rebuild when the latency ratio exceeds a configurable threshold τ (e.g., τ = 2.0 means rebuild when latency doubles). This gives a tunable knob between rebuild cost and search quality.

## Repository Structure

```
elastic_hnsw/
├── run_all_experiments.py          # Main experiment driver (Experiments A–F)
├── scripts/
│   ├── test_elastic_hnsw.py        # ElasticHNSW vs Vanilla comparison
│   ├── baseline_vanilla_complete.py
│   ├── baseline_vanilla_extreme.py
│   ├── exp1_m_impact.py            # Effect of M parameter
│   ├── exp2_churn_degradation.py   # Churn-induced degradation
│   ├── exp3_latency_impact.py      # Latency analysis
│   ├── exp4_elastic_hnsw.py
│   ├── exp5_lazy_bridging_simulation.py
│   ├── exp6_elastic_vs_vanilla.py
│   ├── exp7_fair_comparison.py
│   ├── exp8_smart_rebuild.py
│   ├── exp9_extreme_churn.py
│   └── exp10_faiss_comparison.py
├── results/                        # Early experiment results (.txt, .json)
├── results_supplementary/          # Supplementary experiments A–F (.json)
│   ├── exp_a_tau.json              # τ sensitivity analysis
│   ├── exp_b_highdim.json          # High-dimension experiments (128/384/768d)
│   ├── exp_c_scalability.json      # Scalability (10K–100K)
│   ├── exp_d_intensity.json        # Churn intensity (5%–30%)
│   ├── exp_e_extreme.json          # Extreme 2000% churn
│   └── exp_f_rebuild_cost.json     # Rebuild cost vs index size
└── .gitignore
```

**Note:** The modified `hnswlib` source and the SIFT1M dataset are not included in this repository. See [Setup](#setup) below.

## Setup

### Prerequisites

- Python 3.8+
- A C++ compiler (g++ or clang)
- numpy

### Install hnswlib with ElasticHNSW modifications

1. Clone [hnswlib](https://github.com/nmslib/hnswlib) into `hnswlib/` under this project root.

2. Apply the ElasticHNSW modifications to `hnswlib/hnswlib/hnswalg.h`:

   Add `#include <algorithm>` near the top of the file (after `namespace hnswlib {`), then add the following two functions before `markDeletedInternal`:

   - `repairNeighborsOnDelete(tableint internalId)` — implements Lazy Bridging
   - `tryAddBridgeEdge(tableint n1, tableint n2)` — adds bidirectional bridge edges with Adaptive M

   Modify `markDeletedInternal` to call `repairNeighborsOnDelete(internalId)` before marking the node as deleted.

3. Build and install:

   ```bash
   cd hnswlib
   pip install . --user --force-reinstall --no-deps
   ```

### Download SIFT1M dataset

```bash
mkdir -p data/sift
cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz
```

## Running Experiments

Run all supplementary experiments (A–F):

```bash
cd /path/to/elastic_hnsw
python3 run_all_experiments.py 2>&1 | tee all_results.log
```

This takes roughly 40–60 minutes and writes results to `results_supplementary/`.

### Experiment Overview

| Experiment | Description | Key Variable |
|---|---|---|
| A | τ sensitivity analysis | τ ∈ {1.3, 1.5, 2.0, 2.5, 3.0} |
| B | High-dimensional data | 128d / 384d / 768d |
| C | Index scalability | 10K / 30K / 50K / 100K vectors |
| D | Churn intensity | 5% / 10% / 20% / 30% per round |
| E | Extreme churn (2000%) | Vanilla vs Elastic vs Full Rebuild |
| F | Rebuild cost profiling | Build time vs index size |

All experiments compare three strategies:
- **Vanilla HNSW** — standard hnswlib with mark-delete, no graph repair
- **ElasticHNSW** — Lazy Bridging + Adaptive M + threshold-triggered rebuild
- **Full Rebuild** — rebuild the entire index every round (upper bound on quality, lower bound on efficiency)

## Key Results

ElasticHNSW consistently reduces latency degradation compared to Vanilla HNSW under high-churn workloads, while using significantly fewer rebuilds than the full-rebuild baseline. The τ parameter provides a practical knob: lower τ values trigger more frequent rebuilds (better latency, higher cost), while higher values tolerate more degradation before rebuilding.

## Implementation Details

The core modifications are minimal — roughly 80 lines of C++ added to `hnswalg.h`:

- **`repairNeighborsOnDelete()`**: Called inside `markDeletedInternal()` before the delete mark is set. Inspects the deleted node's neighbor list, computes pairwise distances between valid neighbors, and adds bridge edges for geometrically close pairs.

- **`tryAddBridgeEdge()`**: Adds a bidirectional edge between two nodes. Checks for existing connections, respects the `maxM0_` memory limit, and uses per-node mutex locks for thread safety.

- **`markDeletedInternal()`**: Modified to call `repairNeighborsOnDelete()` as its first operation.

The Python layer (`run_all_experiments.py`) handles the threshold-triggered rebuild logic: it periodically samples query latency and rebuilds the index from scratch when latency exceeds τ × baseline.

## License

This project builds on [hnswlib](https://github.com/nmslib/hnswlib) (Apache 2.0).
