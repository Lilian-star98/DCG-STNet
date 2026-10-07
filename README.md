# DCG-STNet

DCG-STNet: Dynamic Structural Causal Modeling for Urban Traffic Forecasting and Counterfactual Intervention Analysis. Source code for a Transportation Research Part C submission.

This README maps **DCG-STNet / STCausalNet** to five areas: **① model implementation ② scenario generation ③ datasets ④ figure generation ⑤ performance tables**.

> Section numbers follow `manuscript/extracted/01_code_20260111_zh.txt`. In the tables below, ✅ means implemented and checked; ⚠️ means mentioned in the paper but only partially implemented in the code.

---

## ① Model implementation (§3: methods and theoretical framework)

| Paper section | File | Key classes/functions | Status |
|---|---|---|---|
| §3.1–3.2 Problem formulation and SCM assumptions | `01_code/config.py` | `ModelConfig` `DataConfig` `ScenarioConfig` | ✅ |
| §3.3–3.4 Node structural equations / GRU + gated GCN | `01_code/model/node_dynamics.py` | `NodeDynamics` `OutputDecoder` | ✅ |
| §3.5 Sparsity and slow-change regularization | `01_code/model/losses.py` | `L_sparse` `L_smooth` | ✅ |
| §3.6.1 Scenario encoding from multiple sources | `01_code/model/scenario_encoder.py` | `ScenarioEncoder` | ✅ |
| §3.6.2 Physically constrained candidate edges | `01_code/model/candidate_edges.py` | `build_candidate_edges` | ✅ |
| Theorem 3.4 Hierarchical coarsening | `01_code/model/hierarchical_graph.py` | `coarsen_regions` | ✅ |
| §3.6.3 Gumbel + LWR + Top-K dynamic causal graph | `01_code/model/dynamic_causal_graph.py` | `SceneAwareGraphGenerator` | ✅ |
| §3.6.4 Adaptive adjacency fusion | `01_code/model/fusion_adj.py` | `FusionAdjacency` | ✅ |
| §3.7 Spatially structured noise and variational inference for latent macro factors | `01_code/model/latent_macro.py` | `LatentMacroInferencer` | ✅ |
| §3.8 Two-world counterfactuals (Algorithm 3.1) | `01_code/model/counterfactual.py` | `InterventionSpec` `apply_control` `apply_edge_mask` `apply_exog` | ✅ |
| §3.9 Composite loss | `01_code/model/losses.py` | `CompositeLoss` | ✅ |
| §3.9 Adaptive optimization with parameter-group learning rates | `01_code/experiments/train.py` | Grouped optimizer in `main()` | ✅ |
| Full model assembly | `01_code/model/stcausalnet.py` | `STCausalNet` | ✅ |
| Model package entry point | `01_code/model/__init__.py` | Unified exports | ✅ |

**Ablation variants (§4.8):** The paper defines five variants. The `STCausalNet.ablation(cfg, variant)` class method currently implements three:

| Variant | In paper | In code | Notes |
|---|---|---|---|
| `w/o Scene` | ✅ | ✅ | `use_scene=False` |
| `w/o Macro` | ✅ | ✅ | `use_macro=False` |
| `w/o Dynamic` | ✅ | ✅ | `use_dynamic=False` |
| `w/o Causal-Reg` | ✅ | ⚠️ | The `use_causal_reg` constructor argument is reserved, but `ablation()` does not expose it. |
| `w/o Physics` | ✅ | ⚠️ | CTM must be disabled manually at the counterfactual rollout call site. |

The `VARIANTS` list in `eval_ablation.py` also contains only the first three variants. The last two rows of Tables 3 and 4 can currently be rendered only from values reported in the manuscript.

**Training and evaluation entry points** (`01_code/experiments/`):

```bash
python -m 01_code.experiments.train --dataset metr-la --epochs 20 --ckpt out\01_code\stcausalnet.pt
# Ablations: --ablation full|w/o Scene|w/o Macro|w/o Dynamic
python -m 01_code.experiments.eval_ablation --dataset metr-la --epochs 5 --out out\01_code\ablation.json
python -m 01_code.experiments.eval_forecast
python -m 01_code.experiments.eval_events
```

---

## ② Scenario generation (Appendix B and E3/E4/E5)

| Paper component | File | Key functions | Status |
|---|---|---|---|
| Appendix B: CTM core | `01_code/scenarios/ctm.py` | `LinkParams` `step_ctm` `simulate` `physics_violation_rate` `default_params` | ✅ |
| Appendix B: smooth intervention intensity | `01_code/scenarios/intensity.py` | `smooth_beta` | ✅ |
| Appendix B: steps 1–4 workflow | `01_code/scenarios/generate.py` | `ScenarioSpec` `generate_pair` | ✅ |
| §4.6.1 E3: lane reduction | `01_code/scenarios/e3_capacity_drop.py` | `run_e3` | ✅ |
| §4.6.2 E4: rain/snow | `01_code/scenarios/e4_weather.py` | `run_e4` | ✅ |
| §4.7 E5: school and guidance | `01_code/scenarios/e5_school.py` | `run_e5` | ✅ |
| Demonstration network (Chaoyang corridor) | `01_code/scenarios/network.py` | `01_code_NODES` `build_demo_network` | ✅ |
| Real Q-Traffic adapter | `01_code/scenarios/e_real_q.py` | `run_e3_real` `run_e4_real` `run_e5_real` `_apply_local_drop` `_apply_weather` `_apply_school` `_apply_guidance` | ✅ |
| Scenario package entry point | `01_code/scenarios/__init__.py` | Unified exports | ✅ |

---

## ③ Datasets (§4.1–4.2)

### Dataset inventory

| Dataset | Use in paper | Source files | Loader call |
|---|---|---|---|
| METR-LA | §4.1.1 / E1 / E2, univariate speed | `raw_data/METR-LA/METR-LA.h5` + `adj_METR-LA.pkl` | `load_dataset("metr-la")` |
| PEMS-BAY | §4.1.1 / E1, univariate speed | `raw_data/PEMS-BAY/PEMS-BAY.h5` + `adj_PEMS-BAY.pkl` | `load_dataset("pems-bay")` |
| PEMS04 | §4.1.1, multivariate data | `raw_data/PEMS04/PEMS04.npz` | `load_dataset("pems04")` |
| Q-Traffic | §4.1.2, multivariate speed + flow with event labels | `Q/event_traffic_beijing_1km_mv_avg_15min_completion/*.pkl` | `load_dataset("q")` |

Q-Traffic covers central Beijing (39.81–40.01° N, 116.20–116.45° E), 15,073 road segments, from 2017-04-01 through 2017-05-31.

### Data processing

| Processing step (paper §4.2) | File | Key functions | Status |
|---|---|---|---|
| 6:2:2 split, Z-score, and sliding windows (§4.2.1–4.2.2) | `01_code/data/preprocess.py` | `time_split` `scale_zscore` `inv_zscore` `make_windows` | ✅ |
| Time features (sin/cos time of day, weekends, holidays; §4.2.3) | `01_code/data/features.py` | `build_time_features` | ✅ |
| Q-Traffic scenario features (events/weather/POI; §4.2.3) | `01_code/data/features.py` | `build_q_scene_features` | ✅ |
| Real Q-Traffic subgraph sampling and bottleneck/school nodes | `01_code/data/q_real.py` | `load_real_q_subgraph` `pick_bottleneck_node` `pick_school_node` | ✅ |
| Dataset loading and adjacency | `01_code/data/loaders.py` | `load_dataset` `load_adjacency` `load_q_coords` | ✅ |
| Package entry point (lazy `h5py` import) | `01_code/data/__init__.py` | — | ✅ |

The low-level I/O code reuses `dc_ctgcn/data_utils.py`.

---

## ④ Figure generation (Fig. 1–8)

```bash
# Plot the real Q-Traffic road network (default: max_nodes=2000)
python -m 01_code.figures --out out\01_code\figures_real_q --max-nodes 2000
# Plot the synthetic Chaoyang demonstration network
python -m 01_code.figures --out out\01_code\figures --demo
```

| Figure | Paper section | Content | Script | Data source | Status |
|---|---|---|---|---|---|
| Fig. 1 | §4.6.1 | Local road-flow effects (congestion fan) | `01_code/figures/fig1_local_flow.py` | E3 | ✅ |
| Fig. 2 | §4.6.1 | Time series of adjacent nodes after narrowing at node 100 | `01_code/figures/fig2_lane_narrowing.py` | E3 | ✅ |
| Fig. 3 | §4.6.2 | District-level mean speed: clear weather versus rainfall | `01_code/figures/fig3_rainfall_speed.py` | E4 | ✅ |
| Fig. 4 | §4.6.2 | Weather-vulnerability topology snapshot (18:00) | `01_code/figures/fig4_weather_topology.py` | E4 | ✅ |
| Fig. 5 | §4.7.1 | School congestion hotspot identification (long tail and temporal extent) | `01_code/figures/fig5_school_hotspot.py` | E5 | ✅ |
| Fig. 6 | §4.7.2 | Microstate transitions at nodes 52 and 202 | `01_code/figures/fig6_micro_state.py` | E5 | ✅ |
| Fig. 7 | §4.7.2–4.7.3 | Intervention gains and ripple effects | `01_code/figures/fig7_ripple_effect.py` | E5 | ✅ |
| Fig. 8 | §4.7.3 | Subnetwork topology transition snapshots | `01_code/figures/fig8_subnet_topology.py` | E5 | ✅ |

Shared styling: `01_code/figures/style.py` (`SPEED_CMAP`, `draw_graph`).
Entry points: `01_code/figures/__init__.py` (`render_all`, with a `real_q` switch that automatically invokes E3/E4/E5) and `__main__.py`.

---

## ⑤ Performance tables (Tables 1–4)

```bash
# Render four tables from the paper-reported numbers (no retraining)
python -c "from 01_code.tables import render_all; render_all('out/01_code/tables')"
# Render comparisons from measured benchmark results
python -m 01_code.experiments.render_benchmark_tables --results out/01_code/benchmark/results.json
```

| Table | Paper section | Content | Script | Metrics | Status |
|---|---|---|---|---|---|
| Table 1 | §4.5.1 (E1) | In-distribution benchmark (METR-LA/PEMS-BAY, H=12) | `01_code/tables/table1_in_distribution.py` | MAE/RMSE/MAPE | ✅ |
| Table 2 | §4.5.2 (E2) | Event-stratified evaluation (S1–S4 + PDR) | `01_code/tables/table2_event_stratified.py` | MAE + PDR | ✅ |
| Table 3 | §4.8 (Analysis I) | Forecasting ablations | `01_code/tables/table3_ablation_pred.py` | MAE/RMSE | ✅ |
| Table 4 | §4.8 (Analysis II) | Counterfactual-validity ablations | `01_code/tables/table4_ablation_cf.py` | PVR/TED/graph sparsity | ✅ |

**Full training comparison (baselines × four datasets, §4.4):**

```bash
python -m 01_code.experiments.benchmark --datasets metr-la pems-bay pems04 q `
  --models GraphWaveNet D2STGNN CaST PDFormer STAEformer STCausalNet `
  --epochs 12 --batch-size 32 --hidden 32 --max-train 2048 --max-eval 512 `
  --q-max-nodes 500 --out out/01_code/benchmark
```

---

## Baseline models (§4.4)

| Baseline | Publication | File | Status |
|---|---|---|---|
| GraphWaveNet | IJCAI 2019 | `01_code/baselines/graphwavenet.py` | ✅ |
| D2STGNN | VLDB 2022 | `01_code/baselines/d2stgnn.py` | ✅ |
| CaST | NeurIPS 2022 | `01_code/baselines/cast.py` | ✅ |
| PDFormer | AAAI 2023 | `01_code/baselines/pdformer.py` | ✅ |
| STAEformer | CIKM 2023 | `01_code/baselines/staeformer.py` | ✅ |
| Shared utilities | — | `01_code/baselines/common.py` (`normalize_adj`, `TemporalConv`) | ✅ |
| Registry | — | `01_code/baselines/__init__.py` (`BASELINES`, `build_baseline`) | ✅ |

---

## End-to-end run

```bash
python -m 01_code.run_01_code_pipeline --out out/01_code
```

Output directories:

- `out/01_code/scenarios/` — E3/E4/E5 CTM counterfactual data
- `out/01_code/figures/` or `figures_real_q/` — Fig. 1–8
- `out/01_code/tables/` — Tables 1–4
- `out/01_code/benchmark/` — Measured baseline comparison results

---

## Current formal experiments and data limitations

`01_code/formal/` contains the unified-protocol model, calibration code, and independent-event evaluation code. `01_code/formal_workflows/` contains the hyperparameter search, testing, provenance audits, and input specifications. JSON files and figures in `04_results/formal_v3/` record interim results.

The repository omits large raw datasets, model checkpoints, and per-sample prediction files; these remain in the local research archive. Public speed benchmarks alone cannot establish the superiority of the full physics model or real-world intervention effects. In the current five-seed METR-LA results, the forecasting adapter for STCausalNet has a higher MAE than Graph WaveNet, D2STGNN, and STAEformer. Measured Q-Traffic flow, independent density observations, verifiable events, and network-calibration inputs are still needed. See `01_code/formal_workflows/05_真实证据与正式训练_v3.md` for further limitations.
