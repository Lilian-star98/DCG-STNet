# readme


本文件把 **DCG-STNet ** 分为五大类：
**① 实现代码 ② 场景数据生成代码 ③ 数据集 ④ 图生成代码 ⑤ 性能表代码**。

> 论文章节号以 上传论文为准。下表「状态」列：✅ 已实现并核对；⚠️ 论文提及。

---

## ① 实现代码（§3 方法与理论框架）

| 论文章节 | 文件 | 关键类/函数 | 状态 |
|----------|------|-------------|------|
| §3.1–3.2 问题与 SCM 假设 | `01_code/config.py` | `ModelConfig` `DataConfig` `ScenarioConfig` | ✅ |
| §3.3–3.4 节点结构方程 / GRU+门控 GCN | `01_code/model/node_dynamics.py` | `NodeDynamics` `OutputDecoder` | ✅ |
| §3.5 稀疏+缓慢变化正则 | `01_code/model/losses.py` | `L_sparse` `L_smooth` | ✅ |
| §3.6.1 场景编码（多源条件编码） | `01_code/model/scenario_encoder.py` | `ScenarioEncoder` | ✅ |
| §3.6.2 物理约束候选边集 | `01_code/model/candidate_edges.py` | `build_candidate_edges` | ✅ |
| 定理 3.4 分层粗化 | `01_code/model/hierarchical_graph.py` | `coarsen_regions` | ✅ |
| §3.6.3 Gumbel+LWR+Top-K 动态因果图 | `01_code/model/dynamic_causal_graph.py` | `SceneAwareGraphGenerator` | ✅ |
| §3.6.4 自适应融合邻接 | `01_code/model/fusion_adj.py` | `FusionAdjacency` | ✅ |
| §3.7 空间结构化噪声+潜在宏观因子变分推理 | `01_code/model/latent_macro.py` | `LatentMacroInferencer` | ✅ |
| §3.8 双世界反事实（Alg. 3.1） | `01_code/model/counterfactual.py` | `InterventionSpec` `apply_control` `apply_edge_mask` `apply_exog` | ✅ |
| §3.9 复合损失 | `01_code/model/losses.py` | `CompositeLoss` | ✅ |
| §3.9 自适应优化（分组学习率） | `01_code/experiments/train.py` | `main()` 内 grouped optimizer | ✅ |
| 完整模型总装 | `01_code/model/stcausalnet.py` | `STCausalNet` | ✅ |
| 模型包入口 | `01_code/model/__init__.py` | 统一导出 | ✅ |

**消融变体（§4.8）**：论文定义 5 个变体，`STCausalNet.ablation(cfg, variant)` 类方法当前实现 3 个：

| 变体 | 论文 | 代码 | 说明 |
|------|------|------|------|
| `w/o Scene` | ✅ | ✅ | `use_scene=False` |
| `w/o Macro` | ✅ | ✅ | `use_macro=False` |
| `w/o Dynamic` | ✅ | ✅ | `use_dynamic=False` |
| `w/o Causal-Reg` | ✅ | ⚠️ | 构造参数 `use_causal_reg` 已预留，但 `ablation()` 未接入 |
| `w/o Physics` | ✅ | ⚠️ | 反事实推演时关闭 CTM，需在调用处手动关闭 |

`eval_ablation.py` 的 `VARIANTS` 也只含前 3 个；Table 3/4 后两行目前只能用论文报告值。

**训练 / 评估入口**（`01_code/experiments/`）：

```bash
python -m 01_code.experiments.train --dataset metr-la --epochs 20 --ckpt out\01_code\stcausalnet.pt
# 消融 --ablation full|w/o Scene|w/o Macro|w/o Dynamic
python -m 01_code.experiments.eval_ablation --dataset metr-la --epochs 5 --out out\01_code\ablation.json
python -m 01_code.experiments.eval_forecast
python -m 01_code.experiments.eval_events
```

---

## ② 场景数据生成代码（附录 B + E3/E4/E5）

| 论文 | 文件 | 关键函数 | 状态 |
|------|------|----------|------|
| 附录 B CTM 核心 | `01_code/scenarios/ctm.py` | `LinkParams` `step_ctm` `simulate` `physics_violation_rate` `default_params` | ✅ |
| 附录 B 平滑强度 | `01_code/scenarios/intensity.py` | `smooth_beta` | ✅ |
| 附录 B 步骤1–4 工作流 | `01_code/scenarios/generate.py` | `ScenarioSpec` `generate_pair` | ✅ |
| §4.6.1 E3 车道缩减 | `01_code/scenarios/e3_capacity_drop.py` | `run_e3` | ✅ |
| §4.6.2 E4 降雨/降雪 | `01_code/scenarios/e4_weather.py` | `run_e4` | ✅ |
| §4.7 E5 学校+引导 | `01_code/scenarios/e5_school.py` | `run_e5` | ✅ |
| 演示网络（朝阳走廊） | `01_code/scenarios/network.py` | `01_code_NODES` `build_demo_network` | ✅ |
| 真实 Q 适配 | `01_code/scenarios/e_real_q.py` | `run_e3_real` `run_e4_real` `run_e5_real` `_apply_local_drop` `_apply_weather` `_apply_school` `_apply_guidance` | ✅ |
| 场景包入口 | `01_code/scenarios/__init__.py` | 统一导出 | ✅ |

---

## ③ 数据集（§4.1–4.2）

### 数据集清单

| 数据集 | 论文用途 | 原始文件 | 加载调用 |
|--------|----------|----------|----------|
| METR-LA | §4.1.1 / E1 / E2 单变量 speed | `raw_data/METR-LA/METR-LA.h5` + `adj_METR-LA.pkl` | `load_dataset("metr-la")` |
| PEMS-BAY | §4.1.1 / E1 单变量 speed | `raw_data/PEMS-BAY/PEMS-BAY.h5` + `adj_PEMS-BAY.pkl` | `load_dataset("pems-bay")` |
| PEMS04 | §4.1.1 多变量 | `raw_data/PEMS04/PEMS04.npz` | `load_dataset("pems04")` |
| Q-Traffic | §4.1.2 多变量 speed+flow，含事件标注 | `Q/event_traffic_beijing_1km_mv_avg_15min_completion/*.pkl` | `load_dataset("q")` |

Q-Traffic 覆盖北京核心区（39.81–40.01 N, 116.20–116.45 E，15073 路段，2017-04-01 至 05-31）。

### 数据处理代码

| 处理步骤（论文 §4.2） | 文件 | 关键函数 | 状态 |
|----------------------|------|----------|------|
| 6:2:2 切分 / Z-score / 滑动窗口（§4.2.1–4.2.2） | `01_code/data/preprocess.py` | `time_split` `scale_zscore` `inv_zscore` `make_windows` | ✅ |
| 时间特征（sin/cos 时刻、周末、节假日，§4.2.3） | `01_code/data/features.py` | `build_time_features` | ✅ |
| Q 场景特征（事件/天气/POI，§4.2.3） | `01_code/data/features.py` | `build_q_scene_features` | ✅ |
| 真实 Q 子图采样 + 瓶颈/学校节点 | `01_code/data/q_real.py` | `load_real_q_subgraph` `pick_bottleneck_node` `pick_school_node` | ✅ |
| 各数据集加载与邻接 | `01_code/data/loaders.py` | `load_dataset` `load_adjacency` `load_q_coords` | ✅ |
| 包入口（h5py 懒加载） | `01_code/data/__init__.py` | — | ✅ |

底层读写复用 `dc_ctgcn/data_utils.py`。


---

## ④ 图生成代码（Fig.1–Fig.8）

```bash
# 真实 Q 路网出图（默认，max_nodes=2000）
python -m 01_code.figures --out out\01_code\figures_real_q --max-nodes 2000
# 合成朝阳演示网络出图
python -m 01_code.figures --out out\01_code\figures --demo
```

| 图 | 论文位置 | 内容 | 脚本 | 数据来源 | 状态 |
|----|----------|------|------|----------|------|
| Fig.1 | §4.6.1 | 局部路网流影响（拥堵扇面） | `01_code/figures/fig1_local_flow.py` | E3 | ✅ |
| Fig.2 | §4.6.1 | 节点 100 缩道对邻接节点时序 | `01_code/figures/fig2_lane_narrowing.py` | E3 | ✅ |
| Fig.3 | §4.6.2 | 区级均速：晴 vs 降雨演化 | `01_code/figures/fig3_rainfall_speed.py` | E4 | ✅ |
| Fig.4 | §4.6.2 | 天气脆弱性拓扑快照（t=18:00） | `01_code/figures/fig4_weather_topology.py` | E4 | ✅ |
| Fig.5 | §4.7.1 | 学校拥堵热点识别（长尾+时域） | `01_code/figures/fig5_school_hotspot.py` | E5 | ✅ |
| Fig.6 | §4.7.2 | 节点 52/202 微观状态跃迁 | `01_code/figures/fig6_micro_state.py` | E5 | ✅ |
| Fig.7 | §4.7.2–4.7.3 | 干预增益与涟漪效应量化 | `01_code/figures/fig7_ripple_effect.py` | E5 | ✅ |
| Fig.8 | §4.7.3 | 子网拓扑状态跃迁快照 | `01_code/figures/fig8_subnet_topology.py` | E5 | ✅ |

共享样式：`01_code/figures/style.py`（`SPEED_CMAP` `draw_graph`）。
入口：`01_code/figures/__init__.py`（`render_all`，含 `real_q` 开关，自动调用 E3/E4/E5）、`__main__.py`。


---

## ⑤ 性能表代码（Table 1–4）

```bash
# 从论文报告数值渲染四张表（无需重训）
python -c "from 01_code.tables import render_all; render_all('out/01_code/tables')"
# 从 benchmark 实测结果渲染对表
python -m 01_code.experiments.render_benchmark_tables --results out/01_code/benchmark/results.json
```

| 表 | 论文位置 | 内容 | 脚本 | 指标 | 状态 |
|----|----------|------|------|------|------|
| Table 1 | §4.5.1 (E1) | 分布内基准（METR-LA/PEMS-BAY, H=12） | `01_code/tables/table1_in_distribution.py` | MAE/RMSE/MAPE | ✅ |
| Table 2 | §4.5.2 (E2) | 事件分层（S1–S4 + PDR） | `01_code/tables/table2_event_stratified.py` | MAE + PDR | ✅ |
| Table 3 | §4.8 (分析 I) | 预测性能消融 | `01_code/tables/table3_ablation_pred.py` | MAE/RMSE | ✅ |
| Table 4 | §4.8 (分析 II) | 反事实有效性消融 | `01_code/tables/table4_ablation_cf.py` | PVR/TED/图稀疏度 | ✅ |

**完整训练对表（基线 × 4 数据集，§4.4）**：

```bash
python -m 01_code.experiments.benchmark --datasets metr-la pems-bay pems04 q `
  --models GraphWaveNet D2STGNN CaST PDFormer STAEformer STCausalNet `
  --epochs 12 --batch-size 32 --hidden 32 --max-train 2048 --max-eval 512 `
  --q-max-nodes 500 --out out/01_code/benchmark
```

---

## 基线模型（§4.4）

| 基线 | 论文出处 | 文件 | 状态 |
|------|----------|------|------|
| GraphWaveNet | IJCAI 2019 | `01_code/baselines/graphwavenet.py` | ✅ |
| D2STGNN | VLDB 2022 | `01_code/baselines/d2stgnn.py` | ✅ |
| CaST | NeurIPS 2022 | `01_code/baselines/cast.py` | ✅ |
| PDFormer | AAAI 2023 | `01_code/baselines/pdformer.py` | ✅ |
| STAEformer | CIKM 2023 | `01_code/baselines/staeformer.py` | ✅ |
| 公共工具 | — | `01_code/baselines/common.py`（`normalize_adj` `TemporalConv`） | ✅ |
| 注册表 | — | `01_code/baselines/__init__.py`（`BASELINES` `build_baseline`） | ✅ |

---

## 一键跑通

```bash
python -m 01_code.run_01_code_pipeline --out out/01_code
```

产物目录：
- `out/01_code/scenarios/` — E3/E4/E5 CTM 反事实数据
- `out/01_code/figures/` 或 `figures_real_q/` — Fig.1–8
- `out/01_code/tables/` — Table 1–4
- `out/01_code/benchmark/` — 基线对表实测结果

---