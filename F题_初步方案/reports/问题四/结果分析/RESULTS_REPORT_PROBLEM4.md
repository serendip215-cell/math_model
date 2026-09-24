# 问题四结果：开放模型能力变化与条件情景

> 计算日期基于附件快照。预测原点是同协议 C1 的 2025-03-13，12/24 个月情景没有同长度回测，不能解读为当前实测前沿或可信外推预测。

## 数据与三个硬边界

1. **开放性：** 主样本要求 C2 权重明确为 Yes 且许可证在预设宽松集合中。缺失权重证据不视为开放。预训练、后训练分层；合并模型不参与主分解。
2. **跨表连接：** C1/C4 只接受唯一名称归一化候选、参数量接近、发布日期可核、语言领域、Confident/Likely 且权重信息无冲突的连接。该表仅是非随机子样本。
3. **Loss 桥接和未来：** C6 高可比只有同一家族的 7 个 Pythia 点；C1 可比评分截至 2025-03，不能把问题三 Loss 换算成高分前沿，也没有 12/24 个月同协议回测。

### 样本流失

| step | rows | unique_models | families |
| --- | --- | --- | --- |
| C2 raw | 4576 | 4497 |  |
| valid date/positive N/six tasks | 4561 | 4482 |  |
| earliest valid submission per model | 4482 | 4482 |  |
| permissive license | 1725 | 1725 | 284 |
| strict open weight + license | 236 | 236 | 24 |
| strict pretrained | 42 | 42 | 13 |
| strict posttrained | 141 | 141 | 17 |

C1/C4 名称候选 109 行，规则接受 93 行，严格开放预训练且算力可用 30 行。候选和接受均不等于人工逐项核验。

### C8 逐任务核算

扫描 1954 个 JSON；解析 1954 个；模型去重后 1861 个；与 C1 连接 1855 个，其中同一叶任务集合 1855 个。C8 叶任务宏平均与 C1 BBH **数值标尺未证实相同**，因此只做覆盖与秩序审计，不强行回代相等。重复版本以记录时间及文件名排序取末份。

## 规模、时间残差与验证

模型将六任务均分映射到 logit 空间，拟合 log10 参数量与提交时间的 Ridge；超参数在训练期按模型家族 GroupKFold 选择。2025-01 至 03 月留作时间留出，比较仅规模、规模加时间与训练均值。时间系数吸收未观测架构、数据、后训练和选择变化，不能解释为纯技术进步。

| stratum | variant | n | families | train_n | future_holdout_n | future_holdout_rmse | future_holdout_bias | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_pretrained | N_only | 42 | 13 | 37 | 5 | 11.371 | -7.4973 | fit |
| strict_pretrained | N_plus_time | 42 | 13 | 37 | 5 | 8.001 | -1.3651 | fit |
| strict_pretrained | training_mean | 42 | 13 | 37 | 5 | 12.544 | -5.2019 | baseline |
| strict_posttrained | N_only | 141 | 17 | 74 | 67 | 8.8211 | -4.0436 | fit |
| strict_posttrained | N_plus_time | 141 | 17 | 74 | 67 | 10.964 | 5.8357 | fit |
| strict_posttrained | training_mean | 141 | 17 | 74 | 67 | 9.7967 | -0.20245 | baseline |
| license_only_pretrained | N_only | 108 | 28 | 70 | 38 | 3.5396 | -1.2096 | fit |
| license_only_pretrained | N_plus_time | 108 | 28 | 70 | 38 | 2.9849 | -0.091865 | fit |
| license_only_pretrained | training_mean | 108 | 28 | 70 | 38 | 5.8979 | 2.4236 | baseline |

### 共同规模支持

| stratum | early_n | late_n | overlap_logN_low | overlap_logN_high | overlap_early_n | overlap_late_n | overlap_early_families | overlap_late_families | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_pretrained | 30 | 5 | 0.033021 | 1.1694 | 19 | 5 | 11 | 4 | False |
| strict_posttrained | 33 | 67 | 0.063333 | 1.784 | 31 | 66 | 10 | 6 | True |
| license_only_pretrained | 44 | 38 | -0.86967 | 1.3724 | 38 | 38 | 16 | 7 | True |

只有支持审计通过的层才执行早期（截至 2024-08）与晚期（2025-01 至 03）两因素 Shapley 分解。分解数值是模型拟合变化，与实测变化另列；家族聚簇重抽样输出在 `decomposition_family_bootstrap.csv`。未通过者不报贡献百分比。

| stratum | scale_points | conditional_time_points | model_change_points | observed_change_points | unexplained_points | t0 | t1 | causal_attribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_posttrained | -4.9088 | 5.8188 | 0.90996 | 2.316 | 1.406 | 0.95277 | 7.4415 | False |
| license_only_pretrained | -3.6628 | 1.5757 | -2.0872 | -2.2924 | -0.20524 | 0.59138 | 7.3429 | False |

家族重抽样分位数如下。严格后训练层的模型总变化区间跨 0；且时间模型在未来时段留出 RMSE 高于仅规模模型。因此分解只作为探索性统计，不能形成稳定的贡献比例或已验证时间增长机制。许可证单独确认、权重未确认层也仅为敏感性。

| stratum | quantile | scale_points | conditional_time_points | model_change_points |
| --- | --- | --- | --- | --- |
| license_only_pretrained | 0.025 | -6.2245 | -1.1485 | -5.6274 |
| license_only_pretrained | 0.5 | -3.3199 | 1.4989 | -1.5475 |
| license_only_pretrained | 0.975 | 0.96641 | 4.1914 | 4.4462 |
| strict_posttrained | 0.025 | -15.793 | 1.0546 | -6.0385 |
| strict_posttrained | 0.5 | -4.2926 | 5.1121 | 0.81193 |
| strict_posttrained | 0.975 | 1.8372 | 14.153 | 11.129 |

### 任务敏感性

| score_definition | late_minus_early |
| --- | --- |
| six_task_mean | 3.3561 |
| exclude_IFEval | 4.456 |
| exclude_BBH | 3.9386 |
| exclude_MATH Lvl 5 | 1.1734 |
| exclude_GPQA | 3.7309 |
| exclude_MUSR | 3.581 |
| exclude_MMLU-PRO | 3.2564 |
| task_IFEval | -2.1437 |
| task_BBH | 0.44308 |
| task_MATH Lvl 5 | 14.269 |
| task_GPQA | 1.4819 |
| task_MUSR | 2.2314 |
| task_MMLU-PRO | 3.8543 |

六项任务各自用训练期家族分组 CV 选正则化强度，再在 2025-01 至 03 月做时间留出。各任务的尺度和难度不同，原始 RMSE 不能直接相加；它们用于检查综合均分是否掩盖异质性。

| task | variant | future_holdout_n | future_holdout_rmse | future_holdout_bias |
| --- | --- | --- | --- | --- |
| IFEval | N_only | 67 | 21.89 | 3.9552 |
| IFEval | N_plus_time | 67 | 27.386 | 16.817 |
| BBH | N_only | 67 | 15.6 | -4.8122 |
| BBH | N_plus_time | 67 | 16.573 | 4.4189 |
| MATH Lvl 5 | N_only | 67 | 19.469 | -14.663 |
| MATH Lvl 5 | N_plus_time | 67 | 13.962 | 5.5261 |
| GPQA | N_only | 67 | 6.2632 | -3.128 |
| GPQA | N_plus_time | 67 | 6.0086 | 0.4733 |
| MUSR | N_only | 67 | 6.5583 | -2.7345 |
| MUSR | N_plus_time | 67 | 6.7239 | 2.5698 |
| MMLU-PRO | N_only | 67 | 14.633 | -7.6771 |
| MMLU-PRO | N_plus_time | 67 | 14.876 | 3.5578 |

## 算力审计与观测前沿

C4 连接子样本只报告关联；C4 独立资源趋势不与 Benchmark 行序拼接。每月观测最大值和 90% 分位随样本量列出；稀疏月份的分位数尤其不稳定。

| variable | n | families | spearman | p_uncorrected | min | max | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_compute | 30 | 10 | 0.84535 | 4.1482e-09 | 20.522 | 24.545 | association_only_nonrepresentative_linked_subset |
| logN | 30 | 10 | 0.60454 | 0.00040269 | -0.30627 | 1.759 | association_only_nonrepresentative_linked_subset |

| type_group | month | models | families | max_score | q90_score |
| --- | --- | --- | --- | --- | --- |
| posttrained | 2024-06 | 16 | 9 | 33.358 | 28.734 |
| posttrained | 2024-07 | 8 | 6 | 29.404 | 28.782 |
| posttrained | 2024-08 | 9 | 5 | 36.879 | 33.788 |
| posttrained | 2024-09 | 17 | 6 | 46.597 | 41.051 |
| posttrained | 2024-10 | 6 | 3 | 31.82 | 30.076 |
| posttrained | 2024-11 | 11 | 2 | 34.12 | 33.55 |
| posttrained | 2024-12 | 7 | 1 | 46.889 | 43.804 |
| posttrained | 2025-01 | 41 | 5 | 41.559 | 39.06 |
| posttrained | 2025-02 | 13 | 4 | 41.919 | 39.179 |
| posttrained | 2025-03 | 13 | 2 | 41.605 | 29.942 |
| pretrained | 2024-06 | 27 | 10 | 26.728 | 24.369 |
| pretrained | 2024-07 | 1 | 1 | 5.5765 | 5.5765 |
| pretrained | 2024-08 | 2 | 2 | 6.5704 | 6.3541 |
| pretrained | 2024-09 | 6 | 3 | 38.008 | 34.98 |
| pretrained | 2024-11 | 1 | 1 | 7.2241 | 7.2241 |
| pretrained | 2025-01 | 3 | 2 | 29.483 | 24.788 |
| pretrained | 2025-02 | 1 | 1 | 17.51 | 17.51 |
| pretrained | 2025-03 | 1 | 1 | 32.436 | 32.436 |

## Loss 桥接

C6 高可比 n=7、独立家族 1。Loss 范围 [2.0933, 2.5978]，得分范围 [5.07026822083096, 6.059841492942702]。逐点留一单调映射 RMSE=0.4366，常数基线 RMSE=0.4244。这个留一不检验跨家族泛化；前沿数值桥接未识别。

## 12/24 个月条件情景

观测月度高分位参数量趋势为负，不能据此定义正增长放缓；含时间模型的未来时段留出也不优于仅规模模型。因此不计算 12/24 个月得分，避免把缺少支持的外推数值误当预测。这里的增长率是参数量口径，不是训练算力。

| horizon_months | monthly_q90_log10N_trend | positive_growth_established | time_model_improves_future_holdout | same_horizon_backtest_available | predicted_conditional_score | status |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | -0.044288 | False | False | False |  | not_identified_no_positive_growth_or_validated_time_effect |
| 24 | -0.044288 | False | False | False |  | not_identified_no_positive_growth_or_validated_time_effect |

## 图表与可复算文件

图表在 `outputs/problem4/figures/`，全部由同一脚本生成。原始合同、连接逐行理由、C8 文件审计、留出指标、分解支持和重抽样、资源情景均在 `outputs/problem4/results/`。

## 结论边界

排行榜提交不是随机实验；主结果仅覆盖许可证和权重可核的模型。严格开放预训练的晚期样本较少。粗粒度家族划分与模型重复提交可能缩小有效样本量。C4 的同名归一化连接仍应人工复核后用于正式论文。C8 与 C1 BBH 标尺不同。C6 桥接不支持前三问 Loss 向当前或未来排行榜前沿转译。长期情景没有同长度回测；不得写成已经验证的预测。
