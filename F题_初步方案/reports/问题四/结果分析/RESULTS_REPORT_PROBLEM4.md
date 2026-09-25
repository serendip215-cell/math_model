# 问题四结果：开放模型能力变化与条件情景

> 计算日期基于附件快照。预测原点是同协议 C1 的 2025-03-13，12/24 个月情景没有同长度回测，不能解读为当前实测前沿或可信外推预测。

## 数据与三个硬边界

1. **开放性：** 主样本要求 C2 权重明确为 Yes 且许可证在预设宽松集合中。这是可复算的操作性筛选，不能替代法律上的开源认证或证明提交当日已经开放。缺失权重证据不视为开放。预训练、后训练分层；合并模型不参与主分解。
2. **跨表连接：** C1/C4 只接受唯一 C4 名称、可支持的来源身份、参数量接近、发布日期可核、语言领域、Confident/Likely 且权重信息无冲突的连接。C4 给出 Hugging Face 开发者 ID 时须与 C1 命名空间一致；未给出 ID 时，同名候选在 C1 侧也须唯一。该表仅是非随机子样本，规则通过不等于人工逐项核验。
3. **Loss 桥接和未来：** C6 高可比只有同一家族的 7 个 Pythia 点；C1 可比评分截至 2025-03，不能把问题三 Loss 换算成高分前沿，也没有 12/24 个月同协议回测。

### 样本流失

| step | rows | unique_models | families |
| --- | --- | --- | --- |
| C2 raw | 4576 | 4497 |  |
| valid date/positive N/six tasks | 4561 | 4482 |  |
| earliest valid submission per model | 4482 | 4482 |  |
| permissive license | 1725 | 1725 | 284 |
| strict open weight + license | 236 | 236 | 24 |
| permissive license + unknown weight status | 1485 | 1485 | 273 |
| permissive license + explicit No weight status | 4 | 4 | 3 |
| strict pretrained | 42 | 42 | 13 |
| strict posttrained | 141 | 141 | 17 |

### C3 历史补录的口径审计

C3 的排行榜来源 4,573 条均可在 C1 按模型名找到，得分只相差两位小数舍入；C1 另有 3 条参数量缺失记录未进入 C3，名单见 `c3_c1_omitted_rows.csv`。26 条文献/报告补录记录提供更早年份，但其 `Average` 与六项任务均值并非同一计算口径。零任务分值不能自动解释为真实零分，也可能是未测。以下只量化来源覆盖和数值一致性，不把早期补录行拼接为 C1 同协议的回归或长期回测。

| source | rows | year_min | year_max | unique_models | exact_name_overlap_with_C1 | max_nearest_C1_average_difference | median_abs_average_minus_six_mean | rows_abs_discrepancy_gt_0_01 | rows_with_zero_task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open LLM Leaderboard | 4573 | 2024 | 2025 | 4494 | 4573 | 0.0049999 | 0.002491 | 0 | 332 |
| Historical (papers/reports) | 26 | 2019 | 2024 | 26 | 0 |  | 5.1333 | 26 | 23 |

早期补录逐年覆盖如下；`usable_for_C1_protocol_trend=False` 表示不能凭这些值延长可比趋势，并不否认对应模型或文献存在。逐模型差异见 `c3_historical_row_audit.csv`。

| Year | records | median_reported_average | median_six_task_mean | median_abs_discrepancy | records_with_zero_task | usable_for_C1_protocol_trend |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | 1 | 5 | 4.1 | 0.9 | 1 | False |
| 2020 | 1 | 50 | 7.6667 | 42.333 | 1 | False |
| 2021 | 2 | 3.5 | 0.99167 | 2.5083 | 2 | False |
| 2022 | 7 | 12 | 1.4167 | 6.7667 | 7 | False |
| 2023 | 13 | 12 | 3.1667 | 7.2667 | 10 | False |
| 2024 | 2 | 5.5 | 2.15 | 3.35 | 2 | False |

许可证符合预设集合但权重状态缺失的模型，与权重状态明确为 No 的模型分别计数；后者不进入‘状态未知’敏感性层。

### 提交日与 Epoch 元数据发布日期

Epoch 元数据发布日期可能对应匹配的基础模型，不能直接等同于衍生模型发布时间；晚于提交日的日期更不能视为历史可得信息。下表仅审计两日期差异，主模型的时间变量仍是排行榜提交日。

| type_group | period | models | publication_date_available | publication_after_submission | publication_over_180_days_older | median_submission_minus_publication_days |
| --- | --- | --- | --- | --- | --- | --- |
| posttrained | early | 33 | 33 | 1 | 11 | 71 |
| posttrained | late | 67 | 67 | 6 | 22 | 104 |
| posttrained | middle | 41 | 41 | 3 | 7 | 59 |
| pretrained | early | 30 | 30 | 1 | 18 | 253 |
| pretrained | late | 5 | 5 | 0 | 2 | 168 |
| pretrained | middle | 7 | 7 | 0 | 0 | 2 |

C1/C4 名称候选 109 行，规则接受 88 行，按题目附件原始预训练标签且算力可用 29 行。主分析保持附件分类，不用外部网页改写原始标签；候选和接受均不等于逐 checkpoint 核验。
这 29 条算力连接的来源、参数与算力备注已汇成 `c4_manual_review_queue.csv`；其中开发者 ID 缺失 12 条。外部来源核查另见 `c4_external_source_review.csv`，只用于敏感性和局限性说明，不替代题目附件。

### C8 逐任务核算

扫描 1954 个 JSON；解析 1954 个；模型去重后 1861 个；与 C1 连接 1855 个，其中同一叶任务集合 1855 个。C8 叶任务宏平均与 C1 BBH **数值标尺未证实相同**，因此只做覆盖与秩序审计，不强行回代相等。重复版本以记录时间及文件名排序取末份。

## 规模、时间残差与验证

模型将六任务均分映射到 logit 空间，拟合 log10 参数量与提交时间的 Ridge；超参数在训练期按模型家族 GroupKFold，并以家族均衡 RMSE 选择。2025-01 至 03 月留作时间留出，比较仅规模、规模加时间与训练均值。除按记录计算 RMSE，还按家族等权平均各家族均方误差后开方，避免提交数多的家族支配验证。时间系数吸收未观测架构、数据、后训练和选择变化，不能解释为纯技术进步。

| stratum | variant | n | families | train_n | future_holdout_n | future_holdout_families | future_holdout_rmse | future_holdout_family_balanced_rmse | future_holdout_bias | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_pretrained | N_only | 42 | 13 | 37 | 5 | 4 | 11.834 | 13.18 | -7.5294 | fit |
| strict_pretrained | N_plus_time | 42 | 13 | 37 | 5 | 4 | 12.003 | 13.131 | -5.8829 | fit |
| strict_pretrained | training_mean | 42 | 13 | 37 | 5 | 4 | 12.544 | 13.531 | -5.2019 | baseline |
| strict_posttrained | N_only | 141 | 17 | 74 | 67 | 6 | 8.8211 | 8.0189 | -4.0436 | fit |
| strict_posttrained | N_plus_time | 141 | 17 | 74 | 67 | 6 | 10.839 | 11.736 | 5.7083 | fit |
| strict_posttrained | training_mean | 141 | 17 | 74 | 67 | 6 | 9.7967 | 8.901 | -0.20245 | baseline |
| license_only_pretrained | N_only | 104 | 27 | 66 | 38 | 7 | 3.6123 | 4.6523 | -1.3093 | fit |
| license_only_pretrained | N_plus_time | 104 | 27 | 66 | 38 | 7 | 3.0815 | 4.6205 | -0.38499 | fit |
| license_only_pretrained | training_mean | 104 | 27 | 66 | 38 | 7 | 5.867 | 5.9931 | 2.3475 | baseline |

两模型在相同留出家族上的家族均衡 RMSE 差（含时间减仅规模）如下；重抽样单位为家族，区间不能验证 12/24 个月外推。

| stratum | holdout_families | delta_family_balanced_rmse_time_minus_N | family_bootstrap_ci_low | family_bootstrap_ci_high | interpretation |
| --- | --- | --- | --- | --- | --- |
| license_only_pretrained | 7 | -0.031781 | -0.93782 | 1.0308 | descriptive_model_comparison_not_independent_long_horizon_test |
| strict_posttrained | 6 | 3.7171 | -1.1127 | 7.535 | descriptive_model_comparison_not_independent_long_horizon_test |
| strict_pretrained | 4 | -0.048415 | -0.61043 | 1.0737 | descriptive_model_comparison_not_independent_long_horizon_test |

### 逐月短期留出

每折只使用目标月份开始前的模型训练，并预测下一日历月。下表同时给出训练/测试家族数、未见过的测试家族及测试参数量落在训练极值内的比例。描述性筛选预设为完整月份、测试至少 10 条且 3 个家族、参数量极值覆盖率至少 80%；这不是统计功效证明，未通过的折也保留审计。

| stratum | target_month | train_models | train_families | test_models | test_families | unseen_test_families | test_logN_within_train_minmax_fraction | full_calendar_month | descriptive_screen_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_pretrained | 2024-07 | 27 | 10 | 1 | 1 | 1 | 1 | True | False |
| strict_pretrained | 2024-08 | 28 | 11 | 2 | 2 | 1 | 0.5 | True | False |
| strict_pretrained | 2024-09 | 30 | 12 | 6 | 3 | 1 | 1 | True | False |
| strict_pretrained | 2024-10 | 36 | 13 | 0 | 0 | 0 |  | True | False |
| strict_pretrained | 2024-11 | 36 | 13 | 1 | 1 | 0 | 1 | True | False |
| strict_pretrained | 2024-12 | 37 | 13 | 0 | 0 | 0 |  | True | False |
| strict_pretrained | 2025-01 | 37 | 13 | 3 | 2 | 0 | 1 | True | False |
| strict_pretrained | 2025-02 | 40 | 13 | 1 | 1 | 0 | 1 | True | False |
| strict_pretrained | 2025-03 | 41 | 13 | 1 | 1 | 0 | 1 | False | False |
| strict_posttrained | 2024-07 | 16 | 9 | 8 | 6 | 1 | 1 | True | False |
| strict_posttrained | 2024-08 | 24 | 10 | 9 | 5 | 1 | 0.55556 | True | False |
| strict_posttrained | 2024-09 | 33 | 11 | 17 | 6 | 1 | 1 | True | True |
| strict_posttrained | 2024-10 | 50 | 12 | 6 | 3 | 2 | 1 | True | False |
| strict_posttrained | 2024-11 | 56 | 14 | 11 | 2 | 0 | 1 | True | False |
| strict_posttrained | 2024-12 | 67 | 14 | 7 | 1 | 0 | 1 | True | False |
| strict_posttrained | 2025-01 | 74 | 14 | 41 | 5 | 2 | 0.97561 | True | True |
| strict_posttrained | 2025-02 | 115 | 16 | 13 | 4 | 1 | 1 | True | True |
| strict_posttrained | 2025-03 | 128 | 17 | 13 | 2 | 0 | 1 | False | False |

通过描述性筛选的短期折如下；逐模型预测及所有可拟合折的误差保存在结果表。多个月份仍共享训练数据与家族，不能把这些折当独立重复实验，更不能充当 12/24 个月回测。

| stratum | target_month | variant | test_models | test_families | family_balanced_rmse | bias |
| --- | --- | --- | --- | --- | --- | --- |
| strict_posttrained | 2024-09 | N_only | 17 | 6 | 9.0781 | -6.6569 |
| strict_posttrained | 2024-09 | N_plus_time | 17 | 6 | 8.9205 | -4.9081 |
| strict_posttrained | 2024-09 | training_mean | 17 | 6 | 10.963 | -3.4233 |
| strict_posttrained | 2025-01 | N_only | 41 | 5 | 9.0332 | -5.4599 |
| strict_posttrained | 2025-01 | N_plus_time | 41 | 5 | 11.792 | 2.9279 |
| strict_posttrained | 2025-01 | training_mean | 41 | 5 | 8.7255 | -1.2511 |
| strict_posttrained | 2025-02 | N_only | 13 | 4 | 8.2249 | -3.2869 |
| strict_posttrained | 2025-02 | N_plus_time | 13 | 4 | 8.618 | 2.4303 |
| strict_posttrained | 2025-02 | training_mean | 13 | 4 | 9.6658 | -1.2699 |

严格后训练通过筛选 3 折，其中含时间模型家族均衡 RMSE 较低 1 折、较高 2 折；严格预训练通过筛选 0 折。只报告这种异质性，不据此估计长期趋势。
### 家族划分优先复核队列

删除一个留出家族后，时间模型与仅规模模型的家族均衡误差差值会变化。按变化绝对值排序，仅用于确定应先核对哪些基础模型关系；现有家族标签仍是名称启发式，未凭猜测新增归属。

| family_heuristic | holdout_models | example_model | full_delta_rmse | delta_without_family | absolute_influence_points | base_family_identity_review |
| --- | --- | --- | --- | --- | --- | --- |
| llama | 4 | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 3.7171 | 1.8261 | 1.8911 | pending_source_evidence |
| phi | 16 | 1024m/PHI-4-Hindi | 3.7171 | 5.1105 | 1.3934 | pending_source_evidence |
| deepseek | 2 | Sourjayon/DeepSeek-R1-8b-Sify | 3.7171 | 2.6546 | 1.0625 | pending_source_evidence |
| namespace:aidc-ai | 1 | AIDC-AI/Marco-o1 | 3.7171 | 4.4863 | 0.76918 | pending_source_evidence |
| namespace:godlikehhd | 16 | godlikehhd/alpaca_data_full_2 | 3.7171 | 4.2814 | 0.56422 | pending_source_evidence |
| qwen | 28 | 1024m/QWEN-14B-B100 | 3.7171 | 3.6945 | 0.022669 | pending_source_evidence |

### 共同规模支持

| stratum | early_n | late_n | overlap_logN_low | overlap_logN_high | overlap_early_n | overlap_late_n | overlap_early_families | overlap_late_families | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_pretrained | 30 | 5 | 0.033021 | 1.1694 | 19 | 5 | 11 | 4 | False |
| strict_posttrained | 33 | 67 | 0.063333 | 1.784 | 31 | 66 | 10 | 6 | True |
| license_only_pretrained | 41 | 38 | -0.86967 | 1.3724 | 35 | 38 | 15 | 7 | True |

只有支持审计通过的层才执行早期（截至 2024-08）与晚期（2025-01 至 03）两因素 Shapley 分解。分解数值是模型拟合变化，与实测变化另列；家族聚簇重抽样输出在 `decomposition_family_bootstrap.csv`。未通过者不报贡献百分比。

| stratum | scale_points | conditional_time_points | model_change_points | observed_change_points | unexplained_points | t0 | t1 | causal_attribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_posttrained | -4.876 | 5.768 | 0.89198 | 2.316 | 1.424 | 0.95277 | 7.4415 | False |
| license_only_pretrained | -4.074 | 1.7168 | -2.3572 | -2.6676 | -0.31038 | 0.62423 | 7.3429 | False |

家族重抽样分位数如下。严格后训练层的模型总变化区间跨 0；且时间模型没有展示稳健的未来时段留出优势。因此分解只作为探索性统计，不能形成稳定的贡献比例或已验证时间增长机制。许可证单独确认、权重状态缺失层也仅为敏感性。

| stratum | quantile | scale_points | conditional_time_points | model_change_points |
| --- | --- | --- | --- | --- |
| license_only_pretrained | 0.025 | -6.4615 | -1.424 | -6.0135 |
| license_only_pretrained | 0.5 | -3.5599 | 1.7999 | -1.7975 |
| license_only_pretrained | 0.975 | 0.46186 | 4.1 | 3.194 |
| strict_posttrained | 0.025 | -15.269 | 0.96882 | -6.0386 |
| strict_posttrained | 0.5 | -4.2183 | 5.0521 | 0.77488 |
| strict_posttrained | 0.975 | 1.8309 | 13.733 | 11.033 |

将响应面只用早晚期共同参数量支持区间内的记录重拟合，可检查全样本拟合是否由区间外记录驱动。两种拟合的符号或量级不稳时，不能解释为稳健贡献。

| stratum | overlap_fit_n | full_fit_scale_points | overlap_fit_scale_points | full_fit_time_points | overlap_fit_time_points | full_fit_total_points | overlap_fit_total_points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict_posttrained | 136 | -4.876 | -4.6431 | 5.768 | 5.3124 | 0.89198 | 0.6693 |
| license_only_pretrained | 94 | -4.074 | -4.3976 | 1.7168 | 2.2925 | -2.3572 | -2.105 |

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
| IFEval | N_only | 67 | 21.611 | 4.2556 |
| IFEval | N_plus_time | 67 | 27.386 | 16.817 |
| BBH | N_only | 67 | 15.6 | -4.8122 |
| BBH | N_plus_time | 67 | 15.816 | 3.291 |
| MATH Lvl 5 | N_only | 67 | 19.469 | -14.663 |
| MATH Lvl 5 | N_plus_time | 67 | 13.962 | 5.5261 |
| GPQA | N_only | 67 | 6.2632 | -3.128 |
| GPQA | N_plus_time | 67 | 5.6803 | -0.14553 |
| MUSR | N_only | 67 | 6.5583 | -2.7345 |
| MUSR | N_plus_time | 67 | 6.6784 | 2.4902 |
| MMLU-PRO | N_only | 67 | 14.633 | -7.6771 |
| MMLU-PRO | N_plus_time | 67 | 14.76 | 3.4108 |

## 算力审计与观测前沿

C4 连接子样本只报告关联与按家族重抽样的 Spearman 区间，不使用把相关模型当独立样本的普通 p 值；区间也不消除年代、架构等混杂。C4 独立资源趋势不与 Benchmark 行序拼接。每月观测最大值和 90% 分位随样本量列出；稀疏月份的分位数尤其不稳定。

| variable | n | families | spearman | family_bootstrap_ci_low | family_bootstrap_ci_high | family_bootstrap_valid_replicates | min | max | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_compute | 29 | 10 | 0.82892 | 0.3594 | 0.96154 | 1000 | 20.522 | 24.545 | association_only_nonrepresentative_linked_subset |
| logN | 29 | 10 | 0.58597 | 0.28912 | 0.78488 | 1000 | -0.30627 | 1.759 | association_only_nonrepresentative_linked_subset |

主表按题目附件原始口径计算。作为补充敏感性，外部来源核查把 29 条分为近似预训练算力有来源支持、外部资料提示训练阶段/算力口径不一致、证据不足三类。该分类不改写主数据。`source_supported_estimate` 不表示训练方直接测得 FLOPs；也不代表已核对 checkpoint 哈希。下表仅是来源支持子集的敏感性，仍受小样本、家族聚集、年代和架构混杂限制，不作因果推断。

| source_review_status | size |
| --- | --- |
| insufficient_evidence | 9 |
| not_comparable | 5 |
| source_supported_estimate | 15 |

| variable | n | families | spearman | family_bootstrap_ci_low | family_bootstrap_ci_high | family_bootstrap_valid_replicates | min | max | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_compute | 15 | 6 | 0.52904 | -0.17647 | 0.87614 | 999 | 20.868 | 23.468 | association_only_nonrepresentative_linked_subset |
| logN | 15 | 6 | 0.24486 | -0.24324 | 0.66675 | 1000 | -0.30627 | 1.6021 | association_only_nonrepresentative_linked_subset |

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

## 12/24 个月算力放缓条件情景与能力边界

算力情景仅用题给 C4：在权重明确开放、语言领域、来源为 Confident/Likely 且发布日期不晚于 2025-03-13 的记录中，取 2025 年截至该日的**不完整年份样本中位算力**作参考。历史增速来自 2022→2023 与 2023→2024 各年度样本中位数的比值；停滞、两种增速减半、历史增速中点是明确的条件假设，不是概率预测或真实算力上限。年度样本组成变化会影响中位数，不能把这一增速解释为同一家族的算力增长规律。

| origin | C4_anchor_cohort_year | C4_anchor_cohort_partial | C4_anchor_models_with_compute | C4_anchor_median_compute_FLOPs | C4_2022_to_2023_median_ratio | C4_2023_to_2024_median_ratio | paired_C1_C4_compute_models | paired_strict_posttrained_compute_models | source_supported_paired_models | source_supported_max_compute_FLOPs | reference_exceeds_source_supported_compute_max | observed_strict_posttrained_record_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-03-13 | 2025 | True | 22 | 5.7144e+23 | 2.0769 | 3.4472 | 29 | 0 | 15 | 2.94e+23 | True | 46.889 | cross_sectional_cohort_medians_not_validated_growth_law |

下表给出从 C1 最后同口径日 2025-03-13 起算的 12/24 个月条件算力路径。`conditional_compute_FLOPs` 是在该参考样本中位数上套用增长假设的算术结果，不是未来实际观测。

| horizon_months | scenario | annual_compute_multiplier_assumption | conditional_compute_FLOPs | cumulative_frontier_logical_lower_score | cumulative_frontier_logical_upper_score | predicted_conditional_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | stagnation | 1 | 5.7144e+23 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 12 | slow_half_low | 1.4411 | 8.2352e+23 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 12 | slow_half_high | 1.8567 | 1.061e+24 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 12 | historical_mid_reference | 2.6757 | 1.529e+24 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 24 | stagnation | 1 | 5.7144e+23 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 24 | slow_half_low | 1.4411 | 1.1868e+24 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 24 | slow_half_high | 1.8567 | 1.9699e+24 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |
| 24 | historical_mid_reference | 2.6757 | 4.0911e+24 | 46.889 | 100 |  | compute_path_conditional_score_not_identified |

能力列的下界是截至起点已观测的严格开放后训练模型累计最高分；上界 100 是六任务百分制的逻辑上限。这个区间不是置信区间，也不是具有预测力的窄界。C1/C4 可靠连接的 29 条算力—得分记录全部为预训练模型，严格开放后训练的成对记录为 0；来源进一步支持的 15 条配对记录最高算力仍低于本情景的 C4 参考算力，从起点就已越出其支持域。C1 月度高分位参数量趋势为负，含时间模型的未来时段留出也未胜过仅规模模型。因此附件尚不能识别算力放缓对后训练能力前沿的数值影响，`predicted_conditional_score` 保持空值。图 `q4_compute_scenario_identification.pdf` 分开展示资源假设和能力识别范围，不绘制虚构的能力预测曲线。

## 图表与可复算文件

图表在 `outputs/problem4/figures/`，全部由同一脚本生成。原始合同、连接逐行理由、C8 文件审计、留出指标、分解支持和重抽样、资源情景均在 `outputs/problem4/results/`。

## 结论边界

排行榜提交不是随机实验，也不等于模型发布日期；主结果按题目附件原始字段计算，仅覆盖许可证和权重可核的模型。严格开放预训练的晚期样本较少。粗粒度家族划分与模型重复提交可能缩小有效样本量。外部来源核查只作补充敏感性，不改写主模型标签；也未取得逐 checkpoint 身份或完整训练遥测。C8 与 C1 BBH 标尺不同。C6 桥接不支持前三问 Loss 向当前或未来排行榜前沿转译。长期情景没有同长度回测；不得写成已经验证的预测。
