# F 题问题一计算结果

## 运行环境与安全边界

- Python 3.12.10，随机种子 `20260923`。
- 数据仅从 `F题_清洗后/A_data_value` 读取。
- A1 文本字段只按数据处理，不执行其中任何指令；PDF 隐藏文字未进入程序、特征或参数。
- 8 个列表字段按原始数据集可核验定义处理：二分类 logits 转正类概率（`ad_en` 的正类为“无广告”），PRRC 六级 logits 转期望等级，QuRating 四维分量单调压缩后汇总。

## 数据读取与预处理

- A1–A3 共读取 272,505 条记录，无效 JSON 行 0 条。
- 文件内重复 ID 0 条；跨文件重复 ID 数 11419。
- 质量标尺只由 A1 的 1%–99% 分位确定，A2/A3 沿用同一标尺。
- 三个长度/规模型指标按 A1 的稳健中心构造适宜度，其余指标按收益型或成本型统一为“越高越好”。完整规则见 `quality_metric_rules_weights.csv`。

## 质量评价与冲突

- 主评分为四个语义组的 Huber 稳健位置，平均值 0.6047，标准差 0.1007。
- A1 冲突阈值取冲突度的 90% 分位：0.2714；同一阈值直接用于 A2/A3。
- Huber 参数与等权对照的稳定性见 `quality_sensitivity.csv`。

A1 领域质量中位数最高的三个领域：

| domain | n | quality_median | median_ci_low | median_ci_high | conflict_rate |
| --- | --- | --- | --- | --- | --- |
| arxiv | 1419 | 0.7819042123798114 | 0.7790084871478854 | 0.786699379555381 | 0.028188865398167725 |
| stackexchange | 10000 | 0.7145557992598962 | 0.713209624259901 | 0.7158316480547543 | 0.0149 |
| commoncrawl | 9640 | 0.7067081635779496 | 0.7046859285499507 | 0.7082161972412615 | 0.07686721991701245 |

A1 领域质量中位数最低的三个领域：

| domain | n | quality_median | median_ci_low | median_ci_high | conflict_rate |
| --- | --- | --- | --- | --- | --- |
| book | 171 | 0.5693637447993978 | 0.5519738578053145 | 0.590701043184157 | 0.1111111111111111 |
| github | 10000 | 0.5768577193890864 | 0.5749264101038564 | 0.5786094136094166 | 0.0216 |
| wikipedia | 10000 | 0.6215107447610511 | 0.6192693831183744 | 0.6236722955243265 | 0.1082 |

这些分数是本数据集、当前指标体系下的相对评价，不等同于实际训练收益。A2/A3 与 A1 共同域的差异应结合抽样设计解释，不能仅凭均值断言总体质量变化。

A2/A3 扩展样本复核：

| domain | a1_n | extension_n | a1_quality_median | extension_quality_median | median_difference_extension_minus_a1 | a1_conflict_rate | extension_conflict_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arxiv | 1419 | 17523 | 0.7819042123798114 | 0.7765822239050421 | -0.00532198847476939 | 0.028188865398167725 | 0.02197112366603892 |
| github | 10000 | 203752 | 0.5768577193890864 | 0.5760010823373123 | -0.0008566370517740785 | 0.0216 | 0.022453767324983314 |

冲突消解不仅使用 Huber 稳健位置，还按领域拆解主导冲突方向和指标驱动。c4 的高冲突率为 0.2876，其主要成因如下：

| pair | count | share_among_domain_high_conflict | dominant_orientation | orientation_share | high_driver_metric | low_driver_metric |
| --- | --- | --- | --- | --- | --- | --- |
| 语言表达—洁净规范 | 1403 | 0.4878303198887344 | 洁净规范高—语言表达低 | 1.0 | rps_lines_ending_with_terminal_punctution_mark | modernbert_professionalism |
| 知识价值—洁净规范 | 715 | 0.24860917941585536 | 洁净规范高—知识价值低 | 1.0 | rps_lines_ending_with_terminal_punctution_mark | fineweb_edu |
| 语言表达—结构丰富 | 709 | 0.24652294853963838 | 结构丰富高—语言表达低 | 1.0 | rps_doc_mean_word_length | modernbert_professionalism |

完整领域画像见 `conflict_cause_profile.csv`；这里的“成因”表示指标结构上的直接证据，不作文本语义因果推断。

## 质量评分到配比模型的接口

A16 共提供 17 个配方域映射，其中 6 个 direct/near_direct 域可从 A1 获得 Q，剩余 11 个 inferred 域没有可观测 Q。程序对可映射域构造“已映射配比覆盖率”和“覆盖部分的加权平均 Q”，并与不含 Q 的模型做同折交叉验证和 A6–A7 独立检验；没有为 inferred 域主观填值。

| model | quality_features | cv_mse | test_1m_macro_rmse | test_1m_pooled_rmse | selected |
| --- | --- | --- | --- | --- | --- |
| linear_ilr_ridge | False | 0.19437001203884102 | 0.25889686473009 | 0.3994517460381332 | False |
| quadratic_ilr_ridge | False | 0.14714938872049638 | 0.23194974951806316 | 0.33256036462835376 | True |
| quadratic_ilr_ridge_quality | True | 0.1360480723557521 | 0.2344268737651138 | 0.33118767075914224 | False |

质量增强模型虽然内部交叉验证 MSE 更低，但只有在内部 MSE 和独立 1M 平均 Loss RMSE均至少改善 2% 时才采用。本次 `quality_features_adopted=False`，因此 Q 已进入可检验的消融链路，但不强行进入最终预测器。

## 配比模型

- 候选模型：标准化后的 ILR 线性 Ridge、ILR 二阶 Ridge 和质量增强二阶 Ridge；仅按 A4–A5 内部五折交叉验证及上述 Q 外部增益规则选择 `quadratic_ilr_ridge`，A6–A7 保留为独立的 1M 检验集。
- 训练内部交叉验证 MSE：线性 0.194370，二阶 0.147149；二阶改善超过预设 2% 门槛。
- A6–A7 平均 Loss RMSE：线性 0.258897，二阶 0.231950；13 域合并 RMSE 分别为 0.399452、0.332560。
- 主推荐采用实测标准化 Loss 最优前 10% 配方的平均值，总和为 1.000000000000，最大单域占比 0.1513；5%/10%/20% 阈值和 Bootstrap 区间见 `recommendation_stability.csv`。
- 响应面诊断解另在 10%--90% 分量范围和 ILR 距离 90% 支持域内做 19/20 次多起点优化。它仍有 10 个活跃分量边界，故不作为主推荐。

各尺度的平均 Loss 误差（每个配方先对 13 域求均值，再计算指标）：

| split | n | mae | rmse | bias | centered_rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| train_1m | 512 | 0.15142551252442282 | 0.19924164981074768 | 1.214306433183765e-16 | 0.19924164981074768 | 0.797726157574301 |
| test_1m | 256 | 0.1802350859199723 | 0.23194974951806316 | 0.061775325240420176 | 0.22357212593016276 | 0.6658875028610665 |
| test_60m | 256 | 1.52114611392869 | 1.5364108232559037 | 1.52114611392869 | 0.2160386953708255 | 0.6257689116502632 |
| test_1B | 64 | 2.750128505890895 | 2.7544219241037964 | 2.750128505890895 | 0.15373137958781147 | 0.5265567765567765 |
| est_10b | 63 | 3.558691282764712 | 3.5758744849997415 | 3.558691282764712 | 0.35013524022385073 | -0.5931739631336406 |
| est_70b | 63 | 3.941604347477777 | 3.958260861750049 | 3.941604347477777 | 0.3627453895041717 | -0.6668106758832566 |

上表的 `centered_rmse` 对每个尺度的平均 Loss 去均值后计算，仅衡量平均配比效应能否迁移；它不等同于经校准的绝对预测成绩。逐域误差汇总另见 `mixture_model_metrics.csv` 中的 `pooled_domain` 行：1M 检验集的 13 域合并 RMSE 为 0.3326，与平均 Loss RMSE 不是同一指标。10B、70B 表中的 Loss 是由较小尺度外推的估计量，不能当作独立真实测试成绩。

核心判断：1M 检验集的平均 Loss RMSE 为 0.2319、排序相关为 0.6659；跨到 60M 和 1B 后，去均值的平均 Loss RMSE 分别为 0.2160 和 0.1537，说明配比效应只能部分迁移。10B、70B 的平均 Loss 排序相关为 -0.5932、-0.6668，方向已经反转，因此**不支持把 1M 响应面直接用于大尺度绝对预测或定案配比**。

稳健实证建议占比最高的五个领域：

| domain | recommended_share | model_optimal_share | best_observed_share | training_mean_share |
| --- | --- | --- | --- | --- |
| pile_cc | 0.15132765428752817 | 0.30263879868167737 | 0.05189620758483034 | 0.11948759779188561 |
| wikipedia_en | 0.11145182341300443 | 2.9336824555888725e-06 | 0.13572854291417166 | 0.07507775607983627 |
| pubmed_central | 0.1039199304425759 | 0.0014943855625005012 | 0.04391217564870259 | 0.11375419105058357 |
| stackexchange | 0.08542976880316853 | 0.11164681286903487 | 0.10479041916167664 | 0.09943242344413017 |
| uspto_backgrounds | 0.07989060608789102 | 0.2669265954074144 | 0.13672654690618763 | 0.07556475309333588 |

主推荐是 52 个实测优良配方的质心，降低了单个最优样本的偶然性，也避免二阶响应面把大量权重推到经验边界。`model_optimal_share` 只展示模型在收紧支持域后仍存在的边界倾向；两种配比都只能作为下一轮实验候选。

## 多尺度补充模型

补充模型使用 ILR 配比、标准化对数规模、规模二次项和“ILR×规模”交互。它用 train_1m、test_60m、test_1B 拟合后在独立 test_1m 上检查配比泛化，并对 60M、1B 另做留一尺度检验。A12–A15 仍仅作估计压力测试。

| evaluation | role | n | rmse | bias | centered_rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| test_1m | external_composition_test | 256 | 0.2578297549443498 | 0.05961951829933343 | 0.2508419733067393 | 0.4361374170290683 |
| test_60m | leave_one_scale_out | 256 | 0.22965325119365165 | 0.0463245741476119 | 0.224932544585831 | 0.17594558251316086 |
| test_1B | leave_one_scale_out | 64 | 0.3997078648754044 | -0.39143467819668787 | 0.08090284264665833 | 0.42078754578754574 |
| est_10b | estimated_pressure_test | 63 | 1.2115936493284913 | -1.2028930013192494 | 0.1449399823040552 | -0.04891513056835637 |
| est_70b | estimated_pressure_test | 63 | 2.437549620642795 | -2.4325131862108322 | 0.156613383866952 | 0.0771889400921659 |

该模型显著修正绝对 Loss 的尺度截距，但留一尺度误差和 10B/70B 排序仍不稳定，因此作为跨尺度校准补充，不替换 1M 主响应面，也不用于生成最终配比。

## 单域效应与组合解释

`mixture_marginal_effects.csv` 记录从训练均值配比出发，将某域增加 1 个百分点、其余域按比例缩减后的预测 Loss 变化。预测 Loss 降幅最大的三个方向为：

| domain | delta_share | delta_macro_loss | max_abs_domain_loss_change |
| --- | --- | --- | --- |
| ubuntu_irc | 0.01 | -0.006253370385368799 | 0.07686170773961187 |
| dm_mathematics | 0.01 | -0.005982293967158069 | 0.09465717214824032 |
| stackexchange | 0.01 | -0.0012216947090239161 | 0.010804063986753043 |

该效应依赖基准配比，不能解释成全局因果系数。二阶模型仅在 A4–A5 内部五折交叉验证 MSE 至少改善 2% 时采用，防止用检验集选择模型。

## 约束与一致性校验

- 配比表与 Loss 表按 `index` 一对一合并；每个文件的行数、配比和范围见 `mixture_data_audit.csv`。
- 所有输入配比非负，归一化后严格满足和为 1。
- 推荐配比通过单纯形和 ILR 支持域回代检查。
- 质量指标方向、缩放边界和权重均可在结果表追溯。

## 增强验证（problem1_enhancement.py，6 模块）

以下数字为 2026-09-23 实际运行输出，产物在 `outputs/problem1/enhancement/`。

**模块 1：质量外部效度锚点**（配方加权质量 ΣpᵢQᵢ 与实测 macro Loss 的 Pearson 相关）

| 样本 | r | p |
| --- | --- | --- |
| 全部 train_1m 配方 | 0.0297 | 0.503（不显著） |
| Q 覆盖 >30% 的配方 | **0.3566** | 1.813e-13 |
| test_1m（独立检验） | **0.1793** | 4.001e-03 |

如实结论：Q 锚点仅在覆盖充分的配方上成立，全样本相关不显著（11 个域无 Q 稀释）。

**模块 2：R² 与 Δφ 差分相关**

> **更正（2026-09-23）**：早期版本用 `R² = 1 − RMSE²` 计算显式 R²，该公式仅在目标方差为 1 时成立，
> 而 macro Loss 原始单位方差 train=0.1055、test=0.0774，公式误用导致 R² 被严重高估（test 报出 0.946）。
> 正确口径 R² = 1 − SSE/SST = 1 − MSE/Var（Var 为该 split 的总体均方离差，ddof=0），修正表如下，
> 明细见 `corrected_r2.csv`。**代码已同步修正（2026-09-23）**：`problem1_enhancement.py`
> 模块 2 改为从各 split 原始数据计算方差（ddof=1），重跑后 6 尺度 R² 与下表完全一致；
> `wrong_r2_1minus_rmse2` 列保留作错误记录，不得引用。60M 及以上的去均值 R² 近零或为负：
> 大尺度 macro Loss 本身方差极小
> （1B 仅 0.0027），1M 拟合的效应幅度在绝对 MSE 意义下不再匹配，这正是第 8 节置信衰减建模的动机。

| split | RMSE（原始单位） | 去均值 RMSE | macro Loss 方差 | 真实 R²（原始） | 真实去均值 R² |
| --- | --- | --- | --- | --- | --- |
| train_1m | 0.1992 | 0.1992 | 0.1053 | **0.6230** | 0.6230 |
| test_1m | 0.2320 | 0.2236 | 0.0771 | **0.3023** | 0.3519 |
| test_60m | 1.5364 | 0.2160 | 0.0479 | — | 0.0258 |
| test_1B | 2.7544 | 0.1537 | 0.0027 | — | −7.7603 |
| est_10b | 3.5759 | 0.3501 | 0.0092 | — | −12.2706 |
| est_70b | 3.9583 | 0.3627 | 0.0108 | — | −11.1291 |

论文口径统一建议：1M 检验集报告 macro RMSE 0.232、Spearman 0.666、R²≈0.31（相对检验集方差）；
60M/1B 不报 R²，只报去均值 RMSE（0.216/0.154）与 Spearman（0.626/0.527）。

Δφ 差分相关量级小（train ΔL vs Δφ r=0.020；1B 的 |ΔL| vs ‖Δφ‖ r=0.290 最大），仅作辅助视角。

**模块 3：逐域消融**（基线：RMSE 0.3203，Spearman 0.6475）

| 移除域 | RMSE 变化 | Spearman 变化 | 判读 |
| --- | --- | --- | --- |
| github | **+7.59%**（0.3446） | −0.1204 | 刚性域 |
| arxiv | **+5.27%**（0.3372） | −0.0849 | 刚性域 |
| pile_cc | +0.66%（0.3224） | −0.0369 | 质心法已知特性：Loss 方差小的域自然聚集于最优配方 |
| wikipedia_en | −1.43%（0.3157） | −0.0114 | 训练集过拟合迹象 |
| stackexchange | −6.01%（0.3010） | −0.0130 | 训练集过拟合迹象 |

**模块 4：跨尺度推荐质心**（各尺度实测最优前 10% 配方质心，top-5）

| 尺度 | 前 5 域（占比） |
| --- | --- |
| train_1m | pubmed_central 0.113 / stackexchange 0.111 / pubmed_abstracts 0.103 / pile_cc 0.100 / wikipedia_en 0.084 |
| test_60m | stackexchange 0.149 / pile_cc 0.128 / freelaw 0.091 / pubmed_central 0.082 / wikipedia_en 0.078 |
| test_1B | **pile_cc 0.211** / pubmed_central 0.153 / gutenberg_pg_19 0.098 / github 0.088 / stackexchange 0.082 |
| est_10b / 70b | freelaw 0.260 / pubmed_central 0.256 / arxiv 0.197 / wikipedia_en 0.092 / uspto 0.066 |

pile_cc 在三个实测表的样本内优良质心中由 10.0% 增至 12.8%、21.1%。这只能说明题给候选配方中的经验质心随尺度变化，不能据此断言“质量越高，规模越大权重越高”，也不能把 10B/70B 估计表的质心作为部署配比。

**模块 5：冲突敏感性**（见 `conflict_sensitivity.csv`，统一阈值方案最合理）。

**模块 6：指标噪声 vs 冲突**：平均变异系数与冲突率 Pearson r=−0.0407（p=0.931）——
冲突根因是语义组间意见分歧，不是指标噪声。变异系数最高的域：github 1.142、wikipedia 0.846、commoncrawl 0.730。

## inferred 域 Q 补全（inferred_quality_proxy.py）

11 个 inferred 域用两种代理补全 Q（方法 A：train_1m 偏相关反演；方法 B：A16 语义映射；两法平均），
17 域完整 Q 表见 `outputs/problem1/inferred_quality/`。代表性结果：

| domain | 方法 A（反演） | 方法 B（语义） | 组合 Q |
| --- | --- | --- | --- |
| freelaw | 0.402 | 0.569 | 0.485 |
| pubmed_central | 0.400 | 0.782 | 0.591 |
| dm_mathematics | 0.546 | 0.782 | 0.664 |
| philpapers | 0.786 | 0.782 | 0.784 |
| uspto_backgrounds | 0.414 | 0.707 | 0.561 |

**探索性质量代理消融**（简化版模型：alpha 固定 10.0、macro 目标）：

| 模型 | test_1m RMSE | Spearman |
| --- | --- | --- |
| baseline quadratic ILR Ridge | 0.23679 | 0.66237 |
| + 完整 17 域 Q | **0.22538** | 0.69604 |

该版本 RMSE 改善 **4.815%**。但 11 个 inferred 值部分由 train_1m 的 Loss 反演，属于监督式 Loss 代理，不是由 A1 质量指标独立测得的 Q；语义近邻映射也包含人工假设。因此该结果只证明“增加一个由训练 Loss 构造的低维代理可能改善预测”，不能证明质量评分带来独立增益。正式主模型仍以 A16 可核验的 6/17 域映射消融为准，并保持 `quality_features_adopted=False`。

## 标度律桥接（scaling_law_bridge.py）

**纯 D_eff 模型**：a≈0.0000、c=19.0648、β=0.0948，实测尺度拟合 RMSE 0.3391——
配比项在纯规模标度律中贡献趋零，说明规模项与配比项必须分开建模。

**组合模型**（尺度基线 + ILR 配比效应；基线 c=25.2412、β=0.1111）：

| scale | Spearman | 判定 |
| --- | --- | --- |
| train_1m | 0.7974 | 实测，最强 |
| test_1m | 0.6624 | 实测 |
| test_60m | 0.6248 | 实测，开始衰减 |
| test_1B | 0.4777 | 实测，明显衰减 |
| est_10b | **−0.5890** | 方向反转 |
| est_70b | **−0.6634** | 方向反转 |

方向反转是跨尺度失效的直接证据，由下一节的效应标度律解决。

## 配比效应标度律 + 置信衰减（scaling_effect_decay.py）

- **λ\*=0.75**（留一尺度 CV 选出的斜率收缩因子）；**κ 中位 = 0.1503**（效应幅值衰减指数）；
- 置信权重 w(N)=(N/1e9)^(−κ)：w(10b)=0.7075、w(70b)=0.5282；
- 逐域 κ：dm_mathematics 0.448（衰减最快）、arxiv 0.303、pubmed_central 0.287、freelaw 0.046、wikipedia_en 0.012（几乎不衰减）。

**变体对比**（est 尺度仅作外部检验，零参与拟合；指标为 RMSE/Spearman）：

| 变体 | est_10b Spearman | est_70b Spearman | est_10b RMSE | est_70b RMSE |
| --- | --- | --- | --- | --- |
| frozen_1B（不外推） | −0.0837 | −0.0977 | 0.1645 | 0.1732 |
| extrap_trend（趋势外推） | **+0.0213** | **+0.0712** | 0.1794 | 0.2027 |
| decay_fusion（置信衰减） | −0.0837 | −0.0977 | **0.1373** | **0.1333** |
| baseline_only（纯规模） | — | — | 0.0991 | 0.1104 |

趋势外推在实测尺度同样去噪：train_1m Spearman 0.117→0.448，test_1m −0.013→0.368，
test_60m −0.012→0.120，test_1B 0.817。est 尺度反转幅度从主模型直推的 −0.59/−0.67
减弱至 ±0.1 以内（decay_fusion 仍为 −0.08/−0.10，未完全转正），绝对误差向保守基线收敛。

**结论表述**：1M–1B 实测范围内二次响应面排序相关保持为正但随尺度递减
（0.80→0.63→0.53）；10B/70B 排序强度接近零，无实用排序精度；精确配比需问题二的完整标度律框架。

## 三项稳健性升级（refinement_three.py）

产物在 `outputs/problem1/refinement/`。

**升级 1：Loss-informed Q 代理的嵌套折外评估**——外层五折中，每一折都只用其余四折重新估计 11 个缺失域代理，并只在留出折上预测。折外 baseline RMSE/Spearman 为 0.2857/0.5846，加入代理后为 0.2506/0.6812；用完整训练集估计代理后，在独立 test_1m 上由 0.2368/0.6624 改善为 0.2078/0.7339。该增益在预测层面成立，但代理本身由训练 Loss 构造，故仍不解释为“独立质量评分的增益”，也不替换主模型的可观测 Q 结论。

**升级 2：冲突降权消解闭环**——A1 上对比原始中位数 / 降权加权（w=1−K）/ 惩罚式（Q×(1−0.5K)）：
三种口径域排序 Spearman 全部 **1.0000**；逐域数值最大变化降权 1.10% vs 惩罚 11.65%。
结论：消解动作不改变任何域级结论，Q 对冲突定义稳健；惩罚式扭曲数值 10 倍却不改变排序，只作敏感性变体。

**升级 3：质心正则化路径**——信任域扫描 `min_z pred_quad(z) + λ‖z−z_c‖²`：

| λ | 到质心距离 | 最大占比 | 模型预测（原单位） | 最近实测配方真实 Loss |
| --- | --- | --- | --- | --- |
| 0（边界解） | 40.79 | 0.363 | 1.93 | 5.12 |
| 0.03 | 1.93 | 0.145 | 5.05 | 4.98 |
| 1.0 | 0.051 | 0.150 | 5.11 | 4.98 |
| ∞（=质心） | 0 | 0.151 | 5.11 | **4.98** |

解读：①边界解预测 1.93 属外推幻象（离最近实测配方 33.9 ILR 单位，其邻域实测 5.12）；②λ≥0.03 立即收敛到质心邻域，响应面在质心附近平坦；③决定性证据——质心邻域实测 4.98 优于边界解邻域 5.12。保守性代价 −1.28σ 全部来自外推区不可信预测，实测对照下质心反而更优。

## 输出文件

- `outputs/problem1/quality/results/quality_sample_scores.csv`：样本质量、冲突度和主要冲突对。
- `outputs/problem1/quality/results/quality_domain_summary.csv`：领域质量、区间及冲突率。
- `outputs/problem1/quality/results/quality_metric_rules_weights.csv`：方向、缩放规则和 CRITIC 组内权重。
- `outputs/problem1/quality/results/conflict_pair_summary.csv`：主要冲突对。
- `outputs/problem1/quality/results/conflict_cause_profile.csv`：分领域冲突方向和指标驱动。
- `outputs/problem1/mixture/results/mixture_model_metrics.csv`、`mixture_predictions.csv`：训练、检验和外推结果。
- `outputs/problem1/mixture/results/quality_domain_mapping.csv`、`quality_integration_ablation.csv`：A16 映射与 Q 增益检验。
- `outputs/problem1/mixture/results/scale_aware_metrics.csv`：多尺度模型的独立及留一尺度检查。
- `outputs/problem1/mixture/results/recommended_mixture.csv`、`recommendation_stability.csv`：稳健候选配比和区间。
- `outputs/problem1/enhancement/results/`：外部效度锚点、显式 R²、逐域消融、跨尺度质心、冲突敏感性、指标变异系数。
- `outputs/problem1/inferred_quality/`：17 域完整 Q 表（方法 A/B/组合）与质量增强消融。
- `outputs/problem1/scaling_bridge/`：纯 D_eff / 组合模型逐尺度指标、效应标度律参数与变体对比。
- `outputs/problem1/refinement/`：交叉拟合消融、冲突消解对比、正则化路径表。
- `outputs/problem1/figure_manifest.csv`：每张保留图的数值来源和解释边界。
- `outputs/problem1/quality/figures/*.pdf`、`mixture/figures/*.pdf`、`enhancement/figures/*.pdf`：经口径审计后的论文矢量图。

## 可复现运行方式

```powershell
# 虚拟环境：match_model/.venv（Python 3.12.10）
python F题_初步方案/code/problem1/problem1.py                # 主流程：质量评分 + 配比模型 + 推荐配比
python F题_初步方案/code/problem1/problem1_enhancement.py    # 增强验证 6 模块
python F题_初步方案/code/problem1/inferred_quality_proxy.py  # inferred 域 Q 补全 + 消融
python F题_初步方案/code/problem1/scaling_law_bridge.py      # 标度律桥接（纯 D_eff / 组合模型）
python F题_初步方案/code/problem1/scaling_effect_decay.py    # 效应标度律 + 置信衰减
python F题_初步方案/code/problem1/refinement_three.py        # 三项稳健性升级
python F题_初步方案/code/problem1/problem1_figures.py        # 最后运行：生成论文图并清理被替代旧图
```

## 图表口径审计

- 删除六尺度原始预测散点和原始 RMSE 柱状图：它们把 1M 响应面直接用于不同 Loss 尺度，主要展示截距错位，容易被误读成配比效应误差。
- 删除含 10B/70B 的“跨尺度推荐”堆叠图：这两个表的 Loss 是估计量，不作为实测推荐证据。新版只画 1M、60M、1B 样本内经验质心热图。
- 删除部分 Q 覆盖下的简单相关散点：未归一化覆盖份额会混入配比覆盖率，不能单独支撑 Q 的外部效度。
- 推荐图加入 Bootstrap 95% 区间，并把响应面解明确标为诊断解；跨尺度图只比较去均值误差和排序相关，估计区间另行着色。
- 所有论文图取消图内大标题；标题和完整解释交给论文 caption。保留清单、数据来源和解释限制见 `outputs/problem1/figure_manifest.csv`。
