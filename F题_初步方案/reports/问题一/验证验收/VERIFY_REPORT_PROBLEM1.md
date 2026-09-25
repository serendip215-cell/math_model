# 问题一模型改进与核验（2026-09-23）

## 结论

**WARN：主模型计算和约束通过内部一致性检查，13 张保留图已重绘并实际渲染；质量外部效度和候选配比的真实训练效果仍待实验。** 1M 独立检验平均 Loss RMSE 为 **0.231950**，Spearman 为 **0.6659**。主推荐最大单域占比为 **0.1513**，不采用响应面边界解。

## 对反馈的处理

| 反馈 | 处理 | 结论 |
| --- | --- | --- |
| 质量评价与配比模型脱节 | 读取 A16；只对 6 个 direct/near_direct 域使用观测 Q，11 个 inferred 域不主观填值；增加同折交叉验证及独立 1M 消融 | Q 模型 CV MSE 从 0.147149 降到 0.136048，但独立 1M 平均 Loss RMSE 从 0.231950 升到 0.234427，故主预测器不含 Q。 |
| 推荐配比过于极端 | 响应面诊断解收紧到分量 10%--90% 和 ILR 距离 90% 支持域；主推荐改为实测 Loss 最优前 10% 共 52 个配方的质心 | 主推荐前三项为 pile_cc 0.1513、wikipedia_en 0.1115、pubmed_central 0.1039；响应面诊断解仍有 11 个活跃边界，只用于展示风险。 |
| 冲突成因偏薄 | 增加逐域冲突方向、组间差及指标驱动表和领域冲突率图 | c4 冲突率 0.2876；48.78% 为“洁净规范高—语言表达低”，主要表现为终止标点规范得分高而专业表达得分低。 |
| 单尺度模型跨尺度不足 | 增加 ILR、对数规模、规模二次项和 ILR×规模交互的补充模型，做独立 1M 与留一尺度检验 | 尺度截距改善，但 10B/70B 估计集 Spearman 仅 -0.0489/0.0772，不能支撑可靠远尺度排序。 |

固定域级 Q 与配比的乘积通常只是配比的重参数化；A16 也没有给 11 个 inferred 域的质量值。参考文档中“按语义推断 Q”和“近邻统一乘 0.85”缺少当前数据的识别依据，本次没有采用。当前输出明确回答了“是否引入 Q”：已经映射和消融，但没有获得独立检验增益，因此不进入最终预测器。

反馈中“uspto_backgrounds 的 best_observed_share=0”与当前原始 A4–A5 不符。重新按实测标准化 Loss 定位最佳训练配方并归一化后，该占比为 0.136727；原极端推荐的问题成立，但不能用“训练中从未出现”作为依据。

## 核验范围和依据

- 按 `2analysis-modeling`、`3coding-visual` 与 `6verity` 数模技能检查题意对应关系、输入、模型选择、结果表和全部 13 张保留 PDF。
- 输入为 A1–A3 的 272,505 条质量信号，以及 A4–A15 的配比和 Loss 表。对三个质量源文件重新计算 SHA-256，均与 `quality_ingest_audit.json` 一致；JSON 解析错误 0 条，文件内重复 ID 0 条。
- 重新运行 `code/problem1/problem1.py`，并独立从 `mixture_predictions.csv` 计算各尺度平均 Loss RMSE；与 `mixture_model_metrics.csv` 和 `run_summary.json` 逐项一致。质量分数 272,505 条均在 [0,1] 内；组内权重各自求和为 1，全局权重求和为 1。
- 从原始 A4–A5 重新计算标准化实测目标并取前 10% 配方，所得 17 维质心与 `recommended_mixture.csv` 的主推荐逐分量一致（最大绝对差小于 1e-12）；配比非负且和为 1。A16 可观测映射为 6/17，冲突画像各类型占比按域求和为 1。

## 已发现并修正的问题

| 问题 | 修正与复核 |
| --- | --- |
| 原 `macro_average` 的 RMSE 将 13 个验证域误差直接合并，而 Spearman 用逐配比平均 Loss；指标名称和散点图口径不一致 | `macro_average` 改为先对每个配比的 13 域 Loss 求平均，再算误差；另存 `pooled_domain`。改进后 1M 检验集分别为 **0.231950** 和 **0.332560**。 |
| 原代码用 A6–A7 检验误差挑选线性或二阶模型，再报告同一检验集成绩 | 改用标准化特征和 A4–A5 内部五折交叉验证选型。线性 MSE 0.194370，二阶 0.147149，仍选二阶；A6–A7 用于独立评估。 |
| 推荐图中的“最佳实测训练配比”原按模型预测目标挑选 | 改为用 A4–A5 **实测** 13 域标准化目标挑选，并对原始 Loss 独立复算。 |
| 原质量分布图把 A1 七个领域混合分布与仅包含 arxiv/github 的扩展集比较 | 改为 arxiv、github 两个同域面板，使用共同分箱和密度刻度。 |
| 推荐配比图图例可能压住最下方类别 | 图例移到图面上方，重新渲染检查。 |

## 当前结果的合理性

- A1 arxiv 质量中位数 0.7819，A2 arxiv 0.7766；A1 github 0.5769，A3 github 0.5760。图中的相同领域分布与这些数值方向一致。跨文件重复 ID 有 11,419 个；各数据集分别汇总，未将扩展集直接拼入 A1 总体。
- 1M 检验集平均 Loss RMSE 为 **0.231950**，Spearman 为 **0.665888**；模型有一定排序能力，但仍不完美。10B、70B 表来自估计 Loss，主模型排序仍反转，不能作为独立真实验证。
- 主推荐前三项为 pile_cc 0.1513、wikipedia_en 0.1115、pubmed_central 0.1039，最大占比 0.1513。响应面诊断解仍有 11 个活跃边界，说明将其降级为诊断是必要的。
- A1 原文分层抽样 49 条已完成 ID、领域和文本长度回查，均与评分表对应；六条可见文本案例显示评分的解释边界。详见 `RAW_TEXT_AUDIT_PROBLEM1.md`。这不是独立专家盲评。

## 图像核验

13 张 PDF 均为非空单页矢量图，已实际渲染检查中文、坐标轴、图例、内容和边界，无明显乱码或裁切。所有论文图取消图内大标题；每张图的数据来源和解释边界记录在 `outputs/problem1/figure_manifest.csv`。

| 图 | 核验重点 | 结果 |
| --- | --- | --- |
| `conflict_profile_by_domain.pdf` | 左侧冲突率、右侧冲突类型构成均由领域级结果表直接生成 | PASS |
| `quality_by_domain.pdf` | A1 七领域中位数排序及区间与汇总表一致 | PASS |
| `quality_group_correlation.pdf` | 四组对称相关矩阵，标注值与色阶方向一致 | PASS |
| `quality_score_distribution.pdf` | A1 与 A2/A3 按同一领域比较，不跨域混合 | PASS |
| `mixture_prediction_1m.pdf` | 只展示训练尺度和独立 1M 检验；RMSE、Spearman、正确 R² 与结果表一致 | PASS |
| `quality_integration_ablation.pdf` | 内部 CV 与独立 test_1m 分面，清楚显示 6/17 可观测 Q 未通过双门槛 | PASS |
| `cross_scale_transfer.pdf` | 只用去均值 RMSE 和 Spearman 比较迁移，10B/70B 估计区间着色 | PASS |
| `recommended_mixture.pdf` | 经验质心、Bootstrap 95% 区间、训练均值和响应面诊断解与结果表一致 | PASS |
| `mixture_marginal_effects.pdf` | 正负效应及“增加 1 个百分点”的单位说明正确 | PASS |
| `empirical_centroid_by_observed_scale.pdf` | 仅含 1M/60M/1B 实测表，展示全部 17 域而非选择性 top-5 | PASS |
| `domain_ablation.pdf` | 用变化量作图并注明标准化目标口径，避免与原单位 RMSE 混淆 | PASS |
| `estimated_scale_effect_decay_check.pdf` | 明示估计表上纯规模基线 RMSE 更低，含配比变体排序接近零 | PASS |
| `recommendation_regularization_path.pdf` | 响应面预测与最近实测邻域同时展示，外推幻象可见 | PASS |

## 尚未完成的验证

1. 质量分数是 22 项信号构成的**相对指标**；缺少独立人工盲评和真实训练收益对照，不能认定领域排名具有外部效度。
2. inferred 的 11 个 Pile 域缺少同口径质量信号。若后续能对 A18 文本运行与 A1 相同的 22 个评分器，才适合重评完整 17 域 Q 增益。
3. 主推荐已有 Bootstrap 区间且比边界优化稳健，但未经过按该配比重新训练模型的实验，只能作为下一轮候选。
4. 组合联合扰动、预测区间覆盖率和 A1 全文独立人工盲评仍未完成；现有 49 条分层原文抽查及六条片段判读不能代替盲评。

复现入口：在项目根目录先运行数值脚本，最后运行 `python F题_初步方案/code/problem1/problem1_figures.py` 生成经审计图表。结果说明见 `../结果分析/RESULTS_REPORT_PROBLEM1.md`。`ad_en` 的二分类 logits 定义按 [SlimPajama Meta-rater 官方数据说明](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/blob/main/README.md) 核对：第二类表示无广告，因此代码转为第二类概率后作收益型信号是合理的。
