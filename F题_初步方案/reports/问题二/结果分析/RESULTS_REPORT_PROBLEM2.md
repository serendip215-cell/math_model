# 问题二计算结果：广义标度律

## 1. 证据和数据口径

本报告由 `code/problem2/problem2.py` 从清洗后数据直接生成。B1 是经典标度律主拟合数据；B6 是质量项主拟合数据；B2、B4、B5、B7 只作验证；B8、B10 只作风险检查。不同来源未被拼成一个训练集，也没有报告跨 A/B 数据的虚假总体 RMSE。

| dataset | rows | evidence_type | role |
| --- | --- | --- | --- |
| B1 | 1176 | 附件训练日志 | M0 主拟合；近乎确定性幂律，需警惕构造性 |
| B2 | 1029 | 半合成 | 模型族外压力测试 |
| B3 | 4000 | 插值 | 连续性检查，不作独立验证 |
| B4 | 57 | 真实整理 | 跨族外部验证 |
| B5 | 44 | 真实整理 | 文献外部验证 |
| B6 | 360 | 半合成 | M1 主拟合 |
| B7 | 450 | 半合成 | 新增 Q 水平敏感性 |
| B8 | 1704 | 半合成/外推 | 方向冲突与远域压力测试 |
| B9 | 132 | 真实元数据 | 外推范围 |
| B10 | 128 | 估算 | 大模型风险边界 |
| B11 | 18 | 辅助元数据 | 模型族索引 |
| B12 | 1386 | 辅助索引 | 检查点索引 |

B1 的计算量字段满足 $C\approx6ND$：换算到附件单位后为 $C_{10^{21}}=0.006N_BD_B$，绝对相对误差中位数为 $2.62\times10^{-5}$，95% 分位数为 0.00141，99.32% 的记录误差不超过 5%。少数最早期检查点因 C 只保留四位小数而产生较大相对误差。

## 2. M0：经典 N-D 标度律

拟合形式为 $L_0=E+A N^{-\alpha}+B D^{-\beta}$，N 和 D 均采用附件中的十亿单位。参数按 B1 的完整 N 轨迹进行 Bootstrap：

| parameter | estimate | bootstrap_low | bootstrap_high | huber_sensitivity |
| --- | --- | --- | --- | --- |
| E | 1.6898 | 1.6897 | 1.69 | 1.6898 |
| A | 0.35398 | 0.35376 | 0.3541 | 0.35398 |
| B | 1.2403 | 1.2402 | 1.2404 | 1.2403 |
| alpha | 0.33998 | 0.33987 | 0.34014 | 0.33998 |
| beta | 0.27988 | 0.27983 | 0.27994 | 0.27988 |

按完整参数规模轨迹留出的验证结果为 RMSE=0.000151、MAE=0.000108、Spearman=1.000000、偏差=0.000007。这比随机拆分更严格，因为同一训练轨迹不会同时进入训练和验证。误差只有 $10^{-4}$ 量级，且参数 Bootstrap 区间异常窄，表明 B1 与该幂律几乎是确定性关系。它更接近公式化或强规则化数据；这里的区间主要反映轨迹重抽样与数值舍入，不能当作真实预训练过程的完整认知不确定性。

### 外部数据压力测试

| dataset | n | rmse | centered_rmse | spearman | inside_B1_rectangle_fraction |
| --- | --- | --- | --- | --- | --- |
| B2 | 1029 | 1.2431 | 0.39564 | 0.7881 | 0.11662 |
| B4 | 57 | 0.29267 | 0.19186 | 0.98298 | 0.14035 |
| B5 | 44 | 0.1976 | 0.19652 | 0.95881 | 0.18182 |
| B10 | 128 | 0.0010938 | 0.00090521 | 0.99999 | 0 |

B4/B5 的 Loss 口径存在模型族、分词器和验证集差异，因此原始 RMSE 不能单独解释为结构失效；表中同时给出去均值 RMSE 与排序相关。B10 是估算值，只描述远域偏离，不作为真实精度证据。

## 3. M1：质量修正

主候选为 $L_1=\delta_B+E+A N^{-\alpha}+B D^{-\beta}\exp[\gamma_Q(1-Q)]$。$\delta_B$ 只校准 B6 与 B1 的来源基线，不改变质量方向。

| model | rmse_mean | rmse_sd | spearman_mean | gamma_mean |
| --- | --- | --- | --- | --- |
| baseline | 0.14005 | 0.0050075 | 0.88935 | 0 |
| exp_D | 0.093135 | 0.013373 | 0.96 | 0.53696 |
| exp_N | 0.07018 | 0.0070573 | 0.97608 | 0.54251 |
| exp_both | 0.059506 | 0.0062501 | 0.97968 | 0.36501 |
| linear_D | 0.094102 | 0.013438 | 0.95955 | 0.67188 |
| power_D | 0.096264 | 0.011528 | 0.96043 | 0.22144 |

竞争模型包括 Q 作用于 D 项（exp_D）、N 项（exp_N）和 N、D 两项（exp_both），并在 D≤300B 的完整单元分组交叉验证中选择 **exp_both**。D=600B 单独作为外推审计，不并入插值 CV。指数 D-only 的质量参数为 $\gamma_Q=0.5369$，同一响应的轨迹单元 Bootstrap 95% 区间为 [0.4749, 0.6309]；相对无质量项的 RMSE 改善为 33.50%。B7 新增 Q 水平的外推 RMSE 为 0.0676。这组结果不支持事先把质量作用位置固定为 D 项。

B6、B7、B8 的相邻 Q 方向审计如下：

| dataset | comparisons | nonincrease_fraction | median_delta_loss |
| --- | --- | --- | --- |
| B6 | 315 | 0.74603 | -0.0426 |
| B7 | 405 | 0.74815 | -0.0371 |
| B8 | 1554 | 0.15637 | 0.1937 |

B8 只有 15.64% 的相邻变化符合“Q 提高时 Loss 不升”，与题意和 B6/B7 相反，故未进入拟合，也未擅自反转 Q。

## 4. 边际效用和质量—参数等价量

在参考配比处，解析边际量为：

$$
\frac{\partial L}{\partial N}=-\alpha A N^{-\alpha-1},\quad
\frac{\partial L}{\partial D}=-\beta B D^{-\beta-1}e^{\gamma_Q(1-Q)},\quad
\frac{\partial L}{\partial Q}=-\gamma_Q B D^{-\beta}e^{\gamma_Q(1-Q)}.
$$

参数、数据弹性和质量半弹性已保存于 `marginal_elasticities.csv`。在 B6 支持网格内，Q 提升 0.1 的参数等价倍数中位数为 1.219；每个网格点均给出轨迹/单元 Bootstrap 区间和有效抽样比例。该数值是固定 D、固定配比、采用 B6 原生 Q 标尺时的条件等价量，不能脱离基准 N、D、Q 使用。问题一 Q 与 B6 Q 无成对标定，因此不把该倍数用于第一问推荐配比。

## 5. M2 配比接口与领域替换

问题一现有结果保存了 17 域在训练均值处的单域 +1 个百分点扰动，但没有保存二阶响应面的 Hessian 或 Bootstrap 系数抽样。因此本问从这些真实输出恢复一阶中心化梯度，形成可行的成对局部替换矩阵：

$$
S_{i\leftarrow j}(0.01)\approx0.01(\tilde g_i-\tilde g_j).
$$

局部预测 Loss 降幅最大的五个可行替换方向为：

| increase_domain | decrease_domain | shift_share | delta_macro_loss_linearized |
| --- | --- | --- | --- |
| ubuntu_irc | europarl | 0.01 | -0.0077055 |
| dm_mathematics | europarl | 0.01 | -0.0074047 |
| ubuntu_irc | hackernews | 0.01 | -0.0071676 |
| dm_mathematics | hackernews | 0.01 | -0.0068668 |
| ubuntu_irc | freelaw | 0.01 | -0.0066522 |

这些是问题一训练均值附近的一阶模型预测，不是因果效应。当前证据不能估计曲率置信区间，因此不输出“显著互补”标签。配比项在 1M、60M、1B 的问题一结果中采用经验权重，超过 1B 时只能按中位衰减指数 0.1503 做敏感性外推。

以下是 **D-only 候选模型** 的条件计算式，用于质量—参数等价量和 IsoFLOP 敏感性分析；B6 插值 CV 选出的结构为 **exp_both**，故不能把该式表述为已验证的唯一完整模型：

$$
L(N,D,Q,p)=E+A N^{-\alpha}+B D^{-\beta}e^{\gamma_Q(1-Q)}+w_p(N)\Delta_p(p).
$$

当 Q=1 且 $p=p^{ref}$ 时，质量修正因子为 1、配比差为 0，模型严格退化到 M0。规模、质量和配比模块分别验证，未构造跨来源总体拟合指标。领域替换矩阵仅是一阶近似；现有问题一输出缺少边级标准误，272 条有向边无法做 BH-FDR 检验，不能标注显著互补或替代。逐域 $\kappa_i$ 保存在 `domain_decay_weights.csv`，负值原样保留，并只在观测尺度内解释。

## 6. 第一问质量输出到第二问的可识别接口

读取第一问的 17 域质量代理值和训练均值、推荐质心两组配比，按 $Q_A(p)=\sum_i p_i q_i$ 计算质量。17 域代理值中有 6 域由 A16 映射到已有质量域，其余 11 域是由第一问训练 Loss 等信息推得的代理值，不是独立测量。训练均值的代理质量为 0.631999，推荐质心为 0.628948，差值为 -0.003052。

若仅接受 A16 已映射域、并允许未映射域的真实质量各自在 $[0,1]$，推荐质心相对训练均值的质量变化落在 [-0.057492, +0.036224]。区间跨过零，因此现有资料连变化方向也不能无条件识别；该区间还依赖 A16 映射的语义可比性，不能当作统计置信区间。

B6 没有 17 域配比，第一问配比试验没有 B6 的 Q 标尺与共同 Loss 协议，也没有成对实验 ID。故 $Q_A\to Q_B$ 的校准函数和 $L(N,D,Q,p)$ 的联合误差均**不可识别**。代码接口在缺少经实测标定的映射时会拒绝把 $Q_A$ 代入 B6 模型。领域级审计、配比质量与界、联合数据合同分别见 `quality_bridge_domain_audit.csv`、`quality_bridge_recipe_anchors.csv`、`quality_bridge_delta_bounds.csv` 和 `quality_bridge_joint_data_audit.csv`。

## 7. 图表索引

| figure | data_source | interpretation |
| --- | --- | --- |
| m0_scaling_fit.pdf | B1 附件训练日志 | 训练轨迹与 M0 拟合 |
| m0_loo_diagnostics.pdf | B1 按 N 留轨迹 | M0 防泄漏验证 |
| m0_profile_likelihood.pdf | B1 附件训练日志 | M0 参数可识别性 |
| m1_quality_curves.pdf | B6 半合成 | 质量曲线与 M1 拟合 |
| m1_cv_comparison.pdf | B6 D≤300B 完整 (N,D) 单元分组 | 质量位置与函数竞争模型 |
| marginal_elasticities.pdf | M0+M1 解析计算 | 参数、数据和质量边际效用 |
| quality_parameter_equivalence.pdf | M0/M1 Bootstrap | 质量提升与参数增加的条件等价量 |
| domain_substitution_matrix.pdf | 问题一已保存的局部扰动 | 17 域一阶替换矩阵；不识别互补性 |
| external_and_quality_risk.pdf | B2/B4/B5/B10 与 B6/B7/B8 | 外部误差与质量方向风险 |
| quality_bridge_identification.pdf | 问题一 17 域质量、A16 映射与两组配比 | 质量变化的代理值与部分识别界；不是 B6 标尺校准 |

所有图均由本次结果表直接生成，PDF 中不写论文式大标题。真实观测、半合成、插值和估算数据的解释边界见图表清单与数据审计表。

## 8. 可复现运行

```powershell
python F题_初步方案/code/problem2/problem2.py
python F题_初步方案/code/problem2/verify_problem2.py
```

随机种子为 20260924。主要数值汇总位于 `outputs/problem2/results/problem2_summary.json`，独立验收结果位于 `reports/问题二/验证验收/VERIFY_REPORT_PROBLEM2.md`。
