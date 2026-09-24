# 问题三计算结果：支持域内的条件算力配置

## 1. 模型与证据边界

本问列出问题一的 17 域推荐质心配比，使用问题二经 B6 分组 CV 选中的 **exp_both** 质量响应。优化变量为 $N,D,Q_B$；$Q_A$ 与 $Q_B$ 未标定，也不把半合成 B6 解释为真实训练质量处理实验。主情景 $Q_{B,0}=0.4$ 是 B6 已列出的质量水平，**不是**问题一评分实测值。质量成本是题设参数情景，不是采购报价。**当前 Loss 与成本公式均不含配比 $p$；因此第一问推荐配比是外生政策情景，不影响表中的数值最优解，更不能声称已求出 $N,D,Q_B,p$ 的联合最优。** 推荐配比主要来自第一问较小尺度的配比实验；用于本问较大 $N$ 时也缺少跨尺度联合验证。该不可识别性单列于 `recipe_identifiability_audit.json`。

共同支持域为 $N_B\in[0.070542,11.965825]$、$D_B\in[10,299.893]$、$Q_B\in[0.4,1]$。其中 B1 是真实训练轨迹，B6 质量关系为半合成；连续最优 $Q_B$ 是模型在 B6 离散质量水平之间的插值。目标值包括 B6 来源截距 $\delta_B$；截距不影响决策。固定配比参见 `fixed_recipe.csv`，输入范围和 C7 窗口分别见 `input_support_audit.csv`、`c7_context_support.csv`。

三种成本按题面附录 B：$g_{exp}(Q_B)=10^7e^{6Q_B}$、$g_{pow}(Q_B)=5\times10^9Q_B^4$、$g_{log}(Q_B)=2\times10^9\ln(1+10Q_B)$，单位 FLOPs/Token；总成本为 $6ND+\eta NDL_{ctx}+D[g(Q_B)-g(Q_{B,0})]_+$，$\eta=2\times10^{-4}$。固定的 17 域配比在所有预算情景保持同一份额：

| domain | recommended_share |
| --- | --- |
| arxiv | 0.057683 |
| freelaw | 0.078612 |
| nih_exporter | 0.010769 |
| pubmed_central | 0.10392 |
| wikipedia_en | 0.11145 |
| dm_mathematics | 0.030672 |
| github | 0.044036 |
| philpapers | 0.0057878 |
| stackexchange | 0.08543 |
| enron_emails | 0.0028839 |
| gutenberg_pg_19 | 0.077087 |
| pile_cc | 0.15133 |
| ubuntu_irc | 0.040241 |
| europarl | 0.015094 |
| hackernews | 0.025497 |
| pubmed_abstracts | 0.079619 |
| uspto_backgrounds | 0.079891 |

## 2. 预算最优解

主成本为题设指数型，外生上下文长度 2048；表中 $10^{24}$ 预算若出现未用余额，则是支持域饱和，不代表真实高预算最优策略。

| budget_FLOPs | N_star_B | D_star_B | Q_star_B | predicted_loss | C_unused_fraction | loss_gain_vs_fixed_Q0 | active_set | support_saturated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e+19 | 0.12898 | 10 | 0.55744 | 3.3064 | 0 | 0.039241 | Q:interior|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | False |
| 1e+21 | 2.3231 | 53.152 | 1 | 2.3802 | 0 | 0.12993 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | False |
| 1e+22 | 5.6433 | 249.41 | 1 | 2.1676 | 2.2204e-16 | 0.1032 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | False |
| 1e+24 | 11.966 | 299.89 | 1 | 2.11 | 0.97582 | 0.098794 | Q:upper|Nmin:0|Nmax:1|Dmin:0|Dmax:1 | True |

质量、训练、注意力三种成本均回代为原始 FLOPs；三种质量成本函数互斥。$Q_B$ 固定在基线 0.4 的最优方案作为可行对照；`loss_gain_vs_fixed_Q0` 仅为模型条件改善。

### 三种质量成本敏感性：$C=10^{21}$ FLOPs

| quality_cost | N_star_B | D_star_B | Q_star_B | predicted_loss | quality_fraction_of_used |
| --- | --- | --- | --- | --- | --- |
| exponential | 2.3231 | 53.152 | 1 | 2.3802 | 0.20857 |
| power | 2.4289 | 48.923 | 1 | 2.3857 | 0.23835 |
| logarithmic | 2.0228 | 68.765 | 1 | 2.3646 | 0.10844 |

质量处理成本函数在题目中给出，但未由实验测得，故不同成本式的最优 $Q_B$ 应并列报告，不能挑有利于主结论的一种。

### 质量响应结构敏感性

以下仅比较所选 exp_both 与未入选的 exp_D 候选在同一成本下给出的决策；两者来源截距和质量参数不同，**不直接比较预测 Loss 数值**：

| budget_FLOPs | model_structure | N_star_B | D_star_B | Q_star_B |
| --- | --- | --- | --- | --- |
| 1e+19 | selected_exp_both | 0.12898 | 10 | 0.55744 |
| 1e+19 | candidate_exp_D | 0.13216 | 10 | 0.54501 |
| 1e+21 | selected_exp_both | 2.3231 | 53.152 | 1 |
| 1e+21 | candidate_exp_D | 2.3231 | 53.152 | 1 |
| 1e+22 | selected_exp_both | 5.6433 | 249.41 | 1 |
| 1e+22 | candidate_exp_D | 5.6433 | 249.41 | 1 |

完整数据见 `m1_structure_sensitivity.csv`。低预算两种结构给出的 $Q_B$ 分别约 0.557 与 0.545；中高预算均触及上界。该结论仍受 B6 半合成标尺与题设成本控制。

### 基线质量选择的敏感性：$C=10^{19}$ FLOPs

| Q0_B | N_star_B | D_star_B | Q_star_B |
| --- | --- | --- | --- |
| 0.2 | 0.12037 | 10 | 0.54407 |
| 0.4 | 0.12898 | 10 | 0.55744 |
| 0.6 | 0.15602 | 10 | 0.6 |
| 0.8 | 0.15602 | 10 | 0.8 |

其他预算的 $Q_{B,0}$ 扫描见 `quality_baseline_sensitivity.csv`。低预算下最优质量对基线假设敏感，故不把单一 $Q_{B,0}$ 的选择写成确定政策。

## 3. 上下文与结构性转移

题面 $C_{train}=6ND$、$C_{attn}=\eta NDL_{ctx}$ 给出临界值 $L_{ctx}^{crit}=6/\eta=30,000$。C7 的 `max_position_embeddings` 有 [2048, 4096, 8192, 32768, 131072] 五种**最大位置容量**，其中 32768 略高于临界值，131072 只有一条记录。这些数值用来构造假设的上下文情景，**不是模型实际训练序列长度的观测值**，也不能据此验证注意力成本。注意力与基础训练成本比为 $\eta L_{ctx}/6$，不依赖 $N,D$；在 32768 情景时约为 1.092。固定 $C=10^{21}$、指数质量成本时：

| L_ctx | N_star_B | D_star_B | Q_star_B | predicted_loss |
| --- | --- | --- | --- | --- |
| 2048 | 2.3231 | 53.152 | 1 | 2.3802 |
| 4096 | 2.2442 | 52.008 | 1 | 2.3858 |
| 8192 | 2.1073 | 49.949 | 1 | 2.3963 |
| 32768 | 1.6067 | 41.505 | 1 | 2.4449 |
| 1.3107e+05 | 0.97863 | 28.209 | 1 | 2.5501 |

在部分低预算与长窗口组合下，共同的 $N,D$ 下界已不可负担；这类情景在 `context_sensitivity.csv` 标为 `infeasible_lower_support`，不填造最优解。

结构转移预先定义为 $Q_B$ 的下界/内点/上界或 $N,D$ 支持域下界/上界活跃集合变化。预算路径在 `budget_path.csv`，二分后的转折预算如下；完整左右区间在 `active_set_transitions.csv`：

| quality_cost | budget_estimate_FLOPs | from_active_set | to_active_set |
| --- | --- | --- | --- |
| exponential | 8.93945e+19 | Q:interior|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:interior|Nmin:0|Nmax:0|Dmin:0|Dmax:0 |
| exponential | 1.54483e+20 | Q:interior|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 |
| exponential | 1.33956e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 |
| exponential | 2.41753e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 | Q:upper|Nmin:0|Nmax:1|Dmin:0|Dmax:1 |
| logarithmic | 1.71848e+19 | Q:lower|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:interior|Nmin:0|Nmax:0|Dmin:1|Dmax:0 |
| logarithmic | 2.44555e+19 | Q:interior|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:1|Dmax:0 |
| logarithmic | 6.12137e+19 | Q:upper|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 |
| logarithmic | 1.10736e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 |
| logarithmic | 2.34711e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 | Q:upper|Nmin:0|Nmax:1|Dmin:0|Dmax:1 |
| power | 1.11932e+20 | Q:interior|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:1|Dmax:0 |
| power | 1.27127e+20 | Q:upper|Nmin:0|Nmax:0|Dmin:1|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 |
| power | 1.42708e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:0 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 |
| power | 2.44593e+22 | Q:upper|Nmin:0|Nmax:0|Dmin:0|Dmax:1 | Q:upper|Nmin:0|Nmax:1|Dmin:0|Dmax:1 |

这些只是**模型与支持域约束**的转折；连续成本份额变化不称质变。高预算进入 $N,D,Q_B$ 全部上界且预算有余的状态属于数据支持域饱和。

## 4. 结果核验与不确定性

共复核 207 个主解、预算路径和上下文情景；最大预算超限比例 2.220e-16，所有 $N,D,Q_B$ 均在指定范围，逐解记录见 `constraint_checks.csv`。独立 180×90 网格核查的最大“优化 Loss－网格最小 Loss”为 -9.970e-09；另一种随机种子固定的差分进化全局搜索，对 12 个主解的最大“现有解 Loss－独立搜索 Loss”为 -3.550e-12。两者均未找到明显更优解，逐解见 `independent_grid_checks.csv` 和 `global_search_checks.csv`。内点变量的 Loss 边际降低量/FLOP 应相等；可比较的主解最大相对差为 7.224e-08，见 `marginal_kkt_checks.csv`。边界变量不要求满足内点相等式。

来自问题二同编号 B1 轨迹/B6 单元重抽样的 80 个条件参数样本，其配置分位数如下（详见 `conditional_bootstrap_allocations.csv`、`conditional_bootstrap_quantiles.csv`）：

| budget_FLOPs | quantile | N_star_B | D_star_B | Q_star_B | predicted_loss |
| --- | --- | --- | --- | --- | --- |
| 1e+19 | 0.025 | 0.12754 | 10 | 0.55089 | 3.2951 |
| 1e+19 | 0.5 | 0.12879 | 10 | 0.55815 | 3.3075 |
| 1e+19 | 0.975 | 0.13069 | 10 | 0.56281 | 3.3175 |
| 1e+21 | 0.025 | 2.3223 | 53.143 | 1 | 2.3722 |
| 1e+21 | 0.5 | 2.323 | 53.152 | 1 | 2.3797 |
| 1e+21 | 0.975 | 2.3236 | 53.166 | 1 | 2.3891 |
| 1e+22 | 0.025 | 5.64 | 249.33 | 1 | 2.1597 |
| 1e+22 | 0.5 | 5.6433 | 249.41 | 1 | 2.1672 |
| 1e+22 | 0.975 | 5.6453 | 249.54 | 1 | 2.1766 |

区间只覆盖当前模型与数据重抽样，不包含质量成本参数不确定性、A/B 标尺不一致、真实质量处理价格或更大模型尺度外推。B1 近乎公式化，不能把狭窄区间称作现实训练项目的完整认知不确定性。

## 5. 图表与复现

| figure | data_source | interpretation |
| --- | --- | --- |
| budget_path.pdf | budget_path.csv | 支持域内最优配置；高预算平台为支持域饱和 |
| compute_allocation.pdf | optimal_allocations.csv | 三部分成本占已使用预算比例；未用预算另见表 |
| context_sensitivity.pdf | context_sensitivity.csv | 以 C7 最大位置容量构造的假设情景；不是实测训练长度 |

从仓库根目录运行：

```powershell
python F题_初步方案/code/problem3/problem3.py
python F题_初步方案/code/problem3/verify_problem3.py
```
