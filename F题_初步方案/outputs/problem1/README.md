# 问题一输出

| 路径 | 内容 |
| --- | --- |
| `quality/cache/` | 质量数据读取审计及可再生成的特征缓存 |
| `quality/results/` | 样本评分、领域汇总、指标冲突、成因画像及质量敏感性结果 |
| `quality/figures/` | 质量评价图表 |
| `mixture/results/` | 配比模型、A16/Q 消融、多尺度检验、预测和稳健推荐配比 |
| `mixture/figures/` | 配比模型图表 |
| `enhancement/figures/` | 稳健性与估计尺度诊断图表 |
| `figure_manifest.csv` | 保留图的数据来源、用途及解释边界 |
| `run_summary.json` | 问题一整体运行摘要 |

完整解释和局限见 `../../reports/问题一/结果分析/RESULTS_REPORT_PROBLEM1.md`。先运行数值脚本更新结果，最后运行 `../../code/problem1/problem1_figures.py` 生成论文图并清理被替代的旧图。
