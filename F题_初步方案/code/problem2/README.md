# 问题二代码

从仓库根目录运行：

```powershell
python F题_初步方案/code/problem1/export_mixture_curvature.py
python F题_初步方案/code/problem2/problem2.py
python F题_初步方案/code/problem2/verify_problem2.py
```

- `export_mixture_curvature.py`：从第一问已选中的二次 ILR 模型导出系数、ILR Hessian 和按配方行重抽样的系数。会回代第一问原预测验证一致性。
- `problem2.py`：完成数据审计、M0/M1/M2 模块计算、Bootstrap、交叉验证、结果表和 PDF 图表。
- `isoflop_frontier.csv` 和同名 PDF：采用 B6 分组 CV 选中的 exp_both 模型；B1 实测算力分桶与模型条件前沿分开标识。
- `local_pair_nonadditivity.csv`：两个参考配比下 136 个无序域对的局部模型非加性，含近似 Wald/BH 结果；不表示因果互补。
- `quality_bridge.py`：读取第一问 17 域质量和配比，审计两问质量标尺的可识别性，计算推荐配比质量变化及未映射域的上下界。没有成对标定时，接口拒绝把第一问 Q 当作 B6 Q。
- `verify_problem2.py`：独立复算退化条件、等价关系、B8 方向冲突及文件完整性。

计算结果位于 `../../outputs/problem2/`，详细报告位于 `../../reports/问题二/结果分析/`。
