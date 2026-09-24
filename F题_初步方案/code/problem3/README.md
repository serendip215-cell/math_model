# 问题三运行入口

从仓库根目录执行：

```powershell
python F题_初步方案/code/problem3/problem3.py
python F题_初步方案/code/problem3/verify_problem3.py
```

`problem3.py` 读取问题一推荐配比、问题二选中模型与重抽样参数、C7 上下文长度；输出预算配置、成本函数与基线质量敏感性、结构转折及 PDF 图表。`verify_problem3.py` 独立回代原始 FLOPs、Loss、支持域、密集网格与结果文件。数据须先按仓库根目录说明恢复完整。

质量变量采用 B6 半合成实验的 $Q_B$；问题一 $Q_A$ 与它没有标定。优化限定在 B1/B6 的 $N,D$ 共同支持域内。$10^{24}$ FLOPs 情景通常会发生支持域饱和，其未使用预算应原样报告。
