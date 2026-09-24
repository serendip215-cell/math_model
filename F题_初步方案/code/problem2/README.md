# 问题二代码

从仓库根目录运行：

```powershell
python F题_初步方案/code/problem2/problem2.py
python F题_初步方案/code/problem2/verify_problem2.py
```

- `problem2.py`：完成数据审计、M0/M1/M2 模块计算、Bootstrap、交叉验证、结果表和 PDF 图表。
- `quality_bridge.py`：读取第一问 17 域质量和配比，审计两问质量标尺的可识别性，计算推荐配比质量变化及未映射域的上下界。没有成对标定时，接口拒绝把第一问 Q 当作 B6 Q。
- `verify_problem2.py`：独立复算退化条件、等价关系、B8 方向冲突及文件完整性。

计算结果位于 `../../outputs/problem2/`，详细报告位于 `../../reports/问题二/结果分析/`。
