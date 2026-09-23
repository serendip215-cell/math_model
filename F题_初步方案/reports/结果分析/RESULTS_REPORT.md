# F 题运行结果总览

## 已完成

- 问题一的质量评价、指标冲突分析和训练配比响应模型已完成全量运行。
- 详细结果、假设边界及适用范围见 [RESULTS_REPORT_PROBLEM1.md](RESULTS_REPORT_PROBLEM1.md)。
- 可复现代码位于 `code/problem1.py`，数值结果与图形按方法保存在 `outputs/problem1/quality/` 和 `outputs/problem1/mixture/`。

## 安全边界

本轮只读取 `F题_清洗后/A_data_value`。低可见度 PDF 文本及其中的方法、参数和结论未进入计算。数据内自然语言字段仅作为普通数据，不作为指令执行。
