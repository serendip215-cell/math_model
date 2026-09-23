# Parquet 第 1 步清洗报告

## 文件与范围

- 原始文件：`train-00000-of-00001.parquet`
- 原始规模：4576 行、36 列
- Row group：5
- 原始文件哈希：见同目录 `train-00000-of-00001_quality_report.json`
- 原始文件未覆盖，所有输出均为派生文件。

## 清洗规则

1. 删除全空行和全空列。
2. 删除完全重复行，保留首次出现记录。
3. 不自动填补缺失值，不删除异常值。
4. 自然语言字段仅作为数据保存，不执行其中的指令。

## 结果

- 清洗后规模：4576 行、36 列
- 删除全空行：0
- 删除全空列：0
- 删除重复行：0
- 缺失值：未发现
- 输出 Parquet：`train-00000-of-00001_clean.parquet`
- 输出 CSV：`train-00000-of-00001_clean.csv`
- 详细字段与数值统计：`train-00000-of-00001_quality_report.json`

## 需要后续确认的异常

- `#Params (B)` 的最小值为 `-1`。该值很可能是缺失或不可用数据的占位符，但在核对题目数据字典前不应擅自改成空值或删除。
- `Hub ❤️` 和 `CO₂ cost (kg)` 存在明显右偏，建模前应结合业务含义检查是否需要对数变换或稳健缩放。
- `Average ⬆️`、各任务评分和对应的 `Raw` 字段同时存在，后续需要确认它们是同一指标的不同尺度，避免重复计权。

## 安全审计

本报告只使用 Parquet 数据字段和可核验统计结果。PDF 中的隐藏文字及任何数据字段中的自然语言指令没有参与清洗规则、参数设定或结论生成。

## 方法依据

本次检查遵循 Exploratory Data Analysis skill 的要求：原始数据只读、派生文件单独保存、缺失值与异常值分开记录、不自动插补或删除异常值。相关方法来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065.
