# F 题项目目录

本仓库保存赛题原始附件、清洗后建模数据和初步方案。建议从以下入口阅读：

| 路径 | 用途 |
| --- | --- |
| [`LARGE_FILES.md`](LARGE_FILES.md) | 克隆后合并大文件分片 |
| [`real_attachments/`](real_attachments/) | 原始附件，保留来源与复核依据 |
| [`F题_清洗后/`](F题_清洗后/) | 建模代码使用的数据及可见层题面说明 |
| [`F题_初步方案/README.md`](F题_初步方案/README.md) | 环境准备、问题一运行和输出位置 |
| [`F题_初步方案/reports/README.md`](F题_初步方案/reports/README.md) | 按数据处理、方案、结果、验收查找报告 |

根目录的 `数据说明.pdf` 和赛题 `.docx` 是原始题面。`F题_清洗后` 与 `real_attachments` 分别用于建模和保留来源，不能因为文件相似而互相替代。`.partNNN` 是还原 5 个大文件所需的仓库分片，完整文件在本地由 `restore_large_files.ps1` 生成；两者同时出现属于正常情况。

本地的 `.idea/`、`.venv/`、Python `__pycache__/` 和 `F题_初步方案/outputs/problem1/quality/cache/quality_features.csv` 都是可重新生成的环境或缓存文件，已被 Git 忽略，可以在不用时清理。其他报告、图表、审计记录和原始/清洗数据保留用于复现。
