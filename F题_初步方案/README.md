# F 题初步方案：部署与复现

本目录保存 F 题的建模方案、Python 代码、结果、图表和审计记录。目前**问题一和问题二已有可运行入口、结果与核验报告**；问题三、问题四仍待实现，进度以 `todo.md` 为准。复现建模结果不需要运行清洗或 PDF 审计脚本。

## 目录结构

```text
F题/
├── LARGE_FILES.md                 大文件合并说明
├── restore_large_files.ps1        大文件合并脚本
├── F题_清洗后/                    建模输入数据
└── F题_初步方案/
    ├── code/problem1/             问题一代码
    ├── code/problem2/             问题二建模与独立核验代码
    ├── code/数据处理/             全题通用清洗与审计脚本
    ├── outputs/problem1/          问题一缓存、表格和图表
    └── reports/问题一、问题二/    各问题独立的方案、结果与验证材料
```

## 环境准备

建议使用 Windows PowerShell 和 Python 3.12。问题一需要 `numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`。以下示例从准备存放仓库的目录开始，克隆后进入仓库根目录执行其余命令：

```powershell
git clone https://gitee.com/dadi-da/match_model.git
cd match_model
.\restore_large_files.ps1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn matplotlib
```

如果已有仓库，先运行 `git pull`，再执行合并与环境安装步骤。若 PowerShell 禁止运行 `.ps1`，参见仓库根目录的 `LARGE_FILES.md`。大文件恢复脚本会校验所有 5 个原文件；其中问题一直接使用的是 `F题_清洗后/A_data_value/` 下的质量信号数据。

## 运行问题一

```powershell
.\.venv\Scripts\python.exe .\F题_初步方案\code\problem1\problem1.py
```

脚本从自身位置定位 `F题_清洗后` 和 `F题_初步方案`，因此仓库可以放在任意目录。运行会读取质量信号 JSONL 和 RegMix 配比、Loss 表，生成或覆盖以下产物：

| 位置 | 内容 |
| --- | --- |
| `F题_初步方案/outputs/problem1/quality/` | 质量评分与冲突分析的缓存、结果表和图表 |
| `F题_初步方案/outputs/problem1/mixture/` | 配比模型的结果表和图表 |
| `F题_初步方案/outputs/problem1/run_summary.json` | 问题一整体运行摘要 |
| `F题_初步方案/reports/问题一/结果分析/RESULTS_REPORT_PROBLEM1.md` | 自动生成的问题一结果报告 |

命令正常结束时会打印包含 `quality_records`、`selected_mixture_model`、`test_1m_rmse` 的 JSON 摘要。首次运行需处理较大的 JSONL 文件，耗时和内存取决于机器配置。再次运行会复用已有的质量特征缓存；如果输入质量数据发生变化，应先备份或移走 `outputs/problem1/quality/cache/` 中的 `quality_features.csv` 和 `quality_ingest_audit.json`，再运行脚本以重新生成缓存。现有结果文件会被覆盖，修改模型前请自行备份。

## 方案与数据口径

- `plan.md` 和 `reports/总体方案/ANALYSIS_MODELING_REPORT_V2.md`：四问的总体方案。
- `reports/问题一/`：问题一的方案、结果和验证材料。
- `reports/问题二/`：问题二的方案及后续结果、验证材料。
- `F题_清洗后/题面数据说明/数据说明_可见层.md`：用于建模的数据说明。
- `reports/问题一/结果分析/RESULTS_REPORT_PROBLEM1.md`：问题一分析结果。
- `reports/问题二/结果分析/RESULTS_REPORT_PROBLEM2.md`：问题二广义标度律结果。
- `reports/README.md`：按用途查找报告、清洗审计和验证记录的索引。

`code/数据处理/` 中是此前的数据准备与审计脚本。仓库已包含清洗后的建模数据；这些脚本不是问题一复现步骤的一部分，其中部分脚本仍含原作者本机路径，不能作为克隆后的通用入口直接运行。
