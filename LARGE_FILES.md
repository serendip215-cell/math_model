# 克隆后恢复大文件

仓库中的 5 个数据文件因超过 Gitee 的单文件大小限制，被拆成 `.part000`、`.part001` 等分片提交。克隆后需要运行一次恢复脚本，才能得到原始文件。分片顺序、原文件大小和 SHA-256 校验值记录在 `large_files_manifest.json` 中；无需安装 Git LFS。

## 操作步骤

1. 克隆仓库并进入仓库根目录（即包含 `restore_large_files.ps1` 的目录）：

   ```powershell
   git clone https://gitee.com/dadi-da/match_model.git
   cd match_model
   ```

   如果已经克隆，只需在仓库根目录运行 `git pull`，确保分片和清单已下载。

2. 在 PowerShell 中运行：

   ```powershell
   .\restore_large_files.ps1
   ```

   如果 Windows 提示禁止运行脚本，可以仅对这一次调用放宽执行策略，不修改系统的永久设置：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\restore_large_files.ps1
   ```

3. 等待脚本逐项输出 `Restored: ...`。脚本会按清单顺序拼接分片，核对文件长度和 SHA-256，然后将文件写到下面的原路径。再次运行时，已有文件校验通过则输出 `Already valid: ...`。完整执行且没有报错，即表示清单中的 5 个文件均已校验。

## 恢复的文件

| 原路径（相对于仓库根目录） | 用途 |
| --- | --- |
| `F题_清洗后/A_data_value/regmix_domain_sample.jsonl` | RegMix 各领域原始文本样本 |
| `F题_清洗后/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl` | GitHub 领域扩展质量信号 |
| `F题_清洗后/A_data_value/slimpajama_quality_signal_sample.jsonl` | SlimPajama 多领域质量信号抽样数据 |
| `real_attachments/A_data_value/regmix_domain_sample.jsonl.xz` | RegMix 样本压缩文件 |
| `real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz` | SlimPajama 质量信号样本压缩文件 |

恢复过程需要约 1.5 GiB 的额外可用磁盘空间。`.jsonl` 是每行一个 JSON 对象的文本文件；`.xz` 是压缩文件。恢复脚本只负责合并分片，不会解压 `.xz`。

## 遇到错误时

- `Existing file has wrong SHA256`：目标文件已存在，但与清单不符。先检查是否有自己修改的数据；如需重新恢复，备份并移走该文件，再运行脚本。
- `Temporary file exists`：上次恢复中断，留下了同路径的 `.reassembling` 文件。确认没有其他恢复任务在运行后，删除该临时文件并重试。
- `Reassembled file failed verification` 或找不到 `.partNNN`：分片可能未拉取完整或已损坏。运行 `git pull` 并检查对应分片，再重试；若校验失败留下 `.reassembling` 文件，先按上一条处理。

恢复出的 5 个原文件已列入 `.gitignore`，因此 `git status` 不会显示它们。仓库跟踪的是分片、恢复脚本和清单；不要直接提交合并后的大文件。需要更新大文件时，应重新生成对应分片和清单，并提交它们。
