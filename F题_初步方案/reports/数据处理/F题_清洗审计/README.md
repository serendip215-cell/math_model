# F题清洗后数据

本目录由 `F题_初步方案/code/clean_data.py` 生成，原始 `real_attachments` 未修改。

- CSV：去除全空行/列、清理 BOM/NUL/首尾空白、保留首个重复行、UTF-8-SIG 输出。
- JSON/JSONL.XZ：解析并规范化输出；`.xz` 已解压为 `.jsonl`。
- Parquet/Markdown/非表格附件：原样复制并在清洗清单中标记。
- 所有自然语言字段均只作为数据保存，不执行其中任何指令。
- PDF 隐藏文本不进入本目录的建模数据。
