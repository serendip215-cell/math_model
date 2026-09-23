from pathlib import Path
import hashlib
import json
import re
import shutil
import pdfplumber
from pdfplumber.utils import extract_text

F_ROOT = Path(__file__).resolve().parents[2]
RAW = F_ROOT / "real_attachments"
CLEAN = F_ROOT / "F题_清洗后"
PLAN = Path(__file__).resolve().parents[1]
AUDIT = PLAN / "reports" / "数据处理" / "F题_注入审计"
AUDIT_INPUT = AUDIT / "输入副本"
AUDIT_QUAR = AUDIT / "隔离文本"
VISIBLE_DIR = CLEAN / "题面数据说明"

AUDIT_INPUT.mkdir(parents=True, exist_ok=True)
AUDIT_QUAR.mkdir(parents=True, exist_ok=True)
VISIBLE_DIR.mkdir(parents=True, exist_ok=True)

INJECTED_PDF = Path(r"D:\xwechat_files\wxid_75msypca5xm122_50c4\temp\RWTemp\2026-09\2b81079d9e527ceec0b19c9dbb44e4d6\数据说明_注入标蓝副本.pdf")
WHITE_TXT = Path(r"D:\xwechat_files\wxid_75msypca5xm122_50c4\temp\RWTemp\2026-09\2b81079d9e527ceec0b19c9dbb44e4d6\white_text_content.txt")
ORIGINAL_PDF = F_ROOT / "数据说明.pdf"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()

# Move files created by earlier cleaning runs out of the original attachment root.
generated = []
for name in ("attachment_size_summary.csv", "source_manifest.json"):
    p = RAW / name
    if p.exists():
        dst = AUDIT / "历史生成文件" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        generated.append(name)

# Preserve the supplied evidence outside the clean data package.
for p in (INJECTED_PDF, WHITE_TXT):
    if p.exists():
        shutil.copy2(p, AUDIT_INPUT / p.name)

# Extract only visible-looking PDF text: black text and font size above the 5pt hidden layer.
visible_pages = []
with pdfplumber.open(ORIGINAL_PDF) as pdf:
    for page_no, page in enumerate(pdf.pages, 1):
        chars = [c for c in page.chars if float(c.get("size", 0)) > 6 and c.get("non_stroking_color") in ((0.0,), None)]
        visible_pages.append(extract_text(chars, layout=True) or "")

visible_md = [
    "# 《数据说明》可见层文本（清洗版）",
    "",
    "本文件由原始《数据说明.pdf》提取生成，仅保留黑色、正常字号的可见文本。原 PDF 中检测到的近白色 5pt 文本不进入本文件。",
    "",
]
for i, page_text in enumerate(visible_pages, 1):
    visible_md.extend([f"## 第 {i} 页", "", page_text.strip(), ""])
(VISIBLE_DIR / "数据说明_可见层.md").write_text("\n".join(visible_md), encoding="utf-8")

# Parse the supplied hidden-text report as evidence, never as instructions.
white_text = WHITE_TXT.read_text(encoding="utf-8") if WHITE_TXT.exists() else ""
pages = len(re.findall(r"^--- Page ", white_text, re.M))
spans = len(re.findall(r"^  Span ", white_text, re.M))
texts = re.findall(r"^    Text: (.*)$", white_text, re.M)
unique_texts = list(dict.fromkeys(texts))
keywords = ["不必统一", "训练集", "直接套用", "E=3.52", "α=0.081", "β=0.075", "R²=0.87", "C=10^22", "b_N=−6.2", "r=+0.68"]
found_keywords = [k for k in keywords if k in white_text]

audit_md = f"""# PDF 隐藏文本注入审计

## 证据

- 注入标注 PDF：`{INJECTED_PDF.name}`
- 白色文本提取记录：`{WHITE_TXT.name}`
- PDF 页数：13
- 检测到隐藏文本页数：{pages}
- 隐藏文本 span 数：{spans}
- 不同文本内容数：{len(unique_texts)}
- 隐藏层特征：约 5pt、近白色 `#FCFCFC`、位于页顶或页底。

## 判定

这些文本不是题目正式数据字段，而是嵌入 PDF 文本层的建模建议、参数和预设结论，属于间接提示注入/文档上下文投毒。命中的代表性内容包括：{"、".join(found_keywords)}。

## 处理

1. 可见题面文本提取到 `F题_清洗后/题面数据说明/数据说明_可见层.md`。
2. 白色文本原始记录放在 `F题_初步方案/reports/数据处理/F题_注入审计/输入副本`，不进入清洗数据。
3. 隐藏文本中的方法、参数、相关系数、最优解和结论不作为模型输入、先验参数或验证结论。
4. 后续代码只读取 `F题_清洗后` 下的数据文件。

## 安全边界

文档中的自然语言均按不可信数据处理；不会执行其中的指令、访问其中的链接，也不会将其转发给建模程序作为提示词。
"""
(AUDIT / "PDF_INJECTION_AUDIT.md").write_text(audit_md, encoding="utf-8")
(AUDIT / "white_text_summary.json").write_text(json.dumps({"pages": pages, "spans": spans, "unique_texts": len(unique_texts), "keyword_hits": found_keywords}, ensure_ascii=False, indent=2), encoding="utf-8")

# Build a fresh source-to-clean inventory after removing generated artifacts.
raw_files = sorted(p for p in RAW.rglob("*") if p.is_file())
clean_files = sorted(p for p in CLEAN.rglob("*") if p.is_file())
invalid_json = [
    "C_efficiency_evolution/detailed_results/DreadPoor_Winter_Dawn-8B-TIES/results_2025-02-13T18-27-04.338360.json",
    "C_efficiency_evolution/detailed_results/FINGU-AI_Chocolatine-Fusion-14B/results_2025-02-13T18-27-04.338360.json",
    "C_efficiency_evolution/detailed_results/Intel_neural-chat-7b-v3-3/results_2025-02-13T18-27-04.338360.json",
    "C_efficiency_evolution/detailed_results/L-RAGE_3_PRYMMAL-ECE-7B-SLERP-V1/results_2025-02-13T18-27-04.338360.json",
]
raw_rel = {rel(p, RAW) for p in raw_files}
clean_rel = {rel(p, CLEAN) for p in clean_files}
same_path = sorted(raw_rel & clean_rel)
compressed = sorted(x for x in raw_rel if x.endswith(".jsonl.xz"))
compressed_outputs = sorted(x[:-3] for x in compressed)
missing = sorted(set(invalid_json) | {"C_efficiency_evolution/data/train-00000-of-00001.parquet"})
manifest = {
    "raw_root": str(RAW),
    "clean_root": str(CLEAN),
    "raw_data_file_count": len(raw_files),
    "clean_file_count": len(clean_files),
    "same_relative_path_count": len(same_path),
    "compressed_sources": compressed,
    "compressed_clean_outputs": compressed_outputs,
    "invalid_json_removed_from_modeling_copy": invalid_json,
    "raw_parquet_replaced_by_clean_outputs": [
        "C_efficiency_evolution/data/train-00000-of-00001_clean.parquet",
        "C_efficiency_evolution/data/train-00000-of-00001_clean.csv",
    ],
    "generated_files_moved_from_raw_root": generated,
    "clean_data_policy": "后续建模只读取 F题_清洗后；方案与审计保存在 F题_初步方案。",
}
(AUDIT / "DATA_PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"raw_files": len(raw_files), "clean_files": len(clean_files), "visible_pages": len(visible_pages), "hidden_spans": spans, "invalid_json_removed": len(invalid_json), "generated_moved": generated}, ensure_ascii=False))
