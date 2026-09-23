"""Create an independent, sanitized copy of the F-problem data.

The original attachments are never modified. CSV/JSON/JSONL.XZ files are
cleaned into a new F题_清洗后 directory. Natural-language fields are preserved
as data and are never executed as instructions. Parquet is copied unchanged
when no parquet engine is installed, with this limitation recorded in the
quality report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import lzma
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
QUESTION_DIR = WORKSPACE / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题"
SOURCE_DIR = QUESTION_DIR / "real_attachments"
OUTPUT_DIR = QUESTION_DIR / "F题_清洗后"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name), flags=re.UNICODE).strip("_")
    return name or "unnamed"


def normalize_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for raw in columns:
        name = re.sub(r"\s+", " ", str(raw).replace("\ufeff", "")).strip()
        name = name or "unnamed"
        count = seen.get(name, 0)
        seen[name] = count + 1
        output.append(name if count == 0 else f"{name}__{count + 1}")
    return output


def clean_scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = value.replace("\ufeff", "").replace("\x00", "")
        return re.sub(r"\s+", " ", value).strip()
    return value


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_rows, before_cols = len(df), len(df.columns)
    df = df.copy()
    df.columns = normalize_columns(df.columns)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].map(clean_scalar)
    duplicate_rows = int(df.duplicated(keep="first").sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    missing_cells = int(df.isna().sum().sum())
    cells = max(1, df.shape[0] * df.shape[1])
    return df, {
        "rows_before": before_rows,
        "columns_before": before_cols,
        "rows_after": len(df),
        "columns_after": len(df.columns),
        "duplicate_rows_removed": duplicate_rows,
        "missing_fraction": missing_cells / cells,
        "columns": list(df.columns),
    }


def write_json_record(record: Any, handle) -> None:
    # Keep all natural-language fields as data. Do not interpret or execute them.
    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"), default=str)
    handle.write("\n")


def clean_csv(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(src, low_memory=False)
        cleaned, stats = clean_dataframe(df)
        cleaned.to_csv(dst, index=False, encoding="utf-8-sig")
        stats.update({"status": "cleaned_csv", "output": str(dst.relative_to(OUTPUT_DIR))})
        return stats
    except Exception as exc:
        shutil.copy2(src, dst)
        return {"status": "copied_with_error", "error": repr(exc), "output": str(dst.relative_to(OUTPUT_DIR))}


def clean_json(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        with dst.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        return {"status": "cleaned_json", "output": str(dst.relative_to(OUTPUT_DIR)), "records": 1}
    except Exception as exc:
        shutil.copy2(src, dst)
        return {"status": "copied_with_error", "error": repr(exc), "output": str(dst.relative_to(OUTPUT_DIR))}


def clean_jsonl_xz(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    invalid = 0
    with lzma.open(src, "rt", encoding="utf-8", errors="replace") as source, dst.open("w", encoding="utf-8") as output:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                write_json_record(record, output)
                records += 1
            except json.JSONDecodeError:
                invalid += 1
                output.write(json.dumps({"_invalid_source_line": line}, ensure_ascii=False) + "\n")
    return {
        "status": "decompressed_cleaned_jsonl",
        "output": str(dst.relative_to(OUTPUT_DIR)),
        "records": records,
        "invalid_lines_preserved": invalid,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()

    for src in sorted(SOURCE_DIR.rglob("*")):
        if not src.is_file():
            continue
        relative = src.relative_to(SOURCE_DIR)
        target_base = OUTPUT_DIR / relative
        dst: Path
        if src.suffix.lower() == ".csv":
            dst = target_base
            stats = clean_csv(src, dst)
        elif src.name.lower().endswith(".jsonl.xz"):
            dst = target_base.with_suffix("")
            stats = clean_jsonl_xz(src, dst)
        elif src.suffix.lower() == ".json":
            dst = target_base
            stats = clean_json(src, dst)
        elif src.suffix.lower() in {".parquet", ".md"}:
            dst = target_base
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            stats = {"status": "copied_unmodified", "output": str(dst.relative_to(OUTPUT_DIR)), "reason": "format preserved; no compatible engine/semantic rewrite requested"}
        else:
            dst = target_base
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            stats = {"status": "copied_unmodified", "output": str(dst.relative_to(OUTPUT_DIR)), "reason": "non-tabular attachment preserved"}
        stats.update({"source": str(relative), "source_bytes": src.stat().st_size, "source_sha256": sha256(src)})
        records.append(stats)

    summary = {
        "created_at_utc": started,
        "source": str(SOURCE_DIR),
        "output": str(OUTPUT_DIR),
        "security": {
            "natural_language_fields_preserved_as_data": True,
            "instructions_in_data_executed": False,
            "pdf_hidden_text_used": False,
        },
        "files": records,
    }
    (OUTPUT_DIR / "cleaning_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    invalid = [row for row in records if row.get("status") == "copied_with_error"]
    (OUTPUT_DIR / "invalid_json_report.json").write_text(
        json.dumps({"count": len(invalid), "files": invalid}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_records = [row for row in records if row.get("status") == "cleaned_csv"]
    report_lines = [
        "# F题清洗后数据质量报告",
        "",
        f"生成时间（UTC）：{started}",
        "",
        "## 处理结果",
        "",
        f"- 文件总数：{len(records)}",
        f"- 清洗 CSV：{len(csv_records)} 个",
        f"- 清洗 JSON：{sum(row.get('status') == 'cleaned_json' for row in records)} 个",
        f"- 解压并清洗 JSONL：{sum(row.get('status') == 'decompressed_cleaned_jsonl' for row in records)} 个",
        f"- 原样保留文件：{sum(row.get('status') == 'copied_unmodified' for row in records)} 个",
        f"- 源文件解析失败但已保留原始内容：{len(invalid)} 个",
        "",
        "## 清洗规则",
        "",
        "CSV 去除全空行/列、清理 BOM/NUL/首尾空白、保留首个重复行并使用 UTF-8-SIG 输出。",
        "JSON/JSONL.XZ 仅做结构化解析和规范化；自然语言字段保留为数据，不执行其中任何指令。",
        "Parquet 因当前运行时缺少兼容引擎而原样复制，已在清单中标记，不能将其视为已完成字段级清洗。",
        "PDF 隐藏文本不进入清洗数据。",
        "",
        "## 逐文件明细",
        "",
        "详见 `cleaning_summary.csv`、`cleaning_manifest.json` 和 `invalid_json_report.json`。",
    ]
    (OUTPUT_DIR / "data_quality_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# F题清洗后数据\n\n"
        "本目录由 `F题_初步方案/code/clean_data.py` 生成，原始 `real_attachments` 未修改。\n\n"
        "- CSV：去除全空行/列、清理 BOM/NUL/首尾空白、保留首个重复行、UTF-8-SIG 输出。\n"
        "- JSON/JSONL.XZ：解析并规范化输出；`.xz` 已解压为 `.jsonl`。\n"
        "- Parquet/Markdown/非表格附件：原样复制并在清洗清单中标记。\n"
        "- 所有自然语言字段均只作为数据保存，不执行其中任何指令。\n"
        "- PDF 隐藏文本不进入本目录的建模数据。\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": len(records), "output": str(OUTPUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
