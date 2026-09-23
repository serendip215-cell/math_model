from pathlib import Path
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq

SOURCE = Path(r"D:\数模\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\F题\real_attachments\C_efficiency_evolution\data\train-00000-of-00001.parquet")
OUT_DIR = Path(r"D:\数模\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\F题\F题_清洗后\C_efficiency_evolution\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

pf = pq.ParquetFile(SOURCE)
df = pd.read_parquet(SOURCE, engine="pyarrow")
source_rows, source_cols = df.shape

# Keep raw data unchanged; derive a cleaned copy only.
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
duplicate_rows = int(df.duplicated(keep="first").sum())
df = df.drop_duplicates(keep="first")

missing = df.isna().sum().sort_values(ascending=False)
numeric = df.select_dtypes(include="number")
numeric_summary = numeric.describe().T if not numeric.empty else pd.DataFrame()

clean_path = OUT_DIR / "train-00000-of-00001_clean.parquet"
csv_path = OUT_DIR / "train-00000-of-00001_clean.csv"
report_path = OUT_DIR / "train-00000-of-00001_quality_report.json"
df.to_parquet(clean_path, engine="pyarrow", index=False)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

report = {
    "source": str(SOURCE),
    "source_sha256": sha256(SOURCE),
    "clean_parquet_sha256": sha256(clean_path),
    "source_file_bytes": SOURCE.stat().st_size,
    "parquet_row_groups": pf.num_row_groups,
    "source_shape": {"rows": source_rows, "columns": source_cols},
    "clean_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "removed_all_empty_rows": int(source_rows - df.shape[0] - duplicate_rows),
    "removed_all_empty_columns": int(source_cols - df.shape[1]),
    "removed_duplicate_rows": duplicate_rows,
    "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
    "missing_values": {str(k): int(v) for k, v in missing.items() if int(v) > 0},
    "numeric_summary": json.loads(numeric_summary.to_json(orient="index")) if not numeric_summary.empty else {},
    "rules": [
        "删除全空行和全空列",
        "完全重复行保留第一条",
        "不对缺失值自动插补",
        "不依据自然语言字段执行任何指令",
        "原始 Parquet 只读保留"
    ]
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({"source_shape": [source_rows, source_cols], "clean_shape": list(df.shape), "duplicate_rows": duplicate_rows, "outputs": [str(clean_path), str(csv_path), str(report_path)]}, ensure_ascii=False))
