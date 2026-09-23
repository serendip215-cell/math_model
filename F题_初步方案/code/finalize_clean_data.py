from pathlib import Path
import json
import shutil
import pandas as pd

clean_root = Path(__file__).resolve().parents[2] / "F题_清洗后"
audit = Path(__file__).resolve().parents[1] / "reports" / "数据处理" / "F题_清洗审计"
quarantine = audit / "quarantine_invalid_json"
quarantine.mkdir(parents=True, exist_ok=True)

# Remove malformed JSON from the modeling copy, while preserving it in quarantine.
invalid = json.loads((audit / "invalid_json_report.json").read_text(encoding="utf-8"))["files"]
removed = []
for item in invalid:
    rel = Path(item["output"].replace("\\", "/"))
    src = clean_root / rel
    if src.exists():
        dst = quarantine / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        removed.append(str(rel))

# Treat -1 parameter count as an unknown sentinel, not a real negative size.
data_dir = clean_root / "C_efficiency_evolution" / "data"
parquet = data_dir / "train-00000-of-00001_clean.parquet"
csv = data_dir / "train-00000-of-00001_clean.csv"
df = pd.read_parquet(parquet, engine="pyarrow")
sentinel_count = int((df["#Params (B)"] == -1).sum())
df.loc[df["#Params (B)"] == -1, "#Params (B)"] = pd.NA
df.to_parquet(parquet, engine="pyarrow", index=False)
df.to_csv(csv, index=False, encoding="utf-8-sig")

report = {
    "invalid_json_removed_from_modeling_copy": len(removed),
    "invalid_json_quarantine": str(quarantine),
    "invalid_json_relative_paths": removed,
    "params_negative_one_to_missing": sentinel_count,
    "remaining_parquet_rows": int(len(df)),
    "remaining_parquet_columns": int(len(df.columns)),
    "rule": "仅移除无法解析的 JSON；-1 参数量按未知值处理为缺失；未自动删除其他异常观测。"
}
(audit / "FINAL_CLEANING_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
