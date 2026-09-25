"""Reproducible, descriptive raw-text check of A1 quality scores.

The sampled text is paired with its A1 score. Observable text flags are not
treated as ground-truth quality labels or as new versions of the 22 scorers.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT.parent / "F题_清洗后" / "A_data_value" / "slimpajama_quality_signal_sample.jsonl"
SCORES = PROJECT / "outputs" / "problem1" / "quality" / "results" / "quality_sample_scores.csv"
OUT = PROJECT / "outputs" / "problem1" / "quality" / "results"
REPORT = PROJECT / "reports" / "问题一" / "验证验收" / "RAW_TEXT_AUDIT_PROBLEM1.md"


def selected_rows(scores: pd.DataFrame) -> pd.DataFrame:
    subset = scores.loc[scores.dataset.eq("A1_sample")].copy()
    if len(subset) != 51230 or subset.id.duplicated().any():
        raise ValueError("A1 score contract or unique ID changed")
    chosen: list[dict] = []
    for domain, group in subset.groupby("domain", sort=True):
        used: set[str] = set()
        for quantile, label in [(0.10, "low_q10"), (0.50, "median_q50"),
                                (0.90, "high_q90")]:
            target = group.quality_score.quantile(quantile)
            ranked = group.assign(distance=(group.quality_score - target).abs()).sort_values(
                ["distance", "id"])
            for _, row in ranked.iterrows():
                if row.id in used:
                    continue
                chosen.append({"domain": domain, "stratum": label,
                               "id": row.id, "quality_score": row.quality_score,
                               "conflict_score": row.conflict_score,
                               "score_content_length": int(row.content_length)})
                used.add(row.id)
                if sum(x["domain"] == domain and x["stratum"] == label for x in chosen) == 2:
                    break
        conflict = group.loc[~group.id.isin(used)].sort_values(
            ["conflict_score", "id"], ascending=[False, True]).iloc[0]
        chosen.append({"domain": domain, "stratum": "highest_conflict",
                       "id": conflict.id, "quality_score": conflict.quality_score,
                       "conflict_score": conflict.conflict_score,
                       "score_content_length": int(conflict.content_length)})
    chosen_frame = pd.DataFrame(chosen)
    if len(chosen_frame) != 49 or chosen_frame.id.duplicated().any():
        raise ValueError("Raw text audit selection changed")
    return chosen_frame


def text_features(content: str) -> dict:
    length = len(content)
    visible = content[:360].replace("\r", " ").replace("\n", " ").replace("|", "/")
    return {
        "raw_text_chars": length,
        "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "alphabetic_fraction": sum(ch.isalpha() for ch in content) / max(length, 1),
        "control_char_count": sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in content),
        "url_count": len(re.findall(r"https?://|www\.", content, flags=re.I)),
        "html_tag_count": len(re.findall(r"</?[a-z][a-z0-9]*(?:\s[^<>]*)?>", content, flags=re.I)),
        "replacement_char_count": content.count("\ufffd"),
        "preview_360_chars": visible,
    }


def main() -> None:
    selected = selected_rows(pd.read_csv(SCORES))
    wanted = set(selected.id)
    found = {}
    with SOURCE.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["id"] in wanted:
                found[record["id"]] = {
                    "raw_source_domain": record.get("_source_domain"),
                    **text_features(record.get("content", ""))}
    if set(found) != wanted:
        raise ValueError(f"Missing selected A1 raw texts: {sorted(wanted - set(found))}")
    observed = pd.DataFrame.from_dict(found, orient="index").rename_axis("id").reset_index()
    audit = selected.merge(observed, on="id", validate="one_to_one")
    audit["content_length_matches"] = audit.score_content_length.eq(audit.raw_text_chars)
    audit["domain_matches"] = audit.domain.eq(audit.raw_source_domain)
    # Preserve reproducibility by ID/hash without republishing the sampled text.
    audit.drop(columns=["preview_360_chars"]).to_csv(
        OUT / "raw_text_stratified_audit.csv", index=False, encoding="utf-8-sig")
    summary = audit.groupby(["domain", "stratum"], as_index=False).agg(
        sampled_records=("id", "size"),
        quality_score_median=("quality_score", "median"),
        raw_text_chars_median=("raw_text_chars", "median"),
        alphabetic_fraction_median=("alphabetic_fraction", "median"),
        url_count_total=("url_count", "sum"),
        html_tag_count_total=("html_tag_count", "sum"),
        records_with_control_chars=("control_char_count", lambda s: int(s.gt(0).sum())),
        records_with_replacement_chars=("replacement_char_count", lambda s: int(s.gt(0).sum())))
    summary.to_csv(OUT / "raw_text_stratified_summary.csv", index=False, encoding="utf-8-sig")
    case_notes = {
        "BkiUfpo5qhDCnXuZvcNm": ("arxiv", "开头为纤维丛及拉格朗日密度的数学论文表述；低分位不等于内容虚假或无用。"),
        "BkiUftPxK1yAga6JuYW_": ("book", "高分位样本开头仍含版权和出版前置信息；高分不等于没有模板文字。"),
        "BkiUdDk25V5ihkoTWuOy": ("c4", "高冲突样本的开头是连续的搜索关键词短语；这一段缺乏连贯叙述。"),
        "BkiUc4s25V5ioeHBl06k": ("github", "低分位样本是简短但有结构的数据库迁移代码；语言质量分不能等同代码正确性。"),
        "BkiUc4rxK7Tt6AlxDb_9": ("wikipedia", "低分位样本开头是意大利语条目；英语向指标可能影响跨语言比较。"),
        "BkiUeA3xK7Dgs_cY6OuR": ("wikipedia", "高分位样本开头也是非英语条目；不能仅以语言解释所有分数差。"),
    }
    case_review = audit[audit.id.isin(case_notes)].copy()
    case_review["visible_excerpt_observation"] = case_review.id.map(lambda item: case_notes[item][1])
    case_review["review_scope"] = "first_360_characters_only"
    if len(case_review) != len(case_notes) or any(
            row.domain != case_notes[row.id][0] for row in case_review.itertuples()):
        raise ValueError("Qualitative case IDs no longer match A1 samples")
    case_review[["domain", "stratum", "id", "quality_score", "conflict_score",
                 "visible_excerpt_observation", "review_scope"]].to_csv(
        OUT / "raw_text_case_review.csv", index=False, encoding="utf-8-sig")
    source_digest = hashlib.sha256()
    with SOURCE.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            source_digest.update(block)
    lines = ["# 问题一 A1 原始文本抽样核验", "",
        "本核验从与质量分数成对的 A1 原始文本中选样。每个领域取质量分数 10%、50%、90% 分位附近各两条，再取冲突度最高的一条，共 49 条。抽样按分数分层且刻意挑选冲突案例，不代表总体随机样本。",
        "", f"- A1 原始文件：`{SOURCE.name}`；SHA-256：`{source_digest.hexdigest()}`。",
        f"- 样本领域数：{audit.domain.nunique()}；抽样条数：{len(audit)}。",
        f"- ID、领域、原始文本长度与评分表一致：{bool(audit.domain_matches.all() and audit.content_length_matches.all())}。",
        "- 逐条 ID、分数、文本哈希和客观字符特征见 `raw_text_stratified_audit.csv`；分层描述见 `raw_text_stratified_summary.csv`。原始文本可由题目附件按 ID 复核，不在产物中重复发布。",
        "- 六条定性案例的可见段落判读见 `raw_text_case_review.csv`。例如 c4 高冲突案例开头是搜索词堆砌；一本高分位书籍仍有版权前言；Wikipedia 高、低分位均出现非英语条目。这些是具体样本观察，不是总体比例或质量真值。",
        "", "## 解释边界", "",
        "字符比例、URL、HTML 标签和异常字符只能检查可见文本形态，不能证明知识正确、无偏、无版权风险或训练效益。A1 文本与评分可逐条追溯，但这里没有独立专家标签，也没有把 49 条当作总体准确率估计。该抽查不改变质量模型、17 域代理或推荐配比。",
        "", "本次定性案例只判读前 360 字；完整文档的独立盲评尚未完成。因此不声称评分已获得独立人工语义验证，也不据此修改现有模型。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"samples": len(audit), "domains": audit.domain.nunique(),
                      "length_matches": bool(audit.content_length_matches.all()),
                      "domain_matches": bool(audit.domain_matches.all())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
