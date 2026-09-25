from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit, softmax
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SEED = 20260923
RNG = np.random.default_rng(SEED)
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后"
A = DATA / "A_data_value"
OUTPUT_ROOT = PROJECT / "outputs" / "problem1"
QUALITY_OUT = OUTPUT_ROOT / "quality" / "results"
QUALITY_FIG = OUTPUT_ROOT / "quality" / "figures"
CACHE = OUTPUT_ROOT / "quality" / "cache"
MIXTURE_OUT = OUTPUT_ROOT / "mixture" / "results"
MIXTURE_FIG = OUTPUT_ROOT / "mixture" / "figures"
for directory in (OUTPUT_ROOT, QUALITY_OUT, QUALITY_FIG, CACHE, MIXTURE_OUT, MIXTURE_FIG):
    directory.mkdir(parents=True, exist_ok=True)

QUALITY_FILES = {
    "A1_sample": A / "slimpajama_quality_signal_sample.jsonl",
    "A2_arxiv": A / "slimpajama_quality_extended" / "arxiv_part-6777d8857c6e-000486.jsonl",
    "A3_github": A / "slimpajama_quality_extended" / "github_part-6777d8857c6e-000275.jsonl",
}

METRICS = [
    "dsir_books", "dsir_wiki", "dsir_math", "fineweb_edu", "fluency_en",
    "qurater", "ad_en", "modernbert_professionalism", "modernbert_readability",
    "modernbert_reasoning", "modernbert_cleanliness", "rps_doc_word_count",
    "rps_doc_num_sentences", "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]

GROUPS = {
    "知识价值": ["dsir_books", "dsir_wiki", "dsir_math", "fineweb_edu", "qurater", "modernbert_reasoning"],
    "语言表达": ["fluency_en", "modernbert_readability", "modernbert_professionalism"],
    "洁净规范": ["ad_en", "modernbert_cleanliness", "rps_doc_frac_no_alph_words",
             "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
             "rps_lines_uppercase_letter_fraction", "rps_lines_numerical_chars_fraction",
             "rps_lines_ending_with_terminal_punctution_mark"],
    "结构丰富": ["rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
             "rps_doc_frac_unique_words", "rps_doc_mean_word_length"],
}

DIRECTIONS = {
    **{m: "benefit" for m in ["dsir_books", "dsir_wiki", "dsir_math", "fineweb_edu",
                                "fluency_en", "qurater", "ad_en", "modernbert_professionalism",
                                "modernbert_readability", "modernbert_reasoning", "modernbert_cleanliness",
                                "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
                                "rps_lines_ending_with_terminal_punctution_mark"]},
    **{m: "cost" for m in ["rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
                             "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
                             "rps_lines_numerical_chars_fraction"]},
    **{m: "target" for m in ["rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"]},
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Hash large inputs without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame, include_index=False):
    """Render a small DataFrame as Markdown without optional dependencies."""
    table = frame.reset_index() if include_index else frame.copy()
    table = table.fillna("")
    headers = [str(column).replace("|", "\\|") for column in table.columns]
    rows = []
    for values in table.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|") for value in values])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def prob_class1(values):
    if not isinstance(values, list) or len(values) != 2:
        return np.nan
    x = np.asarray(values, dtype=float)
    return float(softmax(x)[1]) if np.isfinite(x).all() else np.nan


def expected_level(values):
    if not isinstance(values, list) or len(values) != 6:
        return np.nan
    x = np.asarray(values, dtype=float)
    return float(np.dot(softmax(x), np.arange(6)) / 5.0) if np.isfinite(x).all() else np.nan


def flatten_record(obj):
    row = {}
    for metric in METRICS:
        value = obj.get(metric)
        if metric in {"fluency_en", "ad_en"}:
            row[metric] = prob_class1(value)
        elif metric.startswith("modernbert_"):
            row[metric] = expected_level(value)
        elif metric == "fineweb_edu":
            row[metric] = finite(value[0]) if isinstance(value, list) and value else np.nan
        elif metric == "qurater":
            # Four named quality dimensions; map their direct scores monotonically to (0,1).
            if isinstance(value, list) and len(value) == 4:
                vals = np.asarray(value, dtype=float)
                row[metric] = float(np.mean(expit(vals))) if np.isfinite(vals).all() else np.nan
            else:
                row[metric] = np.nan
        else:
            row[metric] = finite(value)
    return row


def build_quality_cache():
    cache_file = CACHE / "quality_features.csv"
    audit_file = CACHE / "quality_ingest_audit.json"
    if cache_file.exists() and audit_file.exists():
        return cache_file, json.loads(audit_file.read_text(encoding="utf-8"))
    audit = {"files": {}, "invalid_json_lines": 0, "field_shape_errors": Counter()}
    header = ["dataset", "domain", "id", "content_length"] + METRICS
    with cache_file.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()
        for dataset, path in QUALITY_FILES.items():
            count = 0
            domains = Counter()
            ids = set()
            duplicate_ids = 0
            with path.open("r", encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        audit["invalid_json_lines"] += 1
                        continue
                    count += 1
                    sample_id = str(obj.get("id", ""))
                    if sample_id in ids:
                        duplicate_ids += 1
                    ids.add(sample_id)
                    if dataset == "A1_sample":
                        domain = str(obj.get("_source_domain") or "unknown").lower()
                    else:
                        domain = "arxiv" if "arxiv" in dataset else "github"
                    domains[domain] += 1
                    values = flatten_record(obj)
                    writer.writerow({
                        "dataset": dataset,
                        "domain": domain,
                        "id": sample_id,
                        "content_length": len(obj.get("content", "")) if isinstance(obj.get("content"), str) else np.nan,
                        **values,
                    })
            audit["files"][dataset] = {
                "path": str(path.relative_to(DATA)), "records": count,
                "duplicate_ids_within_file": duplicate_ids, "domains": dict(domains),
                "sha256": sha256_file(path),
            }
    audit["field_shape_errors"] = dict(audit["field_shape_errors"])
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_file, audit


def scale_quality(df):
    ref = df[df["dataset"] == "A1_sample"]
    scaled = pd.DataFrame(index=df.index)
    rules = []
    for metric in METRICS:
        train = pd.to_numeric(ref[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        low, high = train.quantile([0.01, 0.99])
        values = pd.to_numeric(df[metric], errors="coerce")
        direction = DIRECTIONS[metric]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            scaled[metric] = np.nan
            rules.append({"metric": metric, "direction": direction, "status": "no_variation"})
            continue
        if direction == "target":
            transformed_train = np.log1p(np.maximum(train, 0)) if metric != "rps_doc_mean_word_length" else train
            transformed = np.log1p(np.maximum(values, 0)) if metric != "rps_doc_mean_word_length" else values
            center = float(transformed_train.median())
            q25, q75 = transformed_train.quantile([0.25, 0.75])
            spread = max(float(q75 - q25), 1e-9) * 1.5
            score = np.exp(-0.5 * ((transformed - center) / spread) ** 2)
            rules.append({"metric": metric, "direction": direction, "low": float(low), "high": float(high),
                          "target_center": center, "target_spread": spread, "status": "used"})
        else:
            score = ((values - low) / (high - low)).clip(0, 1)
            if direction == "cost":
                score = 1 - score
            rules.append({"metric": metric, "direction": direction, "low": float(low), "high": float(high), "status": "used"})
        scaled[metric] = score
    return scaled, pd.DataFrame(rules)


def critic_group_weights(scaled, ref_mask):
    weights = {}
    rows = []
    for group, metrics in GROUPS.items():
        x = scaled.loc[ref_mask, metrics]
        corr = x.corr().fillna(0).abs()
        sd = x.std(skipna=True).fillna(0)
        info = sd * (1 - corr).sum(axis=1)
        if info.sum() <= 0:
            local = pd.Series(1 / len(metrics), index=metrics)
        else:
            local = info / info.sum()
        for metric in metrics:
            weights[metric] = float(local[metric]) / len(GROUPS)
            rows.append({"metric": metric, "group": group, "within_group_weight": float(local[metric]),
                         "global_weight": weights[metric]})
    return weights, pd.DataFrame(rows)


def weighted_group_scores(scaled, metric_weights):
    scores = pd.DataFrame(index=scaled.index)
    for group, metrics in GROUPS.items():
        w = np.array([metric_weights[m] for m in metrics], dtype=float)
        vals = scaled[metrics].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        denom = (valid * w).sum(axis=1)
        num = np.nansum(vals * w, axis=1)
        scores[group] = np.divide(num, denom, out=np.full(len(vals), np.nan), where=denom > 0)
    return scores


def huber_location(group_scores, delta=0.15):
    x = group_scores.to_numpy(dtype=float)
    q = np.nanmedian(x, axis=1)
    for _ in range(12):
        residual = x - q[:, None]
        robust = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        robust[~np.isfinite(x)] = 0
        denom = robust.sum(axis=1)
        q_new = np.divide(np.nansum(x * robust, axis=1), denom, out=q.copy(), where=denom > 0)
        if np.nanmax(np.abs(q_new - q)) < 1e-9:
            break
        q = q_new
    return q


def bootstrap_ci(values, n_boot=300):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(RNG.choice(arr, size=len(arr), replace=True))
    return tuple(np.quantile(meds, [0.025, 0.975]))


def conflict_cause_analysis(df, scaled, groups, score_out):
    """Describe recurring group conflicts without assigning hand-written domain rules."""
    rows = []
    a1_mask = score_out["dataset"].eq("A1_sample")
    for domain in sorted(score_out.loc[a1_mask, "domain"].unique()):
        domain_mask = a1_mask & score_out["domain"].eq(domain)
        high_total = int((domain_mask & score_out["high_conflict"]).sum())
        for pair in score_out.loc[domain_mask & score_out["high_conflict"], "primary_conflict_pair"].unique():
            pair_mask = domain_mask & score_out["high_conflict"] & score_out["primary_conflict_pair"].eq(pair)
            if not pair_mask.any():
                continue
            left, right = pair.split("—", 1)
            delta = groups.loc[pair_mask, left] - groups.loc[pair_mask, right]
            left_high = bool((delta >= 0).mean() >= 0.5)
            high_group, low_group = (left, right) if left_high else (right, left)
            oriented = pair_mask.copy()
            oriented.loc[pair_mask] = delta.ge(0).to_numpy() if left_high else delta.lt(0).to_numpy()
            if not oriented.any():
                oriented = pair_mask
            high_metric_means = scaled.loc[oriented, GROUPS[high_group]].mean()
            low_metric_means = scaled.loc[oriented, GROUPS[low_group]].mean()
            rows.append({
                "domain": domain, "pair": pair, "count": int(pair_mask.sum()),
                "share_among_domain_high_conflict": float(pair_mask.sum() / max(high_total, 1)),
                "dominant_orientation": f"{high_group}高—{low_group}低",
                "orientation_share": float(oriented.sum() / pair_mask.sum()),
                "mean_group_gap": float((groups.loc[oriented, high_group] - groups.loc[oriented, low_group]).mean()),
                "high_group_mean": float(groups.loc[oriented, high_group].mean()),
                "low_group_mean": float(groups.loc[oriented, low_group].mean()),
                "high_driver_metric": str(high_metric_means.idxmax()),
                "high_driver_score": float(high_metric_means.max()),
                "low_driver_metric": str(low_metric_means.idxmin()),
                "low_driver_score": float(low_metric_means.min()),
            })
    result = pd.DataFrame(rows).sort_values(["domain", "count"], ascending=[True, False])
    result.to_csv(QUALITY_OUT / "conflict_cause_profile.csv", index=False, encoding="utf-8-sig")
    return result


def quality_analysis():
    cache_file, audit = build_quality_cache()
    df = pd.read_csv(cache_file, encoding="utf-8-sig", low_memory=False)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    scaled, rules = scale_quality(df)
    ref_mask = df["dataset"].eq("A1_sample")
    weights, weight_table = critic_group_weights(scaled, ref_mask)
    groups = weighted_group_scores(scaled, weights)
    q_huber = huber_location(groups)
    q_equal = scaled.mean(axis=1, skipna=True).to_numpy()
    group_array = groups.to_numpy(dtype=float)
    pair_names, pair_values = [], []
    group_names = list(groups.columns)
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            pair_names.append(f"{group_names[i]}—{group_names[j]}")
            pair_values.append(np.abs(group_array[:, i] - group_array[:, j]))
    pair_values = np.column_stack(pair_values)
    conflict = np.nanmean(pair_values, axis=1)
    threshold = float(np.nanquantile(conflict[ref_mask], 0.90))
    high_conflict = conflict >= threshold
    primary_pair = np.array(pair_names, dtype=object)[np.nanargmax(np.where(np.isfinite(pair_values), pair_values, -np.inf), axis=1)]

    score_out = df[["dataset", "domain", "id", "content_length"]].copy()
    score_out["quality_score"] = q_huber
    score_out["equal_weight_score"] = q_equal
    score_out["conflict_score"] = conflict
    score_out["high_conflict"] = high_conflict
    score_out["primary_conflict_pair"] = primary_pair
    score_out.to_csv(QUALITY_OUT / "quality_sample_scores.csv", index=False, encoding="utf-8-sig")

    domain_rows = []
    for (dataset, domain), sub in score_out.groupby(["dataset", "domain"], dropna=False):
        lo, hi = bootstrap_ci(sub["quality_score"].to_numpy())
        domain_rows.append({
            "dataset": dataset, "domain": domain, "n": len(sub),
            "quality_mean": sub["quality_score"].mean(), "quality_median": sub["quality_score"].median(),
            "quality_q25": sub["quality_score"].quantile(0.25), "quality_q75": sub["quality_score"].quantile(0.75),
            "median_ci_low": lo, "median_ci_high": hi,
            "conflict_rate": sub["high_conflict"].mean(), "conflict_mean": sub["conflict_score"].mean(),
        })
    domain_summary = pd.DataFrame(domain_rows).sort_values(["dataset", "quality_median"], ascending=[True, False])
    domain_summary.to_csv(QUALITY_OUT / "quality_domain_summary.csv", index=False, encoding="utf-8-sig")
    extension_rows = []
    for domain, extension_dataset in [("arxiv", "A2_arxiv"), ("github", "A3_github")]:
        base = domain_summary[(domain_summary["dataset"] == "A1_sample") & (domain_summary["domain"] == domain)].iloc[0]
        ext = domain_summary[(domain_summary["dataset"] == extension_dataset) & (domain_summary["domain"] == domain)].iloc[0]
        extension_rows.append({
            "domain": domain, "a1_n": int(base["n"]), "extension_n": int(ext["n"]),
            "a1_quality_median": float(base["quality_median"]),
            "extension_quality_median": float(ext["quality_median"]),
            "median_difference_extension_minus_a1": float(ext["quality_median"] - base["quality_median"]),
            "a1_conflict_rate": float(base["conflict_rate"]),
            "extension_conflict_rate": float(ext["conflict_rate"]),
        })
    extension_comparison = pd.DataFrame(extension_rows)
    extension_comparison.to_csv(QUALITY_OUT / "quality_extension_comparison.csv", index=False, encoding="utf-8-sig")

    duplicate_summary = {
        "records": int(len(df)), "invalid_json_lines": int(audit["invalid_json_lines"]),
        "within_dataset_duplicate_ids": int(df.duplicated(["dataset", "id"]).sum()),
        "cross_dataset_repeated_ids": int(df.groupby("id")["dataset"].nunique().gt(1).sum()),
        "missing_fraction_by_metric": {m: float(df[m].isna().mean()) for m in METRICS},
        "conflict_threshold_a1_p90": threshold,
    }
    (QUALITY_OUT / "quality_data_audit.json").write_text(json.dumps(duplicate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rules.merge(weight_table, on="metric", how="left").to_csv(QUALITY_OUT / "quality_metric_rules_weights.csv", index=False, encoding="utf-8-sig")

    conflict_rows = []
    for dataset in score_out["dataset"].unique():
        mask = score_out["dataset"].eq(dataset) & score_out["high_conflict"]
        counts = score_out.loc[mask, "primary_conflict_pair"].value_counts()
        for pair, count in counts.items():
            conflict_rows.append({"dataset": dataset, "pair": pair, "count": int(count),
                                  "share_among_high_conflict": float(count / max(mask.sum(), 1))})
    pd.DataFrame(conflict_rows).to_csv(QUALITY_OUT / "conflict_pair_summary.csv", index=False, encoding="utf-8-sig")

    sensitivity = []
    for delta in [0.10, 0.15, 0.20]:
        q_alt = huber_location(groups, delta)
        sensitivity.append({"variant": f"Huber_delta_{delta}", "correlation_with_main": float(np.corrcoef(q_huber, q_alt)[0, 1]),
                            "mean_abs_difference": float(np.mean(np.abs(q_huber - q_alt)))})
    sensitivity.append({"variant": "equal_weight_mean", "correlation_with_main": float(np.corrcoef(q_huber, q_equal)[0, 1]),
                        "mean_abs_difference": float(np.mean(np.abs(q_huber - q_equal)))})
    pd.DataFrame(sensitivity).to_csv(QUALITY_OUT / "quality_sensitivity.csv", index=False, encoding="utf-8-sig")

    conflict_causes = conflict_cause_analysis(df, scaled, groups, score_out)
    plot_quality(domain_summary, score_out, groups, ref_mask)
    return {
        "audit": duplicate_summary, "domain_summary": domain_summary, "extension_comparison": extension_comparison,
        "conflict_causes": conflict_causes,
        "sensitivity": sensitivity, "quality_score_mean": float(np.mean(q_huber)),
        "quality_score_std": float(np.std(q_huber)),
    }


def configure_plots():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42, "figure.dpi": 140,
    })


def plot_quality(domain_summary, score_out, groups, ref_mask):
    configure_plots()
    a1 = domain_summary[domain_summary["dataset"] == "A1_sample"].sort_values("quality_median")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(a1))
    ax.barh(y, a1["quality_median"], color="#3b82f6")
    ax.errorbar(a1["quality_median"], y,
                xerr=[a1["quality_median"] - a1["median_ci_low"], a1["median_ci_high"] - a1["quality_median"]],
                fmt="none", ecolor="#111827", capsize=3)
    ax.set_yticks(y, a1["domain"])
    ax.set_xlabel("领域质量中位数 Q")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(QUALITY_FIG / "quality_by_domain.pdf", bbox_inches="tight")
    plt.close(fig)

    corr = groups.loc[ref_mask].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="相关系数")
    fig.tight_layout()
    fig.savefig(QUALITY_FIG / "quality_group_correlation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, domain, extension in zip(axes, ["arxiv", "github"], ["A2_arxiv", "A3_github"]):
        sampled = score_out[(score_out["dataset"] == "A1_sample") & (score_out["domain"] == domain)]
        extended = score_out[score_out["dataset"] == extension]
        ax.hist(sampled["quality_score"], bins=bins, density=True, histtype="step",
                linewidth=1.7, label=f"A1 {domain} 抽样")
        ax.hist(extended["quality_score"], bins=bins, density=True, histtype="step",
                linewidth=1.7, label=f"{extension[:2]} {domain} 扩展")
        ax.set_title(domain)
        ax.set_xlabel("样本质量分 Q")
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set_ylabel("密度")
    fig.tight_layout()
    fig.savefig(QUALITY_FIG / "quality_score_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    a1_conflict = domain_summary[domain_summary["dataset"] == "A1_sample"].sort_values("conflict_rate")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(np.arange(len(a1_conflict)), a1_conflict["conflict_rate"], color="#f59e0b")
    ax.set_yticks(np.arange(len(a1_conflict)), a1_conflict["domain"])
    ax.set_xlabel("高冲突样本占比")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(QUALITY_FIG / "conflict_rate_by_domain.pdf", bbox_inches="tight")
    plt.close(fig)


def ilr_transform(p, basis, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, eps)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p) @ basis.T


def ilr_inverse(z, basis):
    logp = np.asarray(z) @ basis
    logp = logp - np.max(logp)
    p = np.exp(logp)
    return p / p.sum()


def load_pair(mixture_name, loss_name):
    mix = pd.read_csv(A / "regmix_tables" / mixture_name, encoding="utf-8-sig")
    loss = pd.read_csv(A / "regmix_tables" / loss_name, encoding="utf-8-sig")
    if mix["index"].duplicated().any() or loss["index"].duplicated().any():
        raise ValueError(f"duplicate index in {mixture_name} or {loss_name}")
    merged = mix.merge(loss, on="index", how="inner", validate="one_to_one")
    if len(merged) != len(mix) or len(merged) != len(loss):
        raise ValueError(f"unmatched index in {mixture_name} and {loss_name}")
    p_cols = [c for c in mix.columns if c.startswith("train_the_pile_")]
    y_cols = [c for c in loss.columns if c.startswith("metric/the_pile_")]
    p = merged[p_cols].to_numpy(dtype=float)
    y = merged[y_cols].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any():
        raise ValueError(f"non-finite/negative values in {mixture_name}")
    sums = p.sum(axis=1)
    audit = {"name": mixture_name, "rows": len(merged), "p_columns": len(p_cols), "loss_columns": len(y_cols),
             "sum_min": float(sums.min()), "sum_max": float(sums.max()), "sum_max_abs_error": float(np.max(np.abs(sums - 1)))}
    p = p / sums[:, None]
    return merged["index"].to_numpy(), p, y, p_cols, y_cols, audit


def metric_rows(split, y, pred, y_cols):
    rows = []
    for j, col in enumerate(y_cols):
        error = pred[:, j] - y[:, j]
        centered_error = (pred[:, j] - pred[:, j].mean()) - (y[:, j] - y[:, j].mean())
        rho = spearmanr(y[:, j], pred[:, j], nan_policy="omit").statistic
        rows.append({"split": split, "target": col.replace("metric/the_pile_", "").replace("_val_loss", ""),
                     "n": len(y), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
                     "bias": float(np.mean(error)),
                     "centered_mae": float(np.mean(np.abs(centered_error))),
                     "centered_rmse": float(np.sqrt(np.mean(centered_error ** 2))),
                     "spearman": float(rho) if np.isfinite(rho) else np.nan})
    actual_avg, pred_avg = y.mean(axis=1), pred.mean(axis=1)
    macro_error = pred_avg - actual_avg
    centered_macro_error = (pred_avg - pred_avg.mean()) - (actual_avg - actual_avg.mean())
    rho = spearmanr(actual_avg, pred_avg).statistic
    rows.append({"split": split, "target": "macro_average", "n": len(y),
                 "mae": float(np.mean(np.abs(macro_error))),
                 "rmse": float(np.sqrt(np.mean(macro_error ** 2))),
                 "bias": float(np.mean(macro_error)),
                 "centered_mae": float(np.mean(np.abs(centered_macro_error))),
                 "centered_rmse": float(np.sqrt(np.mean(centered_macro_error ** 2))),
                 "spearman": float(rho) if np.isfinite(rho) else np.nan})
    error = pred - y
    centered_error = (pred - pred.mean(axis=0)) - (y - y.mean(axis=0))
    rows.append({"split": split, "target": "pooled_domain", "n": len(y),
                 "mae": float(np.mean(np.abs(error))),
                 "rmse": float(np.sqrt(np.mean(error ** 2))),
                 "bias": float(np.mean(error)),
                 "centered_mae": float(np.mean(np.abs(centered_error))),
                 "centered_rmse": float(np.sqrt(np.mean(centered_error ** 2))),
                 "spearman": np.nan})
    return rows


def fit_ridge_cv(features, targets, alphas):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": alphas}, scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
    ).fit(features, targets)
    return search.best_estimator_, float(-search.best_score_), float(search.best_params_["ridge__alpha"])


def quality_feature_matrix(p, quality_vector):
    known = np.isfinite(quality_vector)
    coverage = p[:, known].sum(axis=1)
    quality_mass = (p[:, known] * quality_vector[known]).sum(axis=1)
    quality_average = quality_mass / np.maximum(coverage, 1e-12)
    return np.column_stack([coverage, quality_average])


def scale_feature_matrix(p, scale, basis):
    z = ilr_transform(p, basis)
    s = (math.log10(scale) - 6.0) / 3.0
    return np.column_stack([z, np.full(len(p), s), np.full(len(p), s * s), z * s])


def macro_summary(label, y, pred, role):
    actual, fitted = y.mean(axis=1), pred.mean(axis=1)
    error = fitted - actual
    centered = (fitted - fitted.mean()) - (actual - actual.mean())
    rho = spearmanr(actual, fitted).statistic
    return {
        "evaluation": label, "role": role, "n": len(y),
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)), "centered_rmse": float(np.sqrt(np.mean(centered ** 2))),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def composition_analysis(quality):
    specs = {
        "train_1m": ("train_mixture_1m.csv", "train_pile_loss_1m.csv"),
        "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
        "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
        "test_1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
        "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
        "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv"),
    }
    datasets, audits = {}, []
    for split, names in specs.items():
        datasets[split] = load_pair(*names)
        audits.append(datasets[split][-1])
    _, p_train, y_train, p_cols, y_cols, _ = datasets["train_1m"]
    basis = helmert(len(p_cols), full=False)
    z_train = ilr_transform(p_train, basis)
    alphas = np.logspace(-4, 4, 33)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(z_train)

    mapping = pd.read_csv(A / "domain_mapping_guide.csv", encoding="utf-8-sig")
    q_lookup = (quality["domain_summary"].query("dataset == 'A1_sample'")
                .set_index("domain")["quality_median"].to_dict())
    mapping["quality_score"] = mapping["quality_domain"].map(q_lookup)
    mapping["mapping_available"] = mapping["quality_score"].notna()
    mapping.to_csv(MIXTURE_OUT / "quality_domain_mapping.csv", index=False, encoding="utf-8-sig")
    mixture_domains = [c.replace("train_the_pile_", "") for c in p_cols]
    quality_vector = (mapping.set_index("mixture_domain").reindex(mixture_domains)["quality_score"]
                      .to_numpy(dtype=float))
    q_features_train = quality_feature_matrix(p_train, quality_vector)

    linear, linear_cv_mse, linear_alpha = fit_ridge_cv(z_train, y_train, alphas)
    quadratic, quadratic_cv_mse, quadratic_alpha = fit_ridge_cv(x_poly, y_train, alphas)
    quadratic_q, quadratic_q_cv_mse, quadratic_q_alpha = fit_ridge_cv(
        np.column_stack([x_poly, q_features_train]), y_train, alphas)

    _, p_test1, y_test1, *_ = datasets["test_1m"]
    z_test1 = ilr_transform(p_test1, basis)
    test_poly = poly.transform(z_test1)
    pred_linear = linear.predict(z_test1)
    pred_quad = quadratic.predict(test_poly)
    pred_quad_q = quadratic_q.predict(np.column_stack([test_poly, quality_feature_matrix(p_test1, quality_vector)]))
    if quadratic_cv_mse <= 0.98 * linear_cv_mse:
        selected_name, selected, selected_cv_mse = "quadratic_ilr_ridge", quadratic, quadratic_cv_mse
    else:
        selected_name, selected, selected_cv_mse = "linear_ilr_ridge", linear, linear_cv_mse

    selected_test_pred = pred_quad if selected_name.startswith("quadratic") else pred_linear
    selected_test_rmse = float(np.sqrt(np.mean(
        (selected_test_pred.mean(axis=1) - y_test1.mean(axis=1)) ** 2)))
    quality_test_rmse = float(np.sqrt(np.mean(
        (pred_quad_q.mean(axis=1) - y_test1.mean(axis=1)) ** 2)))
    quality_adopted = bool(
        quadratic_q_cv_mse <= 0.98 * selected_cv_mse
        and quality_test_rmse <= 0.98 * selected_test_rmse
    )
    if quality_adopted:
        selected_name, selected, selected_cv_mse = "quadratic_ilr_ridge_quality", quadratic_q, quadratic_q_cv_mse

    def predict_p(p):
        z = ilr_transform(np.atleast_2d(p), basis)
        if selected_name == "linear_ilr_ridge":
            return selected.predict(z)
        features = poly.transform(z)
        if selected_name.endswith("_quality"):
            features = np.column_stack([features, quality_feature_matrix(np.atleast_2d(p), quality_vector)])
        return selected.predict(features)

    rows = []
    prediction_frames = []
    for split, data in datasets.items():
        indices, p, y, *_ = data
        pred = predict_p(p)
        rows.extend(metric_rows(split, y, pred, y_cols))
        prediction_frames.append(pd.DataFrame({
            "split": split, "index": indices,
            "actual_macro_loss": y.mean(axis=1), "predicted_macro_loss": pred.mean(axis=1),
        }))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(MIXTURE_OUT / "mixture_model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(MIXTURE_OUT / "mixture_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(MIXTURE_OUT / "mixture_data_audit.csv", index=False, encoding="utf-8-sig")

    quality_ablation = pd.DataFrame([
        {"model": "linear_ilr_ridge", "cv_mse": linear_cv_mse,
         "test_1m_macro_rmse": float(np.sqrt(np.mean((pred_linear.mean(axis=1) - y_test1.mean(axis=1)) ** 2))),
         "test_1m_pooled_rmse": float(np.sqrt(np.mean((pred_linear - y_test1) ** 2))), "quality_features": False},
        {"model": "quadratic_ilr_ridge", "cv_mse": quadratic_cv_mse,
         "test_1m_macro_rmse": float(np.sqrt(np.mean((pred_quad.mean(axis=1) - y_test1.mean(axis=1)) ** 2))),
         "test_1m_pooled_rmse": float(np.sqrt(np.mean((pred_quad - y_test1) ** 2))), "quality_features": False},
        {"model": "quadratic_ilr_ridge_quality", "cv_mse": quadratic_q_cv_mse,
         "test_1m_macro_rmse": quality_test_rmse,
         "test_1m_pooled_rmse": float(np.sqrt(np.mean((pred_quad_q - y_test1) ** 2))), "quality_features": True},
    ])
    quality_ablation["selected"] = quality_ablation["model"].eq(selected_name)
    quality_ablation.to_csv(MIXTURE_OUT / "quality_integration_ablation.csv", index=False, encoding="utf-8-sig")

    scale_sizes = {"train_1m": 1e6, "test_1m": 1e6, "test_60m": 60e6,
                   "test_1B": 1e9, "est_10b": 1e10, "est_70b": 7e10}

    def fit_scale_model(keys):
        x = np.vstack([scale_feature_matrix(datasets[k][1], scale_sizes[k], basis) for k in keys])
        y = np.vstack([datasets[k][2] for k in keys])
        return fit_ridge_cv(x, y, alphas)[0]

    scale_full_keys = ["train_1m", "test_60m", "test_1B"]
    scale_model = fit_scale_model(scale_full_keys)
    scale_rows = []
    p, y = datasets["test_1m"][1:3]
    pred = scale_model.predict(scale_feature_matrix(p, scale_sizes["test_1m"], basis))
    scale_rows.append(macro_summary("test_1m", y, pred, "external_composition_test"))
    for held, train_keys in [("test_60m", ["train_1m", "test_1B"]),
                             ("test_1B", ["train_1m", "test_60m"])]:
        held_model = fit_scale_model(train_keys)
        p, y = datasets[held][1:3]
        pred = held_model.predict(scale_feature_matrix(p, scale_sizes[held], basis))
        scale_rows.append(macro_summary(held, y, pred, "leave_one_scale_out"))
    scale_prediction_frames = []
    for split in ["est_10b", "est_70b"]:
        indices, p, y, *_ = datasets[split]
        pred = scale_model.predict(scale_feature_matrix(p, scale_sizes[split], basis))
        scale_rows.append(macro_summary(split, y, pred, "estimated_pressure_test"))
        scale_prediction_frames.append(pd.DataFrame({
            "split": split, "index": indices, "actual_macro_loss": y.mean(axis=1),
            "predicted_macro_loss": pred.mean(axis=1),
        }))
    scale_metrics = pd.DataFrame(scale_rows)
    scale_metrics.to_csv(MIXTURE_OUT / "scale_aware_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(scale_prediction_frames, ignore_index=True).to_csv(
        MIXTURE_OUT / "scale_aware_predictions.csv", index=False, encoding="utf-8-sig")

    # Keep the response-surface optimum as a diagnostic, but make the robust top-decile
    # empirical centroid the primary recommendation.
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0, ddof=1)
    y_std[y_std == 0] = 1
    z_mu = z_train.mean(axis=0)
    cov = np.cov(z_train, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    d2_train = np.einsum("ij,jk,ik->i", z_train - z_mu, cov_inv, z_train - z_mu)
    d2_limit = float(np.quantile(d2_train, 0.90))
    p_lower = np.quantile(p_train, 0.10, axis=0)
    p_upper = np.quantile(p_train, 0.90, axis=0)

    def objective(z):
        return float(np.mean((predict_p(ilr_inverse(z, basis)[None, :])[0] - y_mean) / y_std))

    def support(z):
        dz = np.asarray(z) - z_mu
        return d2_limit - float(dz @ cov_inv @ dz)

    def lower_component_support(z):
        return ilr_inverse(z, basis) - p_lower

    def upper_component_support(z):
        return p_upper - ilr_inverse(z, basis)

    train_objectives = np.mean((predict_p(p_train) - y_mean) / y_std, axis=1)
    starts = [z_mu, z_train[np.argmin(train_objectives)]]
    starts += [z_train[i] for i in RNG.choice(len(z_train), size=18, replace=False)]
    candidates = []
    constraints = [
        {"type": "ineq", "fun": support},
        {"type": "ineq", "fun": lower_component_support},
        {"type": "ineq", "fun": upper_component_support},
    ]
    for start in starts:
        result = minimize(objective, start, method="SLSQP", constraints=constraints,
                          options={"maxiter": 2000, "ftol": 1e-10})
        if (result.success and support(result.x) >= -1e-6
                and np.min(lower_component_support(result.x)) >= -1e-6
                and np.min(upper_component_support(result.x)) >= -1e-6):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("No feasible mixture optimization result")
    best = min(candidates, key=lambda x: x.fun)
    p_model_opt = ilr_inverse(best.x, basis)
    observed_train_objectives = np.mean((y_train - y_mean) / y_std, axis=1)
    best_observed_idx = int(np.argmin(observed_train_objectives))
    top_centroids = {}
    for fraction in (0.05, 0.10, 0.20):
        count = int(math.ceil(len(p_train) * fraction))
        top_centroids[fraction] = p_train[np.argsort(observed_train_objectives)[:count]].mean(axis=0)
    p_rec = top_centroids[0.10]

    boot_recommendations = []
    for _ in range(300):
        sample = RNG.choice(len(p_train), size=len(p_train), replace=True)
        count = int(math.ceil(len(sample) * 0.10))
        chosen = sample[np.argsort(observed_train_objectives[sample])[:count]]
        boot_recommendations.append(p_train[chosen].mean(axis=0))
    boot_recommendations = np.asarray(boot_recommendations)
    rec = pd.DataFrame({
        "domain": mixture_domains,
        "recommended_share": p_rec,
        "model_optimal_share": p_model_opt,
        "best_observed_share": p_train[best_observed_idx],
        "training_mean_share": p_train.mean(axis=0),
    })
    rec.to_csv(MIXTURE_OUT / "recommended_mixture.csv", index=False, encoding="utf-8-sig")
    stability = pd.DataFrame({
        "domain": mixture_domains, "top_5pct_centroid": top_centroids[0.05],
        "top_10pct_centroid": top_centroids[0.10], "top_20pct_centroid": top_centroids[0.20],
        "bootstrap_ci_low": np.quantile(boot_recommendations, 0.025, axis=0),
        "bootstrap_ci_high": np.quantile(boot_recommendations, 0.975, axis=0),
    })
    stability.to_csv(MIXTURE_OUT / "recommendation_stability.csv", index=False, encoding="utf-8-sig")

    # Feasible perturbation effects: increase one domain by one percentage point, shrink others proportionally.
    base = p_train.mean(axis=0)
    base_pred = predict_p(base[None, :])[0]
    effects = []
    step = 0.01
    for j, col in enumerate(p_cols):
        pert = base.copy()
        other_total = 1 - pert[j]
        if other_total <= step:
            continue
        pert[np.arange(len(pert)) != j] *= (1 - pert[j] - step) / other_total
        pert[j] += step
        pred = predict_p(pert[None, :])[0]
        effects.append({"domain": col.replace("train_the_pile_", ""), "delta_share": step,
                        "delta_macro_loss": float(np.mean(pred - base_pred)),
                        "max_abs_domain_loss_change": float(np.max(np.abs(pred - base_pred)))})
    effects = pd.DataFrame(effects).sort_values("delta_macro_loss")
    effects.to_csv(MIXTURE_OUT / "mixture_marginal_effects.csv", index=False, encoding="utf-8-sig")

    model_summary = {
        "selected_model": selected_name, "linear_alpha": linear_alpha, "quadratic_alpha": quadratic_alpha,
        "quadratic_quality_alpha": quadratic_q_alpha,
        "selection_rule": "train_1m_5fold_cv_mse_2pct",
        "linear_cv_mse": linear_cv_mse, "quadratic_cv_mse": quadratic_cv_mse,
        "quadratic_quality_cv_mse": quadratic_q_cv_mse,
        "quality_mapping_available_domains": int(np.isfinite(quality_vector).sum()),
        "quality_mapping_total_domains": len(quality_vector), "quality_features_adopted": quality_adopted,
        "test_1m_pooled_rmse_linear": float(np.sqrt(np.mean((pred_linear - y_test1) ** 2))),
        "test_1m_pooled_rmse_quadratic": float(np.sqrt(np.mean((pred_quad - y_test1) ** 2))),
        "test_1m_macro_rmse_linear": float(np.sqrt(np.mean((pred_linear.mean(axis=1) - y_test1.mean(axis=1)) ** 2))),
        "test_1m_macro_rmse_quadratic": float(np.sqrt(np.mean((pred_quad.mean(axis=1) - y_test1.mean(axis=1)) ** 2))),
        "support_d2_limit_p90": d2_limit,
        "recommended_support_slack": float(support(ilr_transform(p_rec[None, :], basis)[0])),
        "recommended_objective": float(np.mean((predict_p(p_rec[None, :])[0] - y_mean) / y_std)),
        "recommendation_rule": "centroid_of_best_observed_10pct",
        "model_optimal_objective": float(best.fun),
        "model_optimal_max": float(p_model_opt.max()),
        "model_optimal_active_component_bounds": int(np.isclose(p_model_opt, p_lower, atol=1e-6).sum()
                                                     + np.isclose(p_model_opt, p_upper, atol=1e-6).sum()),
        "best_observed_actual_objective": float(observed_train_objectives[best_observed_idx]),
        "best_observed_predicted_objective": float(train_objectives[best_observed_idx]),
        "recommended_sum": float(p_rec.sum()), "recommended_min": float(p_rec.min()),
        "recommended_max": float(p_rec.max()),
        "min_component_lower_slack": float(np.min(p_rec - p_lower)),
        "min_component_upper_slack": float(np.min(p_upper - p_rec)),
        "component_bounds": "A4 empirical 10th--90th percentiles for diagnostic model optimum",
        "optimization_successful_starts": len(candidates), "optimization_total_starts": len(starts),
        "external_tables_are_estimated_not_independent_observations": True,
    }
    (MIXTURE_OUT / "mixture_model_summary.json").write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_composition(metrics, predictions, rec, effects, scale_metrics)
    return {"model_summary": model_summary, "metrics": metrics, "recommendation": rec, "effects": effects,
            "quality_ablation": quality_ablation, "scale_metrics": scale_metrics,
            "mapping": mapping, "recommendation_stability": stability}


def plot_composition(metrics, predictions, rec, effects, scale_metrics):
    configure_plots()
    splits = ["train_1m", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, split in zip(axes.ravel(), splits):
        sub = predictions[predictions["split"] == split]
        ax.scatter(sub["actual_macro_loss"], sub["predicted_macro_loss"], s=13, alpha=.65, color="#2563eb")
        lo = min(sub["actual_macro_loss"].min(), sub["predicted_macro_loss"].min())
        hi = max(sub["actual_macro_loss"].max(), sub["predicted_macro_loss"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#6b7280", linewidth=1)
        ax.set_title(split)
        ax.set_xlabel("实际平均 Loss")
        ax.set_ylabel("预测平均 Loss")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(MIXTURE_FIG / "mixture_prediction_by_scale.pdf", bbox_inches="tight")
    plt.close(fig)

    macro = metrics[metrics["target"] == "macro_average"].set_index("split").loc[splits]
    pooled = metrics[metrics["target"] == "pooled_domain"].set_index("split").loc[splits]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(splits))
    ax.bar(x - .18, macro["rmse"], width=.35, label="平均 Loss RMSE", color="#2563eb")
    ax.bar(x + .18, pooled["rmse"], width=.35, label="13 域合并 RMSE", color="#6b7280")
    labels = [s if not s.startswith("est_") else f"{s}\n估计" for s in splits]
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(MIXTURE_FIG / "mixture_rmse_by_scale.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_rec = rec.sort_values("recommended_share")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(plot_rec))
    ax.barh(y - .24, plot_rec["recommended_share"], height=.23, label="稳健实证建议", color="#2563eb")
    ax.barh(y, plot_rec["model_optimal_share"], height=.23, label="响应面边界解", color="#f59e0b")
    ax.barh(y + .24, plot_rec["training_mean_share"], height=.23, label="训练配比均值", color="#9ca3af")
    ax.set_yticks(y, plot_rec["domain"])
    ax.set_xlabel("配比")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=3)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(MIXTURE_FIG / "recommended_mixture.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_eff = effects.sort_values("delta_macro_loss")
    colors = np.where(plot_eff["delta_macro_loss"] <= 0, "#16a34a", "#dc2626")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(plot_eff)), plot_eff["delta_macro_loss"], color=colors)
    ax.set_yticks(np.arange(len(plot_eff)), plot_eff["domain"])
    ax.axvline(0, color="#111827", linewidth=.8)
    ax.set_xlabel("配比增加 1 个百分点时的预测平均 Loss 变化")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(MIXTURE_FIG / "mixture_marginal_effects.pdf", bbox_inches="tight")
    plt.close(fig)

    comparison_splits = ["test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
    base_rmse = (metrics[metrics["target"] == "macro_average"].set_index("split")
                 .loc[comparison_splits, "rmse"])
    scale_rmse = scale_metrics.set_index("evaluation").loc[comparison_splits, "rmse"]
    x = np.arange(len(comparison_splits))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - .18, base_rmse, width=.35, label="1M 主响应面", color="#6b7280")
    ax.bar(x + .18, scale_rmse, width=.35, label="多尺度补充模型（外部/留一尺度）", color="#2563eb")
    labels = [s if not s.startswith("est_") else f"{s}\n估计压力测试" for s in comparison_splits]
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("平均 Loss RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(MIXTURE_FIG / "scale_aware_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def build_report(quality, composition):
    domain = quality["domain_summary"]
    a1 = domain[domain["dataset"] == "A1_sample"]
    top = a1.sort_values("quality_median", ascending=False).head(3)
    bottom = a1.sort_values("quality_median").head(3)
    macro = composition["metrics"][composition["metrics"]["target"] == "macro_average"].set_index("split")
    pooled = composition["metrics"][composition["metrics"]["target"] == "pooled_domain"].set_index("split")
    summary = composition["model_summary"]
    rec = composition["recommendation"].sort_values("recommended_share", ascending=False).head(5)
    effects = composition["effects"]
    extension = quality["extension_comparison"]
    causes = quality["conflict_causes"]
    c4_causes = causes[causes["domain"] == "c4"].head(3)
    ablation = composition["quality_ablation"]
    scale_metrics = composition["scale_metrics"]
    mapping = composition["mapping"]
    report = rf"""# F 题问题一计算结果

## 运行环境与安全边界

- Python {sys.version.split()[0]}，随机种子 `{SEED}`。
- 数据仅从 `F题_清洗后/A_data_value` 读取。
- A1 文本字段只按数据处理，不执行其中任何指令；PDF 隐藏文字未进入程序、特征或参数。
- 8 个列表字段按原始数据集可核验定义处理：二分类 logits 转正类概率（`ad_en` 的正类为“无广告”），PRRC 六级 logits 转期望等级，QuRating 四维分量单调压缩后汇总。

## 数据读取与预处理

- A1–A3 共读取 {quality['audit']['records']:,} 条记录，无效 JSON 行 {quality['audit']['invalid_json_lines']} 条。
- 文件内重复 ID {quality['audit']['within_dataset_duplicate_ids']} 条；跨文件重复 ID 数 {quality['audit']['cross_dataset_repeated_ids']}。
- 质量标尺只由 A1 的 1%–99% 分位确定，A2/A3 沿用同一标尺。
- 三个长度/规模型指标按 A1 的稳健中心构造适宜度，其余指标按收益型或成本型统一为“越高越好”。完整规则见 `quality_metric_rules_weights.csv`。

## 质量评价与冲突

- 主评分为四个语义组的 Huber 稳健位置，平均值 {quality['quality_score_mean']:.4f}，标准差 {quality['quality_score_std']:.4f}。
- A1 冲突阈值取冲突度的 90% 分位：{quality['audit']['conflict_threshold_a1_p90']:.4f}；同一阈值直接用于 A2/A3。
- Huber 参数与等权对照的稳定性见 `quality_sensitivity.csv`。

A1 领域质量中位数最高的三个领域：

{markdown_table(top[['domain','n','quality_median','median_ci_low','median_ci_high','conflict_rate']])}

A1 领域质量中位数最低的三个领域：

{markdown_table(bottom[['domain','n','quality_median','median_ci_low','median_ci_high','conflict_rate']])}

这些分数是本数据集、当前指标体系下的相对评价，不等同于实际训练收益。A2/A3 与 A1 共同域的差异应结合抽样设计解释，不能仅凭均值断言总体质量变化。

原始文本抽样核验由独立脚本 `code/problem1/raw_text_audit.py` 完成：A1 七领域按质量分位与冲突度选取 49 条成对原文，检查 ID、领域、文本长度并保存可见文本特征。六条案例的前 360 字可见段落显示：高分并不保证没有版权前言，低分也不等于数学论文或代码无用；Wikipedia 高、低分位均出现非英语条目。抽样不是独立专家盲评，不估计总体准确率，不据此修改质量分或配比。详情见 `reports/问题一/验证验收/RAW_TEXT_AUDIT_PROBLEM1.md`。

A2/A3 扩展样本复核：

{markdown_table(extension[['domain','a1_n','extension_n','a1_quality_median','extension_quality_median','median_difference_extension_minus_a1','a1_conflict_rate','extension_conflict_rate']])}

冲突消解不仅使用 Huber 稳健位置，还按领域拆解主导冲突方向和指标驱动。c4 的高冲突率为 {a1.set_index('domain').loc['c4','conflict_rate']:.4f}，其主要成因如下：

{markdown_table(c4_causes[['pair','count','share_among_domain_high_conflict','dominant_orientation','orientation_share','high_driver_metric','low_driver_metric']])}

完整领域画像见 `conflict_cause_profile.csv`；这里的“成因”表示指标结构上的直接证据，不作文本语义因果推断。

## 质量评分到配比模型的接口

A16 共提供 {len(mapping)} 个配方域映射，其中 {int(mapping['mapping_available'].sum())} 个 direct/near_direct 域可从 A1 获得 Q，剩余 {int((~mapping['mapping_available']).sum())} 个 inferred 域没有可观测 Q。程序对可映射域构造“已映射配比覆盖率”和“覆盖部分的加权平均 Q”，并与不含 Q 的模型做同折交叉验证和 A6–A7 独立检验；没有为 inferred 域主观填值。

{markdown_table(ablation[['model','quality_features','cv_mse','test_1m_macro_rmse','test_1m_pooled_rmse','selected']])}

质量增强模型虽然内部交叉验证 MSE 更低，但只有在内部 MSE 和独立 1M 平均 Loss RMSE均至少改善 2% 时才采用。本次 `quality_features_adopted={summary['quality_features_adopted']}`，因此 Q 已进入可检验的消融链路，但不强行进入最终预测器。

## 配比模型

- 候选模型：标准化后的 ILR 线性 Ridge、ILR 二阶 Ridge 和质量增强二阶 Ridge；仅按 A4–A5 内部五折交叉验证及上述 Q 外部增益规则选择 `{summary['selected_model']}`，A6–A7 保留为独立的 1M 检验集。
- 训练内部交叉验证 MSE：线性 {summary['linear_cv_mse']:.6f}，二阶 {summary['quadratic_cv_mse']:.6f}；二阶改善超过预设 2% 门槛。
- A6–A7 平均 Loss RMSE：线性 {summary['test_1m_macro_rmse_linear']:.6f}，二阶 {summary['test_1m_macro_rmse_quadratic']:.6f}；13 域合并 RMSE 分别为 {summary['test_1m_pooled_rmse_linear']:.6f}、{summary['test_1m_pooled_rmse_quadratic']:.6f}。
- 主推荐采用实测标准化 Loss 最优前 10% 配方的平均值，总和为 {summary['recommended_sum']:.12f}，最大单域占比 {summary['recommended_max']:.4f}；5%/10%/20% 阈值和 Bootstrap 区间见 `recommendation_stability.csv`。
- 响应面诊断解另在 10%--90% 分量范围和 ILR 距离 90% 支持域内做 {summary['optimization_successful_starts']}/{summary['optimization_total_starts']} 次多起点优化。它仍有 {summary['model_optimal_active_component_bounds']} 个活跃分量边界，故不作为主推荐。

各尺度的平均 Loss 误差（每个配方先对 13 域求均值，再计算指标）：

{markdown_table(macro[['n','mae','rmse','bias','centered_rmse','spearman']], include_index=True)}

上表的 `centered_rmse` 对每个尺度的平均 Loss 去均值后计算，仅衡量平均配比效应能否迁移；它不等同于经校准的绝对预测成绩。逐域误差汇总另见 `mixture_model_metrics.csv` 中的 `pooled_domain` 行：1M 检验集的 13 域合并 RMSE 为 {pooled.loc['test_1m','rmse']:.4f}，与平均 Loss RMSE 不是同一指标。10B、70B 表中的 Loss 是由较小尺度外推的估计量，不能当作独立真实测试成绩。

核心判断：1M 检验集的平均 Loss RMSE 为 {macro.loc['test_1m','rmse']:.4f}、排序相关为 {macro.loc['test_1m','spearman']:.4f}；跨到 60M 和 1B 后，去均值的平均 Loss RMSE 分别为 {macro.loc['test_60m','centered_rmse']:.4f} 和 {macro.loc['test_1B','centered_rmse']:.4f}，说明配比效应只能部分迁移。10B、70B 的平均 Loss 排序相关为 {macro.loc['est_10b','spearman']:.4f}、{macro.loc['est_70b','spearman']:.4f}，方向已经反转，因此**不支持把 1M 响应面直接用于大尺度绝对预测或定案配比**。

稳健实证建议占比最高的五个领域：

{markdown_table(rec[['domain','recommended_share','model_optimal_share','best_observed_share','training_mean_share']])}

主推荐是 52 个实测优良配方的质心，降低了单个最优样本的偶然性，也避免二阶响应面把大量权重推到经验边界。`model_optimal_share` 只展示模型在收紧支持域后仍存在的边界倾向；两种配比都只能作为下一轮实验候选。

## 多尺度补充模型

补充模型使用 ILR 配比、标准化对数规模、规模二次项和“ILR×规模”交互。它用 train_1m、test_60m、test_1B 拟合后在独立 test_1m 上检查配比泛化，并对 60M、1B 另做留一尺度检验。A12–A15 仍仅作估计压力测试。

{markdown_table(scale_metrics[['evaluation','role','n','rmse','bias','centered_rmse','spearman']])}

该模型显著修正绝对 Loss 的尺度截距，但留一尺度误差和 10B/70B 排序仍不稳定，因此作为跨尺度校准补充，不替换 1M 主响应面，也不用于生成最终配比。

## 单域效应与组合解释

`mixture_marginal_effects.csv` 记录从训练均值配比出发，将某域增加 1 个百分点、其余域按比例缩减后的预测 Loss 变化。预测 Loss 降幅最大的三个方向为：

{markdown_table(effects.head(3)[['domain','delta_share','delta_macro_loss','max_abs_domain_loss_change']])}

该效应依赖基准配比，不能解释成全局因果系数。二阶模型仅在 A4–A5 内部五折交叉验证 MSE 至少改善 2% 时采用，防止用检验集选择模型。

## 约束与一致性校验

- 配比表与 Loss 表按 `index` 一对一合并；每个文件的行数、配比和范围见 `mixture_data_audit.csv`。
- 所有输入配比非负，归一化后严格满足和为 1。
- 推荐配比通过单纯形和 ILR 支持域回代检查。
- 质量指标方向、缩放边界和权重均可在结果表追溯。

## 输出文件

- `outputs/problem1/quality/results/quality_sample_scores.csv`：样本质量、冲突度和主要冲突对。
- `outputs/problem1/quality/results/quality_domain_summary.csv`：领域质量、区间及冲突率。
- `outputs/problem1/quality/results/quality_metric_rules_weights.csv`：方向、缩放规则和 CRITIC 组内权重。
- `outputs/problem1/quality/results/conflict_pair_summary.csv`：主要冲突对。
- `outputs/problem1/quality/results/conflict_cause_profile.csv`：分领域冲突方向和指标驱动。
- `outputs/problem1/quality/results/raw_text_stratified_audit.csv`、`raw_text_case_review.csv`：原始文本分层核验与具体案例。
- `outputs/problem1/mixture/results/mixture_model_metrics.csv`、`mixture_predictions.csv`：训练、检验和外推结果。
- `outputs/problem1/mixture/results/quality_domain_mapping.csv`、`quality_integration_ablation.csv`：A16 映射与 Q 增益检验。
- `outputs/problem1/mixture/results/scale_aware_metrics.csv`：多尺度模型的独立及留一尺度检查。
- `outputs/problem1/mixture/results/recommended_mixture.csv`、`recommendation_stability.csv`：稳健候选配比和区间。
- `outputs/problem1/quality/figures/*.pdf` 与 `outputs/problem1/mixture/figures/*.pdf`：论文可用矢量图。

## 可复现运行方式

```powershell
python F题_初步方案/code/problem1/problem1.py
```
"""
    (PROJECT / "reports" / "问题一" / "结果分析" / "RESULTS_REPORT_PROBLEM1.md").write_text(report, encoding="utf-8")


def main():
    quality = quality_analysis()
    composition = composition_analysis(quality)
    build_report(quality, composition)
    run_summary = {
        "quality_records": quality["audit"]["records"],
        "selected_mixture_model": composition["model_summary"]["selected_model"],
        "quality_features_adopted": composition["model_summary"]["quality_features_adopted"],
        "test_1m_rmse": float(composition["metrics"].query("split == 'test_1m' and target == 'macro_average'")["rmse"].iloc[0]),
        "test_1m_pooled_domain_rmse": float(composition["metrics"].query("split == 'test_1m' and target == 'pooled_domain'")["rmse"].iloc[0]),
        "recommended_max_share": composition["model_summary"]["recommended_max"],
        "recommendation_rule": composition["model_summary"]["recommendation_rule"],
        "outputs": OUTPUT_ROOT.relative_to(PROJECT).as_posix(),
        "quality_results": QUALITY_OUT.relative_to(PROJECT).as_posix(),
        "mixture_results": MIXTURE_OUT.relative_to(PROJECT).as_posix(),
        "quality_figures": QUALITY_FIG.relative_to(PROJECT).as_posix(),
        "mixture_figures": MIXTURE_FIG.relative_to(PROJECT).as_posix(),
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
