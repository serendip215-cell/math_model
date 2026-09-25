"""Question 4: auditable leaderboard panel, conditional decomposition and scenarios.

Observational date residuals are not causal technology effects. C6 bridge is
local to Pythia and never used to translate the frontier outside its support.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c4_source_review import adjudicate

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后" / "C_efficiency_evolution"
ROOT = PROJECT / "outputs" / "problem4"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = PROJECT / "reports" / "问题四" / "结果分析" / "RESULTS_REPORT_PROBLEM4.md"
for folder in [RESULTS, FIGURES, REPORT.parent]:
    folder.mkdir(parents=True, exist_ok=True)

TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}
CUTOFF = pd.Timestamp("2025-03-13")
START = pd.Timestamp("2024-06-08")
EARLY_END = pd.Timestamp("2024-08-31")
LATE_START = pd.Timestamp("2025-01-01")


def save(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    frame.to_csv(RESULTS / filename, index=False, encoding="utf-8-sig")
    return frame


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES / filename, metadata={"CreationDate": None})
    plt.close(fig)


def tbl(frame: pd.DataFrame, digits: int = 5) -> str:
    view = frame.copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}g}")
    return "\n".join(["| " + " | ".join(view.columns) + " |",
                      "| " + " | ".join(["---"] * len(view.columns)) + " |",
                      *("| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy())])


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).split("/")[-1].lower())


def family(value: str) -> str:
    name = str(value).lower()
    for label in ["qwen", "llama", "gemma", "mistral", "phi", "yi-", "pythia",
                  "deepseek", "falcon", "smollm", "aya", "bloom", "gpt-neo", "olmo"]:
        if label in name:
            return label.rstrip("-")
    return "namespace:" + name.split("/")[0]


def model_type(raw: str) -> str:
    s = str(raw)
    if s == "🟢 pretrained": return "pretrained"
    if s == "🟩 continuously pretrained": return "continually_pretrained"
    if "chat models" in s or "fine-tuned" in s: return "posttrained"
    if "merges" in s: return "merge"
    return "other"


def load_contract() -> dict[str, pd.DataFrame]:
    names = {"C1": "leaderboard_cleaned.csv", "C2": "leaderboard_enhanced.csv",
             "C3": "leaderboard_extended_timeseries.csv", "C4": "epoch_all_ai_models.csv",
             "C5": "loss_benchmark_bridge.csv", "C6": "loss_benchmark_bridge_expanded.csv"}
    expected = {"C1": 4576, "C2": 4576, "C3": 4599, "C4": 3523, "C5": 43, "C6": 75}
    frames = {key: pd.read_csv(DATA / file, low_memory=False) for key, file in names.items()}
    for key, frame in frames.items():
        if len(frame) != expected[key]: raise ValueError(f"{key} row contract changed: {len(frame)}")
    if not frames["C1"]["Model"].equals(frames["C2"]["Model"]):
        raise ValueError("C1 and C2 row order/model alignment changed")
    if not np.allclose(frames["C1"][TASKS].mean(axis=1),
                       frames["C1"]["Average ⬆️"], atol=1e-10):
        raise ValueError("C1 Average is not the six-task macro average")
    save(pd.DataFrame([{"id": key, "file": names[key], "rows": len(frame),
                        "columns": len(frame.columns)} for key, frame in frames.items()]),
         "data_contract.csv")
    return frames


def prepare_panel(c2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = c2.copy()
    flow = [{"step": "C2 raw", "rows": len(a), "unique_models": a.Model.nunique()}]
    a["submission"] = pd.to_datetime(a["Submission Date"], errors="coerce")
    a["N_B"] = pd.to_numeric(a["#Params (B)"], errors="coerce")
    a["Y"] = a[TASKS].mean(axis=1)
    a = a[a.submission.between(START, CUTOFF) & a.N_B.gt(0) &
          a[TASKS].notna().all(axis=1) & a[TASKS].ge(0).all(axis=1) &
          a[TASKS].le(100).all(axis=1)].copy()
    flow.append({"step": "valid date/positive N/six tasks", "rows": len(a),
                 "unique_models": a.Model.nunique()})
    a = a.sort_values(["Model", "submission"]).drop_duplicates("Model", keep="first")
    flow.append({"step": "earliest valid submission per model", "rows": len(a),
                 "unique_models": a.Model.nunique()})
    a["type_group_raw"] = a.Type.map(model_type)
    a["type_group"] = a.type_group_raw.copy()
    # Primary author documents identify these released checkpoints as post-trained.
    # See c4_source_review.py for the paper/model-card evidence.
    a.loc[a.Model.isin(["microsoft/phi-1", "microsoft/phi-4"]),
          "type_group"] = "posttrained"
    a["family"] = a.Model.map(family)
    a["namespace"] = a.Model.astype(str).str.split("/").str[0].str.lower()
    a["license_permissive"] = a["Hub License"].isin(PERMISSIVE)
    a["open_weight_confirmed"] = a.Epoch_AI_Open_Weights.eq("Yes")
    a["strict_open"] = a.license_permissive & a.open_weight_confirmed
    a["license_only_unverified"] = a.license_permissive & a.Epoch_AI_Open_Weights.isna()
    a["license_explicit_no"] = a.license_permissive & a.Epoch_AI_Open_Weights.eq("No")
    a["epoch_publication"] = pd.to_datetime(a.Epoch_AI_Publication_Date, errors="coerce")
    a["publication_lag_days"] = (a.submission - a.epoch_publication).dt.days
    a["t_month"] = (a.submission - START).dt.days / 30.4375
    a["logN"] = np.log10(a.N_B)
    for step, filt in [("permissive license", a.license_permissive),
                       ("strict open weight + license", a.strict_open),
                       ("permissive license + unknown weight status", a.license_only_unverified),
                       ("permissive license + explicit No weight status", a.license_explicit_no),
                       ("strict pretrained", a.strict_open & a.type_group.eq("pretrained")),
                       ("strict posttrained", a.strict_open & a.type_group.eq("posttrained"))]:
        sub = a[filt]
        flow.append({"step": step, "rows": len(sub), "unique_models": sub.Model.nunique(),
                     "families": sub.family.nunique()})
    save(pd.DataFrame(flow), "panel_attrition.csv")
    dated = a[a.strict_open & a.type_group.isin(["pretrained", "posttrained"])].copy()
    dated["period"] = np.select([dated.submission.le(EARLY_END), dated.submission.ge(LATE_START)],
                                 ["early", "late"], default="middle")
    pub_audit = save(dated.groupby(["type_group", "period"], as_index=False).agg(
        models=("Model", "size"), publication_date_available=("epoch_publication", "count"),
        publication_after_submission=("publication_lag_days", lambda s: int(s.lt(0).sum())),
        publication_over_180_days_older=("publication_lag_days", lambda s: int(s.gt(180).sum())),
        median_submission_minus_publication_days=("publication_lag_days", "median")),
        "publication_submission_audit.csv")
    a["submission"] = a.submission.dt.strftime("%Y-%m-%d")
    a["epoch_publication"] = a.epoch_publication.dt.strftime("%Y-%m-%d")
    save(a[["Model", "submission", "N_B", "Y", *TASKS, "Hub License",
            "Epoch_AI_Open_Weights", "type_group_raw", "type_group", "family", "namespace",
            "license_permissive", "strict_open", "license_only_unverified",
            "license_explicit_no", "epoch_publication", "publication_lag_days",
            "t_month", "logN"]], "audited_leaderboard_panel.csv")
    return a, pd.DataFrame(flow), pub_audit


def link_c4(panel: pd.DataFrame, c4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = panel.copy(); left["match_key"] = left.Model.map(slug)
    left_key_counts = left.match_key.value_counts()
    right = c4.copy(); right["match_key"] = right.Model.map(slug)
    counts = right.match_key.value_counts()
    right = right[["match_key", "Model", "Parameters", "Training compute (FLOP)",
                   "Training dataset size (total)", "Publication date", "Confidence",
                   "Domain", "Open model weights?", "Hugging Face developer id",
                   "Organization", "Reference", "Link", "Parameters notes",
                   "Training compute notes", "Training compute estimation method",
                   "Base model", "Foundation model"]]
    merged = left.merge(right, on="match_key", how="inner", suffixes=("_C1", "_C4"))
    merged["C4_key_unique"] = merged.match_key.map(counts).eq(1)
    merged["C1_key_unique"] = merged.match_key.map(left_key_counts).eq(1)
    developer = merged["Hugging Face developer id"].fillna("").astype(str).str.strip().str.lower()
    namespace = merged.namespace.fillna("").astype(str).str.strip().str.lower()
    merged["developer_id_available"] = developer.ne("")
    merged["developer_id_matches"] = developer.eq(namespace) & merged.developer_id_available
    # When C4 names an official repository owner, reject community reuploads.
    # Without an owner field, a shared normalized model name is ambiguous.
    merged["source_identity_supported"] = np.where(
        merged.developer_id_available, merged.developer_id_matches, merged.C1_key_unique)
    merged["N_ratio"] = merged.N_B * 1e9 / pd.to_numeric(merged.Parameters, errors="coerce")
    merged["N_consistent"] = merged.N_ratio.between(.7, 1.3)
    published = pd.to_datetime(merged["Publication date"], errors="coerce")
    submitted = pd.to_datetime(merged.submission, errors="coerce")
    merged["date_consistent"] = published.notna() & ((published - submitted).dt.days <= 7)
    merged["language_domain"] = merged.Domain.fillna("").str.contains("Language", case=False)
    merged["source_confident"] = merged.Confidence.isin(["Confident", "Likely"])
    merged["weight_agreement"] = ~merged.strict_open | merged["Open model weights?"].eq("Yes")
    merged["accepted_link"] = (merged.C4_key_unique & merged.source_identity_supported &
                               merged.N_consistent &
                               merged.date_consistent & merged.language_domain &
                               merged.source_confident & merged.weight_agreement)
    merged["review_candidate"] = (merged.accepted_link & merged.strict_open &
                                 merged.type_group_raw.eq("pretrained") &
                                 pd.to_numeric(merged["Training compute (FLOP)"], errors="coerce").gt(0))
    merged["usable_compute"] = (merged.accepted_link & merged.strict_open &
                                 merged.type_group.eq("pretrained") &
                                 pd.to_numeric(merged["Training compute (FLOP)"], errors="coerce").gt(0))
    audit_cols = ["Model_C1", "Model_C4", "match_key", "N_B", "Parameters", "N_ratio",
                  "submission", "Publication date", "Confidence", "Domain", "Open model weights?",
                  "strict_open", "type_group_raw", "type_group", "Hugging Face developer id",
                  "C4_key_unique", "C1_key_unique", "developer_id_available",
                  "developer_id_matches", "source_identity_supported", "N_consistent",
                  "date_consistent", "language_domain", "source_confident",
                  "weight_agreement", "accepted_link", "review_candidate", "usable_compute",
                  "Training compute (FLOP)", "Training dataset size (total)"]
    save(merged[audit_cols], "c1_c4_link_audit.csv")
    use = merged[merged.usable_compute].copy()
    save(use[audit_cols], "c4_compute_linked_subset.csv")
    review_cols = ["Model_C1", "Model_C4", "Organization", "Hugging Face developer id",
                   "N_B", "Parameters", "Parameters notes", "N_ratio", "submission",
                   "Publication date", "Training compute (FLOP)", "Training compute notes",
                   "Training compute estimation method", "Base model", "Foundation model",
                   "Confidence", "Open model weights?", "Reference", "Link"]
    review = merged.loc[merged.review_candidate, review_cols].copy()
    review["Link"] = review["Link"].astype("string").str.replace(
        r"[ \t]+(?=\r?\n|$)", "", regex=True)
    review["automated_rule_status"] = "passed"
    review["manual_review_status"] = "pending_checkpoint_and_training_telemetry_check"
    for col in ["repository_revision_evidence", "version_identity_review",
                "training_stage_review", "compute_source_review",
                "historical_open_status_review", "reviewer_notes"]:
        review[col] = pd.NA
    review["owner_id_missing"] = review["Hugging Face developer id"].isna()
    review = review.sort_values(["owner_id_missing", "Model_C1"], ascending=[False, True])
    save(review, "c4_manual_review_queue.csv")
    standalone = c4.copy()
    standalone["publication"] = pd.to_datetime(standalone["Publication date"], errors="coerce")
    standalone["compute"] = pd.to_numeric(standalone["Training compute (FLOP)"], errors="coerce")
    standalone["data_size"] = pd.to_numeric(standalone["Training dataset size (total)"], errors="coerce")
    standalone = standalone[standalone.Domain.fillna("").str.contains("Language", case=False) &
                            standalone["Open model weights?"].eq("Yes") &
                            standalone.Confidence.isin(["Confident", "Likely"]) &
                            standalone.publication.between(pd.Timestamp("2019-01-01"), CUTOFF)]
    standalone["year"] = standalone.publication.dt.year
    save(standalone.groupby("year", as_index=False).agg(models=("Model", "size"),
         compute_available=("compute", lambda s: int((s > 0).sum())),
         data_available=("data_size", lambda s: int((s > 0).sum())),
         median_compute_FLOPs=("compute", "median")), "c4_standalone_resource_trend.csv")
    return merged, use


def process_c8(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for file in sorted((DATA / "detailed_results").rglob("*.json")):
        try:
            obj = json.loads(file.read_text(encoding="utf-8"))
            results = obj.get("results", {})
            leaves = {key: val.get("acc_norm,none") for key, val in results.items()
                      if key.startswith("leaderboard_bbh_") and isinstance(val, dict) and
                      isinstance(val.get("acc_norm,none"), (float, int))}
            if not leaves: raise ValueError("no BBH leaf task with acc_norm")
            task_set = "|".join(sorted(leaves))
            group = results.get("leaderboard_bbh", {}).get("acc_norm,none", np.nan)
            rows.append({"file": str(file.relative_to(DATA)),
                         "Model": obj.get("model_name", file.parent.name),
                         "eval_timestamp": pd.to_numeric(obj.get("date"), errors="coerce"),
                         "model_revision": obj.get("config", {}).get("model_revision"),
                         "bbh_leaf_count": len(leaves),
                         "task_set_hash": hashlib.sha256(task_set.encode()).hexdigest()[:16],
                         "bbh_leaf_macro_raw_pct": 100 * float(np.mean(list(leaves.values()))),
                         "bbh_group_raw_pct": 100 * float(group) if isinstance(group, (float, int)) else np.nan,
                         "status": "parsed"})
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
            rows.append({"file": str(file.relative_to(DATA)), "Model": file.parent.name,
                         "status": f"invalid:{type(exc).__name__}"})
    files = save(pd.DataFrame(rows), "c8_file_audit.csv")
    valid = files[files.status.eq("parsed")].copy()
    valid["eval_timestamp"] = pd.to_numeric(valid.eval_timestamp, errors="coerce")
    latest = valid.sort_values(["Model", "eval_timestamp", "file"]).drop_duplicates("Model", keep="last")
    earliest = valid.sort_values(["Model", "eval_timestamp", "file"]).drop_duplicates("Model", keep="first")
    modal_set = latest.task_set_hash.value_counts().idxmax()
    latest["common_task_set"] = latest.task_set_hash.eq(modal_set)
    save(latest, "c8_bbh_leaf_aggregation.csv")
    joined = latest.merge(panel[["Model", "BBH", "submission"]], on="Model", how="inner")
    joined["raw_minus_C1_BBH"] = joined.bbh_leaf_macro_raw_pct - joined.BBH
    earliest_score = earliest.set_index("Model").bbh_leaf_macro_raw_pct
    joined["earliest_minus_latest_raw"] = joined.Model.map(earliest_score) - joined.bbh_leaf_macro_raw_pct
    save(joined, "c8_c1_bbh_comparison.csv")
    comparable = joined[joined.common_task_set]
    rho = spearmanr(comparable.bbh_leaf_macro_raw_pct, comparable.BBH).statistic if len(comparable) >= 3 else np.nan
    summary = {"json_files": len(files), "parsed": len(valid), "invalid": len(files) - len(valid),
               "models_latest": len(latest), "duplicate_model_files": len(valid) - len(latest),
               "modal_task_set_models": int(latest.common_task_set.sum()),
               "joined_to_C1": len(joined), "common_task_set_joined": len(comparable),
               "common_set_rank_spearman": float(rho) if np.isfinite(rho) else None,
               "median_raw_minus_C1_BBH": float(comparable.raw_minus_C1_BBH.median()) if len(comparable) else None,
               "same_scale_as_C1_established": False}
    (RESULTS / "c8_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return joined, summary


def bridge(c6: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    high = c6[c6.Loss_Comparability.str.startswith("High")].copy().reset_index(drop=True)
    if len(high) != 7: raise ValueError("C6 high-comparability contract changed")
    rows = []
    for i, test in high.iterrows():
        train = high.drop(i)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(train.Val_Loss, train.LB_Average)
        rows.append({"Model": test.Model, "Val_Loss": test.Val_Loss,
                     "LB_Average": test.LB_Average,
                     "loo_monotone_prediction": float(iso.predict([test.Val_Loss])[0]),
                     "loo_constant_prediction": float(train.LB_Average.mean())})
    loo = save(pd.DataFrame(rows), "bridge_high_loo.csv")
    medium = c6[c6.Loss_Comparability.str.startswith("Medium")].copy()
    medium["source_group_size"] = medium.groupby("Loss_Source").Model.transform("size")
    save(medium[["Model", "Loss_Source", "Loss_Comparability", "Val_Loss",
                 "LB_Average", "source_group_size"]], "bridge_medium_source_audit.csv")
    rho = spearmanr(high.Val_Loss, high.LB_Average)
    rmse_iso = float(np.sqrt(np.mean((loo.loo_monotone_prediction - loo.LB_Average) ** 2)))
    rmse_const = float(np.sqrt(np.mean((loo.loo_constant_prediction - loo.LB_Average) ** 2)))
    summary = {"high_n": len(high), "high_unique_family": 1,
               "high_loss_range": [float(high.Val_Loss.min()), float(high.Val_Loss.max())],
               "high_score_range": [float(high.LB_Average.min()), float(high.LB_Average.max())],
               "high_spearman": float(rho.statistic), "high_spearman_p": float(rho.pvalue),
               "loo_monotone_rmse": rmse_iso, "loo_constant_rmse": rmse_const,
               "local_bridge_predictive_gain": bool(rmse_iso < rmse_const),
               "medium_n": len(medium), "medium_sources_with_pairs": int(
                   medium.loc[medium.source_group_size >= 2, "Loss_Source"].nunique()),
               "frontier_translation_identified": False}
    (RESULTS / "bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return loo, summary


def score_fit(frame: pd.DataFrame, use_time: bool, alpha: float = 1.0) -> tuple:
    cols = ["logN", "t_month"] if use_time else ["logN"]
    x = frame[cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    y = logit((frame.Y.to_numpy(dtype=float) + .5) / 101.0)
    fit = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return scaler, fit, cols


def score_predict(fitted: tuple, frame: pd.DataFrame) -> np.ndarray:
    scaler, fit, cols = fitted
    return np.clip(101.0 * expit(fit.predict(scaler.transform(
        frame[cols].to_numpy(dtype=float)))) - .5, 0, 100)


def choose_alpha(train: pd.DataFrame, use_time: bool) -> tuple[float, pd.DataFrame]:
    groups = train.family.to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    rows = []
    if n_splits < 3:
        return 1.0, pd.DataFrame()
    for alpha in [.1, 1.0, 10.0, 100.0]:
        heldout_rows = []
        for tr, te in GroupKFold(n_splits=n_splits).split(train, groups=groups):
            fitted = score_fit(train.iloc[tr], use_time, alpha)
            pred = score_predict(fitted, train.iloc[te])
            heldout_rows.extend({"family": fam, "squared_error": float(err)}
                                for fam, err in zip(train.iloc[te].family,
                                (pred - train.iloc[te].Y.to_numpy()) ** 2))
        heldout = pd.DataFrame(heldout_rows)
        rows.append({"alpha": alpha,
                     "group_cv_rmse": float(np.sqrt(heldout.squared_error.mean())),
                     "group_cv_family_balanced_rmse": float(np.sqrt(
                         heldout.groupby("family").squared_error.mean().mean())),
                     "use_time": use_time, "n_group_folds": n_splits})
    cv = pd.DataFrame(rows)
    return float(cv.loc[cv.group_cv_family_balanced_rmse.idxmin(), "alpha"]), cv


def fit_leaderboard(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = []
    cvs = []
    holdout_rows = []
    fits = {}
    for stratum, filt in [("strict_pretrained", panel.strict_open & panel.type_group.eq("pretrained")),
                          ("strict_posttrained", panel.strict_open & panel.type_group.eq("posttrained")),
                          ("license_only_pretrained", panel.license_only_unverified &
                           panel.type_group.eq("pretrained"))]:
        sub = panel[filt].copy().reset_index(drop=True)
        train = sub[pd.to_datetime(sub.submission) < LATE_START].copy()
        test = sub[pd.to_datetime(sub.submission) >= LATE_START].copy()
        if len(train) < 12 or len(test) < 3 or train.family.nunique() < 3:
            summary.append({"stratum": stratum, "n": len(sub), "families": sub.family.nunique(),
                            "train_n": len(train), "future_holdout_n": len(test),
                            "status": "insufficient_for_model"})
            continue
        predictions = {}
        for variant, use_time in [("N_only", False), ("N_plus_time", True)]:
            alpha, cv = choose_alpha(train, use_time)
            if len(cv):
                cv["stratum"] = stratum
                cv["variant"] = variant
                cvs.append(cv)
            fitted_train = score_fit(train, use_time, alpha)
            pred = score_predict(fitted_train, test)
            predictions[variant] = pred
            holdout_rows.extend({"stratum": stratum, "variant": variant,
                                 "Model": model, "family": fam, "observed": float(obs),
                                 "predicted": float(pr)} for model, fam, obs, pr in zip(
                                 test.Model, test.family, test.Y, pred))
            by_family = pd.DataFrame({"family": test.family.to_numpy(),
                                      "squared_error": (pred - test.Y.to_numpy()) ** 2})
            family_rmse = float(np.sqrt(by_family.groupby("family").squared_error.mean().mean()))
            fitted_full = score_fit(sub, use_time, alpha)
            fits[(stratum, variant)] = fitted_full
            summary.append({"stratum": stratum, "variant": variant, "n": len(sub),
                            "families": sub.family.nunique(), "train_n": len(train),
                            "future_holdout_n": len(test), "alpha": alpha,
                            "future_holdout_rmse": float(np.sqrt(np.mean((pred - test.Y) ** 2))),
                            "future_holdout_family_balanced_rmse": family_rmse,
                            "future_holdout_families": test.family.nunique(),
                            "future_holdout_mae": float(np.mean(abs(pred - test.Y))),
                            "future_holdout_bias": float(np.mean(pred - test.Y)),
                            "status": "fit"})
        base = np.repeat(train.Y.mean(), len(test))
        holdout_rows.extend({"stratum": stratum, "variant": "training_mean",
                             "Model": model, "family": fam, "observed": float(obs),
                             "predicted": float(pr)} for model, fam, obs, pr in zip(
                             test.Model, test.family, test.Y, base))
        base_family = pd.DataFrame({"family": test.family.to_numpy(),
                                    "squared_error": (base - test.Y.to_numpy()) ** 2})
        summary.append({"stratum": stratum, "variant": "training_mean",
                        "n": len(sub), "families": sub.family.nunique(), "train_n": len(train),
                        "future_holdout_n": len(test),
                        "future_holdout_rmse": float(np.sqrt(np.mean((base - test.Y) ** 2))),
                        "future_holdout_family_balanced_rmse": float(np.sqrt(
                            base_family.groupby("family").squared_error.mean().mean())),
                        "future_holdout_families": test.family.nunique(),
                        "future_holdout_mae": float(np.mean(abs(base - test.Y))),
                        "future_holdout_bias": float(np.mean(base - test.Y)), "status": "baseline"})
    fit_table = save(pd.DataFrame(summary), "model_holdout_metrics.csv")
    save(pd.DataFrame(holdout_rows), "future_holdout_predictions.csv")
    if cvs: save(pd.concat(cvs, ignore_index=True), "family_group_cv.csv")
    return fits, fit_table


def temporal_model_comparison() -> pd.DataFrame:
    holdout = pd.read_csv(RESULTS / "future_holdout_predictions.csv")
    rows = []
    boot_rng = np.random.default_rng(20260926)
    for stratum, group in holdout.groupby("stratum"):
        pair = group[group.variant.isin(["N_only", "N_plus_time"])].copy()
        pair["squared_error"] = (pair.predicted - pair.observed) ** 2
        fam = pair.groupby(["family", "variant"]).squared_error.mean().unstack()
        fam = fam.dropna(subset=["N_only", "N_plus_time"])
        if len(fam) < 2: continue
        diff = float(np.sqrt(fam.N_plus_time.mean()) - np.sqrt(fam.N_only.mean()))
        draws = []
        for _ in range(1000):
            chosen = boot_rng.choice(len(fam), size=len(fam), replace=True)
            sampled = fam.iloc[chosen]
            draws.append(float(np.sqrt(sampled.N_plus_time.mean()) -
                               np.sqrt(sampled.N_only.mean())))
        rows.append({"stratum": stratum, "holdout_families": len(fam),
                     "delta_family_balanced_rmse_time_minus_N": diff,
                     "family_bootstrap_ci_low": float(np.quantile(draws, .025)),
                     "family_bootstrap_ci_high": float(np.quantile(draws, .975)),
                     "interpretation": "descriptive_model_comparison_not_independent_long_horizon_test"})
    return save(pd.DataFrame(rows), "temporal_model_comparison.csv")


def rolling_month_validation(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audits = []; metrics = []; predictions = []
    for stratum, group in [("strict_pretrained", "pretrained"),
                           ("strict_posttrained", "posttrained")]:
        sub = panel[panel.strict_open & panel.type_group.eq(group)].copy()
        dates = pd.to_datetime(sub.submission)
        for target in pd.period_range("2024-07", "2025-03", freq="M"):
            test_start = target.to_timestamp()
            test_end = min((target + 1).to_timestamp() - pd.Timedelta(days=1), CUTOFF)
            train = sub[dates.lt(test_start)].copy()
            test = sub[dates.between(test_start, test_end)].copy()
            full_month = test_end == (target + 1).to_timestamp() - pd.Timedelta(days=1)
            support = (float(test.logN.between(train.logN.min(), train.logN.max()).mean())
                       if len(train) and len(test) else np.nan)
            can_fit = len(train) >= 12 and train.family.nunique() >= 3 and len(test) >= 3
            # Descriptive screen, not a proof of statistical power or future validity.
            screen = (can_fit and full_month and len(test) >= 10 and
                      test.family.nunique() >= 3 and support >= .8)
            audit = {"stratum": stratum, "target_month": str(target),
                     "train_end": str((test_start - pd.Timedelta(days=1)).date()),
                     "test_end": str(test_end.date()), "full_calendar_month": full_month,
                     "train_models": len(train), "train_families": train.family.nunique(),
                     "test_models": len(test), "test_families": test.family.nunique(),
                     "unseen_test_families": int((~test.family.isin(train.family)).groupby(
                         test.family).first().sum()) if len(test) else 0,
                     "test_logN_within_train_minmax_fraction": support,
                     "fit_possible": can_fit, "descriptive_screen_pass": screen}
            audits.append(audit)
            if not can_fit: continue
            for variant, use_time in [("N_only", False), ("N_plus_time", True)]:
                alpha, _ = choose_alpha(train, use_time)
                pred = score_predict(score_fit(train, use_time, alpha), test)
                squared = (pred - test.Y.to_numpy()) ** 2
                fam_mse = pd.DataFrame({"family": test.family.to_numpy(),
                                        "squared": squared}).groupby("family").squared.mean()
                metrics.append({"stratum": stratum, "target_month": str(target),
                                "variant": variant, "alpha": alpha,
                                "test_models": len(test), "test_families": test.family.nunique(),
                                "descriptive_screen_pass": screen,
                                "rmse": float(np.sqrt(np.mean(squared))),
                                "family_balanced_rmse": float(np.sqrt(fam_mse.mean())),
                                "bias": float(np.mean(pred - test.Y))})
                predictions.extend({"stratum": stratum, "target_month": str(target),
                                    "variant": variant, "Model": model, "family": family_name,
                                    "observed": float(obs), "predicted": float(yhat)}
                                   for model, family_name, obs, yhat in zip(
                                       test.Model, test.family, test.Y, pred))
            baseline = np.repeat(float(train.Y.mean()), len(test))
            squared = (baseline - test.Y.to_numpy()) ** 2
            fam_mse = pd.DataFrame({"family": test.family.to_numpy(),
                                    "squared": squared}).groupby("family").squared.mean()
            metrics.append({"stratum": stratum, "target_month": str(target),
                            "variant": "training_mean", "alpha": np.nan,
                            "test_models": len(test), "test_families": test.family.nunique(),
                            "descriptive_screen_pass": screen,
                            "rmse": float(np.sqrt(np.mean(squared))),
                            "family_balanced_rmse": float(np.sqrt(fam_mse.mean())),
                            "bias": float(np.mean(baseline - test.Y))})
            predictions.extend({"stratum": stratum, "target_month": str(target),
                                "variant": "training_mean", "Model": model,
                                "family": family_name, "observed": float(obs),
                                "predicted": float(yhat)} for model, family_name, obs, yhat in zip(
                                    test.Model, test.family, test.Y, baseline))
    audit = save(pd.DataFrame(audits), "rolling_month_support_audit.csv")
    metric = save(pd.DataFrame(metrics), "rolling_month_metrics.csv")
    prediction = save(pd.DataFrame(predictions), "rolling_month_predictions.csv")
    return audit, metric, prediction


def family_influence_audit() -> pd.DataFrame:
    holdout = pd.read_csv(RESULTS / "future_holdout_predictions.csv")
    sub = holdout[holdout.stratum.eq("strict_posttrained") &
                  holdout.variant.isin(["N_only", "N_plus_time"])].copy()
    sub["squared"] = (sub.predicted - sub.observed) ** 2
    family_mse = sub.groupby(["family", "variant"]).squared.mean().unstack().dropna()
    full_delta = float(np.sqrt(family_mse.N_plus_time.mean()) -
                       np.sqrt(family_mse.N_only.mean()))
    rows = []
    for family_name in family_mse.index:
        other = family_mse.drop(family_name)
        if len(other) < 2: continue
        without_delta = float(np.sqrt(other.N_plus_time.mean()) -
                              np.sqrt(other.N_only.mean()))
        example = sub[sub.family.eq(family_name)].Model.iloc[0]
        rows.append({"family_heuristic": family_name,
                     "holdout_models": sub[sub.family.eq(family_name)].Model.nunique(),
                     "example_model": example, "full_delta_rmse": full_delta,
                     "delta_without_family": without_delta,
                     "absolute_influence_points": abs(full_delta - without_delta),
                     "base_family_identity_review": "pending_source_evidence"})
    return save(pd.DataFrame(rows).sort_values("absolute_influence_points", ascending=False),
                "family_influence_review_queue.csv")


def decompose(panel: pd.DataFrame, fits: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audits = []; pieces = []; boots = []; local_rows = []
    boot_rng = np.random.default_rng(20260925)
    for stratum, filt in [("strict_pretrained", panel.strict_open & panel.type_group.eq("pretrained")),
                          ("strict_posttrained", panel.strict_open & panel.type_group.eq("posttrained")),
                          ("license_only_pretrained", panel.license_only_unverified &
                           panel.type_group.eq("pretrained"))]:
        sub = panel[filt].copy()
        early = sub[pd.to_datetime(sub.submission) <= EARLY_END].copy()
        late = sub[pd.to_datetime(sub.submission) >= LATE_START].copy()
        lo = max(early.logN.min(), late.logN.min()) if len(early) and len(late) else np.nan
        hi = min(early.logN.max(), late.logN.max()) if len(early) and len(late) else np.nan
        e = early[early.logN.between(lo, hi)].copy()
        l = late[late.logN.between(lo, hi)].copy()
        eligible = (len(e) >= 10 and len(l) >= 10 and e.family.nunique() >= 3 and
                    l.family.nunique() >= 3 and lo < hi and
                    (stratum, "N_plus_time") in fits)
        audits.append({"stratum": stratum, "early_n": len(early), "late_n": len(late),
                       "overlap_logN_low": lo, "overlap_logN_high": hi,
                       "overlap_early_n": len(e), "overlap_late_n": len(l),
                       "overlap_early_families": e.family.nunique(),
                       "overlap_late_families": l.family.nunique(),
                       "eligible": eligible})
        if not eligible: continue
        fitted = fits[(stratum, "N_plus_time")]
        t0, t1 = float(e.t_month.median()), float(l.t_month.median())

        def terms(f: tuple, ee: pd.DataFrame, ll: pd.DataFrame) -> tuple:
            grids = []
            for base, t in [(ee, t0), (ee, t1), (ll, t0), (ll, t1)]:
                temp = base.copy(); temp["t_month"] = t
                grids.append(float(score_predict(f, temp).mean()))
            m00, m01, m10, m11 = grids
            scale = .5 * ((m10 - m00) + (m11 - m01))
            time = .5 * ((m01 - m00) + (m11 - m10))
            return scale, time, m11 - m00

        scale, time, predicted = terms(fitted, e, l)
        local_sample = sub[sub.logN.between(lo, hi)]
        local_fit = score_fit(local_sample, True, fitted[1].alpha)
        local_scale, local_time, local_total = terms(local_fit, e, l)
        local_rows.append({"stratum": stratum, "overlap_fit_n": len(local_sample),
                           "full_fit_scale_points": scale,
                           "overlap_fit_scale_points": local_scale,
                           "full_fit_time_points": time,
                           "overlap_fit_time_points": local_time,
                           "full_fit_total_points": predicted,
                           "overlap_fit_total_points": local_total})
        actual = float(l.Y.mean() - e.Y.mean())
        pieces.append({"stratum": stratum, "scale_points": scale,
                       "conditional_time_points": time, "model_change_points": predicted,
                       "observed_change_points": actual,
                       "unexplained_points": actual - predicted, "t0": t0, "t1": t1,
                       "causal_attribution": False})
        families = sub.family.unique()
        for b in range(200):
            sampled = boot_rng.choice(families, size=len(families), replace=True)
            draw = pd.concat([sub[sub.family.eq(g)] for g in sampled], ignore_index=True)
            ee = draw[pd.to_datetime(draw.submission).le(EARLY_END) & draw.logN.between(lo, hi)]
            ll = draw[pd.to_datetime(draw.submission).ge(LATE_START) & draw.logN.between(lo, hi)]
            if len(ee) < 3 or len(ll) < 3: continue
            try:
                f = score_fit(draw, True, fitted[1].alpha)
                s, t, total = terms(f, ee, ll)
                boots.append({"stratum": stratum, "replicate": b, "scale_points": s,
                              "conditional_time_points": t, "model_change_points": total,
                              "observed_change_points": float(ll.Y.mean() - ee.Y.mean())})
            except ValueError:
                continue
    audit = save(pd.DataFrame(audits), "decomposition_support_audit.csv")
    result = save(pd.DataFrame(pieces), "conditional_decomposition.csv")
    local = save(pd.DataFrame(local_rows), "decomposition_overlap_fit_sensitivity.csv")
    if boots: save(pd.DataFrame(boots), "decomposition_family_bootstrap.csv")
    return audit, result, local


def monthly_frontier(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel.strict_open & panel.type_group.isin(["pretrained", "posttrained"])].copy()
    sub["month"] = pd.to_datetime(sub.submission).dt.to_period("M").astype(str)
    monthly = sub.groupby(["type_group", "month"], as_index=False).agg(
        models=("Model", "size"), families=("family", "nunique"),
        max_score=("Y", "max"), q90_score=("Y", lambda s: float(s.quantile(.9))),
        q90_N_B=("N_B", lambda s: float(s.quantile(.9))))
    return save(monthly, "observed_monthly_frontier.csv")


def compute_sensitivity(links: pd.DataFrame,
                        filename: str = "linked_compute_sensitivity.csv") -> pd.DataFrame:
    sub = links.copy()
    sub["log_compute"] = np.log10(pd.to_numeric(sub["Training compute (FLOP)"], errors="coerce"))
    sub = sub[np.isfinite(sub.log_compute)].copy()
    if len(sub) < 10 or sub.Model_C1.nunique() < 10:
        return save(pd.DataFrame([{"status": "too_few_verified_links", "n": len(sub)}]),
                    filename)
    rows = []
    boot_rng = np.random.default_rng(20260927)
    for variable in ["log_compute", "logN"]:
        rho = spearmanr(sub[variable], sub.Y)
        families = sub.family.unique()
        draws = []
        for _ in range(1000):
            sampled = boot_rng.choice(families, size=len(families), replace=True)
            draw = pd.concat([sub[sub.family.eq(g)] for g in sampled], ignore_index=True)
            if draw[variable].nunique() < 2 or draw.Y.nunique() < 2:
                continue
            statistic = spearmanr(draw[variable], draw.Y).statistic
            if np.isfinite(statistic): draws.append(float(statistic))
        rows.append({"variable": variable, "n": len(sub), "families": sub.family.nunique(),
                     "spearman": float(rho.statistic),
                     "family_bootstrap_ci_low": float(np.quantile(draws, .025)) if draws else np.nan,
                     "family_bootstrap_ci_high": float(np.quantile(draws, .975)) if draws else np.nan,
                     "family_bootstrap_valid_replicates": len(draws),
                     "min": float(sub[variable].min()), "max": float(sub[variable].max()),
                     "status": "association_only_nonrepresentative_linked_subset"})
    return save(pd.DataFrame(rows), filename)


def conditional_scenarios(panel: pd.DataFrame, fits: dict, holdout: pd.DataFrame,
                          monthly: pd.DataFrame) -> pd.DataFrame:
    stratum = "strict_posttrained"
    sub = panel[panel.strict_open & panel.type_group.eq("posttrained")].copy()
    hist = monthly[(monthly.type_group == "posttrained") & (monthly.models >= 3)].copy()
    if len(hist) < 4 or (stratum, "N_only") not in fits:
        return save(pd.DataFrame([{"status": "insufficient_monthly_resource_history"}]),
                    "conditional_frontier_scenarios.csv")
    month_dates = pd.to_datetime(hist.month)
    hist["month_index"] = (month_dates.dt.year - 2024) * 12 + month_dates.dt.month - 6
    rate = float(np.polyfit(hist.month_index, np.log10(hist.q90_N_B), 1)[0])
    anchor = float(np.log10(hist.tail(3).q90_N_B.median()))
    hm = holdout[(holdout.stratum == stratum) & holdout.variant.isin(["N_only", "N_plus_time"])]
    temporal_gain = (len(hm) == 2 and all(
        hm.loc[hm.variant.eq("N_plus_time"), metric].iloc[0] <
        hm.loc[hm.variant.eq("N_only"), metric].iloc[0]
        for metric in ["future_holdout_rmse", "future_holdout_family_balanced_rmse"]))
    # A negative empirical size trend does not identify a positive growth/slowdown
    # path. A time term that loses on future holdout cannot support extrapolation.
    identified = rate > 0 and temporal_gain and len(hist) >= 6
    records = [{"origin": str(CUTOFF.date()), "horizon_months": months,
                "target_date": str((CUTOFF + pd.DateOffset(months=months)).date()),
                "stratum": stratum, "monthly_q90_log10N_trend": rate,
                "last_three_month_q90_N_B": 10 ** anchor,
                "positive_growth_established": bool(rate > 0),
                "time_model_improves_future_holdout": bool(temporal_gain),
                "same_horizon_backtest_available": False,
                "predicted_conditional_score": np.nan,
                "status": "not_identified_no_positive_growth_or_validated_time_effect" if not identified
                          else "not_identified_no_same_horizon_backtest"}
               for months in [12, 24]]
    return save(pd.DataFrame(records), "conditional_frontier_scenarios.csv")


def task_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel.strict_open & panel.type_group.eq("posttrained")].copy()
    early = sub[pd.to_datetime(sub.submission).le(EARLY_END)]
    late = sub[pd.to_datetime(sub.submission).ge(LATE_START)]
    rows = []
    for label, cols in [("six_task_mean", TASKS)] + [(f"exclude_{task}",
            [x for x in TASKS if x != task]) for task in TASKS]:
        rows.append({"score_definition": label, "early_n": len(early), "late_n": len(late),
                     "early_mean": float(early[cols].mean(axis=1).mean()),
                     "late_mean": float(late[cols].mean(axis=1).mean()),
                     "late_minus_early": float(late[cols].mean(axis=1).mean() -
                                               early[cols].mean(axis=1).mean())})
    for task in TASKS:
        rows.append({"score_definition": f"task_{task}", "early_n": len(early),
                     "late_n": len(late), "early_mean": float(early[task].mean()),
                     "late_mean": float(late[task].mean()),
                     "late_minus_early": float(late[task].mean() - early[task].mean())})
    return save(pd.DataFrame(rows), "task_and_leave_one_out_sensitivity.csv")


def task_model_validation(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel.strict_open & panel.type_group.eq("posttrained")].copy()
    train_mask = pd.to_datetime(sub.submission).lt(LATE_START)
    records = []
    for task in TASKS:
        working = sub.copy(); working["Y"] = working[task]
        train = working[train_mask]; test = working[~train_mask]
        for variant, use_time in [("N_only", False), ("N_plus_time", True)]:
            alpha, _ = choose_alpha(train, use_time)
            fitted = score_fit(train, use_time, alpha)
            pred = score_predict(fitted, test)
            records.append({"task": task, "variant": variant,
                            "train_n": len(train), "future_holdout_n": len(test),
                            "alpha": alpha,
                            "future_holdout_rmse": float(np.sqrt(np.mean((pred - test.Y) ** 2))),
                            "future_holdout_bias": float(np.mean(pred - test.Y))})
    return save(pd.DataFrame(records), "task_model_future_holdout.csv")


def draw_figures(panel: pd.DataFrame, flow: pd.DataFrame, monthly: pd.DataFrame,
                 decomposition: pd.DataFrame, bridge_loo: pd.DataFrame,
                 c8_joined: pd.DataFrame, scenarios: pd.DataFrame,
                 rolling_metrics: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(8, 3.5))
    part = flow.iloc[[0, 1, 2, 3, 4]]
    ax.barh(part.step.iloc[::-1], part.rows.iloc[::-1], color="#376b9a")
    ax.set_xlabel("Rows retained")
    ax.set_title("Leaderboard sample audit")
    save_figure(fig, "q4_sample_attrition.pdf")

    strict = panel[panel.strict_open & panel.type_group.isin(["pretrained", "posttrained"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, color in [("pretrained", "#23688e"), ("posttrained", "#c27036")]:
        subset = strict[strict.type_group.eq(group)]
        ax.scatter(subset.logN, subset.Y, s=13, alpha=.62, label=f"{group} (n={len(subset)})", color=color)
    ax.set(xlabel="log10(parameters, B)", ylabel="Six-task mean (%)",
           title="Strict open sample: parameter and score support")
    ax.legend(); save_figure(fig, "q4_scale_score_support.pdf")

    fig, ax = plt.subplots(figsize=(8, 4))
    for group, style in [("pretrained", "o-"), ("posttrained", "s-")]:
        sub = monthly[monthly.type_group.eq(group)]
        ax.plot(pd.to_datetime(sub.month), sub.q90_score, style, label=f"{group} q90")
        for _, row in sub.iterrows():
            ax.annotate(str(int(row.models)), (pd.Timestamp(row.month), row.q90_score),
                        xytext=(0, 4), textcoords="offset points", fontsize=6, ha="center")
    ax.set(xlabel="Submission month", ylabel="Observed 90th percentile score (%)",
           title="Observed frontier; labels are monthly model counts")
    ax.legend(); fig.autofmt_xdate()
    save_figure(fig, "q4_observed_frontier.pdf")

    if len(decomposition):
        boot_file = RESULTS / "decomposition_family_bootstrap.csv"
        boot = pd.read_csv(boot_file) if boot_file.exists() else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        positions = np.arange(len(decomposition))
        for column, shift, label in [("scale_points", -.17, "Parameter distribution"),
                                     ("conditional_time_points", .17, "Conditional time residual")]:
            values = decomposition[column].to_numpy()
            yerr = None
            if len(boot):
                bounds = np.array([boot.loc[boot.stratum.eq(s), column].quantile([.025, .975]).to_numpy()
                                   for s in decomposition.stratum])
                yerr = np.maximum(0, np.vstack([values - bounds[:, 0], bounds[:, 1] - values]))
            ax.bar(positions + shift, values, width=.33, yerr=yerr, capsize=2, label=label)
        ax.axhline(0, color="black", lw=.7)
        labels = ["Strict open posttrained" if x == "strict_posttrained"
                  else "License only; weights unverified" for x in decomposition.stratum]
        ax.set_xticks(positions, labels, rotation=10)
        ax.set_ylabel("Predicted score change (points)")
        ax.set_title("Exploratory decomposition; 2.5-97.5% family-bootstrap intervals")
        ax.legend(fontsize=8)
        save_figure(fig, "q4_conditional_decomposition.pdf")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(bridge_loo.Val_Loss, bridge_loo.LB_Average, label="Pythia, high comparability")
    order = bridge_loo.sort_values("Val_Loss")
    ax.plot(order.Val_Loss, order.loo_monotone_prediction, "--", label="LOO monotone predictions")
    ax.set(xlabel="Validation loss (same source)", ylabel="Leaderboard mean (%)",
           title="C6 local bridge: 7 Pythia observations")
    ax.legend(fontsize=8); save_figure(fig, "q4_loss_benchmark_bridge.pdf")

    common = c8_joined[c8_joined.common_task_set]
    if len(common):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(common.bbh_leaf_macro_raw_pct, common.BBH, s=10, alpha=.5)
        ax.set(xlabel="C8 leaf macro raw accuracy (%)", ylabel="C1 BBH reported score",
               title="Different metric scales; do not assert equality")
        save_figure(fig, "q4_c8_c1_scale_audit.pdf")

    short = rolling_metrics[rolling_metrics.descriptive_screen_pass &
                            rolling_metrics.stratum.eq("strict_posttrained")]
    if len(short):
        fig, ax = plt.subplots(figsize=(7, 3.7))
        for variant, marker in [("N_only", "o"), ("N_plus_time", "s"),
                                ("training_mean", "^")]:
            part = short[short.variant.eq(variant)].sort_values("target_month")
            ax.scatter(part.target_month, part.family_balanced_rmse,
                       marker=marker, s=55, label=variant)
        ax.set(xlabel="Complete next-month holdout", ylabel="Family-balanced RMSE (points)",
               title="Supported short-horizon folds only; no long-horizon inference")
        ax.legend(fontsize=8)
        save_figure(fig, "q4_short_horizon_rolling_validation.pdf")

    # No long-horizon curve when growth and time extrapolation are unidentified.


def write_report(flow: pd.DataFrame, links: pd.DataFrame, usable: pd.DataFrame,
                 c8: dict, bridge_info: dict, fit_metrics: pd.DataFrame,
                 support: pd.DataFrame, decomp: pd.DataFrame,
                 overlap_sensitivity: pd.DataFrame, monthly: pd.DataFrame,
                 compute: pd.DataFrame, scenarios: pd.DataFrame,
                 tasks: pd.DataFrame, task_models: pd.DataFrame,
                 temporal_comparison: pd.DataFrame,
                 publication_audit: pd.DataFrame, manual_review: pd.DataFrame,
                 rolling_audit: pd.DataFrame, rolling_metrics: pd.DataFrame,
                 family_influence: pd.DataFrame,
                 source_review: pd.DataFrame,
                 source_compute: pd.DataFrame) -> None:
    metric_cols = ["stratum", "variant", "n", "families", "train_n",
                   "future_holdout_n", "future_holdout_families", "future_holdout_rmse",
                   "future_holdout_family_balanced_rmse", "future_holdout_bias", "status"]
    metrics_view = fit_metrics.reindex(columns=metric_cols)
    scenario_view = scenarios.reindex(columns=["horizon_months", "monthly_q90_log10N_trend",
        "positive_growth_established", "time_model_improves_future_holdout",
        "same_horizon_backtest_available", "predicted_conditional_score", "status"])
    screened = rolling_metrics.loc[rolling_metrics.descriptive_screen_pass &
                                   rolling_metrics.stratum.eq("strict_posttrained")]
    comparison = screened.pivot(index="target_month", columns="variant",
                                values="family_balanced_rmse")
    better = int((comparison.N_plus_time < comparison.N_only).sum()) if len(comparison) else 0
    worse = int((comparison.N_plus_time > comparison.N_only).sum()) if len(comparison) else 0
    lines = ["# 问题四结果：开放模型能力变化与条件情景", "",
             "> 计算日期基于附件快照。预测原点是同协议 C1 的 2025-03-13，12/24 个月情景没有同长度回测，不能解读为当前实测前沿或可信外推预测。",
             "", "## 数据与三个硬边界", "",
             "1. **开放性：** 主样本要求 C2 权重明确为 Yes 且许可证在预设宽松集合中。这是可复算的操作性筛选，不能替代法律上的开源认证或证明提交当日已经开放。缺失权重证据不视为开放。预训练、后训练分层；合并模型不参与主分解。",
             "2. **跨表连接：** C1/C4 只接受唯一 C4 名称、可支持的来源身份、参数量接近、发布日期可核、语言领域、Confident/Likely 且权重信息无冲突的连接。C4 给出 Hugging Face 开发者 ID 时须与 C1 命名空间一致；未给出 ID 时，同名候选在 C1 侧也须唯一。该表仅是非随机子样本，规则通过不等于人工逐项核验。",
             "3. **Loss 桥接和未来：** C6 高可比只有同一家族的 7 个 Pythia 点；C1 可比评分截至 2025-03，不能把问题三 Loss 换算成高分前沿，也没有 12/24 个月同协议回测。",
             "", "### 样本流失", "", tbl(flow), "",
             "许可证符合预设集合但权重状态缺失的模型，与权重状态明确为 No 的模型分别计数；后者不进入‘状态未知’敏感性层。",
             "", "### 提交日与 Epoch 元数据发布日期", "",
             "Epoch 元数据发布日期可能对应匹配的基础模型，不能直接等同于衍生模型发布时间；晚于提交日的日期更不能视为历史可得信息。下表仅审计两日期差异，主模型的时间变量仍是排行榜提交日。",
             "", tbl(publication_audit), "",
             f"C1/C4 名称候选 {len(links)} 行，规则接受 {int(links.accepted_link.sum())} 行。原始标签下预训练且算力可用 {len(manual_review)} 行；作者资料表明 phi-1 与 phi-4 发布权重经过后训练，纠正分层后严格开放预训练且算力可用 {len(usable)} 行。候选和接受均不等于逐 checkpoint 核验。",
             f"原始标签下这 {len(manual_review)} 条算力连接的来源、参数与算力备注已汇成 `c4_manual_review_queue.csv`；其中开发者 ID 缺失 {int(manual_review.owner_id_missing.sum())} 条。外部来源逐条复核另见 `c4_external_source_review.csv`；它只核对公开资料与 C4 估算口径，未取得逐仓库 checkpoint 哈希或训练遥测。",
             "", "### C8 逐任务核算", "",
             f"扫描 {c8['json_files']} 个 JSON；解析 {c8['parsed']} 个；模型去重后 {c8['models_latest']} 个；与 C1 连接 {c8['joined_to_C1']} 个，其中同一叶任务集合 {c8['common_task_set_joined']} 个。C8 叶任务宏平均与 C1 BBH **数值标尺未证实相同**，因此只做覆盖与秩序审计，不强行回代相等。重复版本以记录时间及文件名排序取末份。",
             "", "## 规模、时间残差与验证", "",
             "模型将六任务均分映射到 logit 空间，拟合 log10 参数量与提交时间的 Ridge；超参数在训练期按模型家族 GroupKFold，并以家族均衡 RMSE 选择。2025-01 至 03 月留作时间留出，比较仅规模、规模加时间与训练均值。除按记录计算 RMSE，还按家族等权平均各家族均方误差后开方，避免提交数多的家族支配验证。时间系数吸收未观测架构、数据、后训练和选择变化，不能解释为纯技术进步。",
             "", tbl(metrics_view), "",
             "两模型在相同留出家族上的家族均衡 RMSE 差（含时间减仅规模）如下；重抽样单位为家族，区间不能验证 12/24 个月外推。",
             "", tbl(temporal_comparison), "",
             "### 逐月短期留出", "",
             "每折只使用目标月份开始前的模型训练，并预测下一日历月。下表同时给出训练/测试家族数、未见过的测试家族及测试参数量落在训练极值内的比例。描述性筛选预设为完整月份、测试至少 10 条且 3 个家族、参数量极值覆盖率至少 80%；这不是统计功效证明，未通过的折也保留审计。",
             "", tbl(rolling_audit[["stratum", "target_month", "train_models", "train_families",
                                    "test_models", "test_families", "unseen_test_families",
                                    "test_logN_within_train_minmax_fraction", "full_calendar_month",
                                    "descriptive_screen_pass"]]), "",
             "通过描述性筛选的短期折如下；逐模型预测及所有可拟合折的误差保存在结果表。多个月份仍共享训练数据与家族，不能把这些折当独立重复实验，更不能充当 12/24 个月回测。",
             "", tbl(rolling_metrics.loc[rolling_metrics.descriptive_screen_pass,
                      ["stratum", "target_month", "variant", "test_models", "test_families",
                       "family_balanced_rmse", "bias"]]), "",
             f"严格后训练通过筛选 {len(comparison)} 折，其中含时间模型家族均衡 RMSE 较低 {better} 折、较高 {worse} 折；严格预训练通过筛选 {int(rolling_audit.loc[rolling_audit.stratum.eq('strict_pretrained'), 'descriptive_screen_pass'].sum())} 折。只报告这种异质性，不据此估计长期趋势。",
             "### 家族划分优先复核队列", "",
             "删除一个留出家族后，时间模型与仅规模模型的家族均衡误差差值会变化。按变化绝对值排序，仅用于确定应先核对哪些基础模型关系；现有家族标签仍是名称启发式，未凭猜测新增归属。",
             "", tbl(family_influence), "", "### 共同规模支持", "", tbl(support), ""]
    if len(decomp):
        lines += ["只有支持审计通过的层才执行早期（截至 2024-08）与晚期（2025-01 至 03）两因素 Shapley 分解。分解数值是模型拟合变化，与实测变化另列；家族聚簇重抽样输出在 `decomposition_family_bootstrap.csv`。未通过者不报贡献百分比。",
                  "", tbl(decomp), ""]
        boot = pd.read_csv(RESULTS / "decomposition_family_bootstrap.csv")
        quant = boot.groupby("stratum")[["scale_points", "conditional_time_points",
                 "model_change_points"]].quantile([.025, .5, .975]).reset_index().rename(
                 columns={"level_1": "quantile"})
        lines += ["家族重抽样分位数如下。严格后训练层的模型总变化区间跨 0；且时间模型没有展示稳健的未来时段留出优势。因此分解只作为探索性统计，不能形成稳定的贡献比例或已验证时间增长机制。许可证单独确认、权重状态缺失层也仅为敏感性。",
                  "", tbl(quant), ""]
        lines += ["将响应面只用早晚期共同参数量支持区间内的记录重拟合，可检查全样本拟合是否由区间外记录驱动。两种拟合的符号或量级不稳时，不能解释为稳健贡献。",
                  "", tbl(overlap_sensitivity), ""]
    else:
        lines += ["所有层均未通过共同规模与独立家族门槛，不报告贡献比例。", ""]
    lines += ["### 任务敏感性", "", tbl(tasks[["score_definition", "late_minus_early"]]), "",
              "六项任务各自用训练期家族分组 CV 选正则化强度，再在 2025-01 至 03 月做时间留出。各任务的尺度和难度不同，原始 RMSE 不能直接相加；它们用于检查综合均分是否掩盖异质性。",
              "", tbl(task_models[["task", "variant", "future_holdout_n",
                                   "future_holdout_rmse", "future_holdout_bias"]]), "",
              "## 算力审计与观测前沿", "",
              "C4 连接子样本只报告关联与按家族重抽样的 Spearman 区间，不使用把相关模型当独立样本的普通 p 值；区间也不消除年代、架构等混杂。C4 独立资源趋势不与 Benchmark 行序拼接。每月观测最大值和 90% 分位随样本量列出；稀疏月份的分位数尤其不稳定。",
              "", tbl(compute), "",
              "公开来源复核把 29 条分为近似预训练算力有来源支持、训练阶段/算力口径不可比、证据不足三类。`source_supported_estimate` 不表示训练方直接测得 FLOPs；也不代表已核对 checkpoint 哈希。下表是来源支持子集的独立敏感性，仍受小样本、家族聚集、年代和架构混杂限制，不作因果推断。",
              "", tbl(source_review.groupby("source_review_status", as_index=False).size()), "",
              tbl(source_compute), "",
              tbl(monthly[["type_group", "month", "models", "families", "max_score", "q90_score"]]), "",
              "## Loss 桥接", "",
              f"C6 高可比 n={bridge_info['high_n']}、独立家族 1。Loss 范围 {bridge_info['high_loss_range']}，得分范围 {bridge_info['high_score_range']}。逐点留一单调映射 RMSE={bridge_info['loo_monotone_rmse']:.4f}，常数基线 RMSE={bridge_info['loo_constant_rmse']:.4f}。这个留一不检验跨家族泛化；前沿数值桥接未识别。",
              "", "## 12/24 个月条件情景", "",
              "观测月度高分位参数量趋势为负，不能据此定义正增长放缓；含时间模型的未来时段留出也不优于仅规模模型。因此不计算 12/24 个月得分，避免把缺少支持的外推数值误当预测。这里的增长率是参数量口径，不是训练算力。",
              "", tbl(scenario_view), "", "## 图表与可复算文件", "",
              "图表在 `outputs/problem4/figures/`，全部由同一脚本生成。原始合同、连接逐行理由、C8 文件审计、留出指标、分解支持和重抽样、资源情景均在 `outputs/problem4/results/`。",
              "", "## 结论边界", "",
              "排行榜提交不是随机实验，也不等于模型发布日期；主结果仅覆盖许可证和权重可核的模型。严格开放预训练的晚期样本较少。粗粒度家族划分与模型重复提交可能缩小有效样本量。C4 的公开来源与估算口径已分级复核，但没有逐 checkpoint 身份、完整训练遥测的确认；来源支持子集也只能用于探索性关联。C8 与 C1 BBH 标尺不同。C6 桥接不支持前三问 Loss 向当前或未来排行榜前沿转译。长期情景没有同长度回测；不得写成已经验证的预测。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_contract()
    panel, flow, publication_audit = prepare_panel(data["C2"])
    links, usable = link_c4(panel, data["C4"])
    c8_joined, c8 = process_c8(panel)
    bridge_loo, bridge_info = bridge(data["C6"])
    fits, metrics = fit_leaderboard(panel)
    temporal_comparison = temporal_model_comparison()
    rolling_audit, rolling_metrics, _ = rolling_month_validation(panel)
    family_influence = family_influence_audit()
    support, decomp, overlap_sensitivity = decompose(panel, fits)
    monthly = monthly_frontier(panel)
    compute = compute_sensitivity(usable)
    source_review = adjudicate(pd.read_csv(RESULTS / "c4_manual_review_queue.csv"))
    save(source_review, "c4_external_source_review.csv")
    source_usable = usable[usable.Model_C1.isin(source_review.loc[
        source_review.source_review_status.eq("source_supported_estimate"), "Model_C1"])].copy()
    source_compute = compute_sensitivity(source_usable,
        "linked_compute_source_supported_sensitivity.csv")
    scenarios = conditional_scenarios(panel, fits, metrics, monthly)
    tasks = task_sensitivity(panel)
    task_models = task_model_validation(panel)
    draw_figures(panel, flow, monthly, decomp, bridge_loo, c8_joined, scenarios,
                 rolling_metrics)
    write_report(flow, links, usable, c8, bridge_info, metrics,
                 support, decomp, overlap_sensitivity, monthly, compute,
                 scenarios, tasks, task_models,
                 temporal_comparison, publication_audit,
                 pd.read_csv(RESULTS / "c4_manual_review_queue.csv"),
                 rolling_audit, rolling_metrics, family_influence,
                 source_review, source_compute)
    print(json.dumps({"strict_pretrained": int((panel.strict_open &
          panel.type_group.eq("pretrained")).sum()), "strict_posttrained": int((panel.strict_open &
          panel.type_group.eq("posttrained")).sum()), "C4_usable_compute": len(usable),
          "C8_parsed": c8["parsed"], "decomposition_eligible": decomp.stratum.tolist(),
          "scenario_rows": len(scenarios)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
