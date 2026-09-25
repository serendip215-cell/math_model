"""Independent checks for Question 4 outputs, using raw files and saved artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后" / "C_efficiency_evolution"
RESULTS = PROJECT / "outputs" / "problem4" / "results"
REPORT = PROJECT / "reports" / "问题四" / "验证验收" / "VERIFY_REPORT_PROBLEM4.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def main() -> None:
    checks = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    raw = pd.read_csv(DATA / "leaderboard_enhanced.csv")
    c3_raw = pd.read_csv(DATA / "leaderboard_extended_timeseries.csv")
    c3_sources = read("c3_source_comparability.csv")
    c3_rows = read("c3_historical_row_audit.csv")
    c3_years = read("c3_historical_year_coverage.csv")
    c3_omitted = read("c3_c1_omitted_rows.csv")
    check("C3 all supplied rows classified", len(c3_raw) == 4599 and
          c3_sources.rows.sum() == len(c3_raw) and len(c3_rows) == 26)
    source_board = c3_sources.set_index("source").loc["Open LLM Leaderboard"]
    source_hist = c3_sources.set_index("source").loc["Historical (papers/reports)"]
    check("C3 leaderboard rows are C1 duplicates up to rounding",
          int(source_board.rows) == 4573 and
          int(source_board.exact_name_overlap_with_C1) == 4573 and
          float(source_board.max_nearest_C1_average_difference) <= .00501)
    c1_raw = pd.read_csv(DATA / "leaderboard_cleaned.csv")
    board_names = set(c3_raw.loc[c3_raw.Source.eq("Open LLM Leaderboard"), "Model"])
    check("C1 records omitted from C3 have missing parameter size",
          len(c3_omitted) == 3 and c3_omitted["#Params (B)"].isna().all() and
          set(c3_omitted.Model) == set(c1_raw.loc[~c1_raw.Model.isin(board_names), "Model"]))
    raw_hist = c3_raw[c3_raw.Source.eq("Historical (papers/reports)")].set_index("Model")
    saved_hist = c3_rows.set_index("Model")
    columns = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]
    check("C3 historical score discrepancy independently recomputed",
          saved_hist.index.is_unique and set(saved_hist.index) == set(raw_hist.index) and
          np.allclose(saved_hist.six_task_mean,
                      raw_hist.loc[saved_hist.index, columns].mean(axis=1)) and
          np.allclose(saved_hist.abs_average_discrepancy,
                      (raw_hist.loc[saved_hist.index, "Average"] -
                       raw_hist.loc[saved_hist.index, columns].mean(axis=1)).abs()) and
          int(source_hist.rows_abs_discrepancy_gt_0_01) == 26 and
          not c3_years.usable_for_C1_protocol_trend.any())
    panel = read("audited_leaderboard_panel.csv")
    flow = read("panel_attrition.csv")
    check("C2 raw row contract", len(raw) == 4576)
    check("Panel model unique", panel.Model.is_unique)
    check("Six-task average", np.allclose(panel[TASKS].mean(axis=1), panel.Y, atol=1e-10))
    check("Strict open evidence", panel.loc[panel.strict_open, "Epoch_AI_Open_Weights"].eq("Yes").all()
          and panel.loc[panel.strict_open, "Hub License"].isin(PERMISSIVE).all())
    check("Unknown weight stratum excludes explicit No", panel.loc[
          panel.license_only_unverified, "Epoch_AI_Open_Weights"].isna().all() and
          panel.loc[panel.license_explicit_no, "Epoch_AI_Open_Weights"].eq("No").all())
    check("Panel cutoff", pd.to_datetime(panel.submission).max() <= pd.Timestamp("2025-03-13"))
    phi = panel[panel.Model.isin(["microsoft/phi-1", "microsoft/phi-4"])]
    check("Main panel preserves attachment Phi type labels",
          len(phi) == 2 and phi.type_group_raw.eq("pretrained").all() and
          phi.type_group.eq("pretrained").all())
    check("Pretrained late count", len(panel[panel.strict_open &
          panel.type_group.eq("pretrained") & pd.to_datetime(panel.submission).ge("2025-01-01")]) == 5)
    check("Flow raw row count", int(flow.iloc[0].rows) == len(raw))
    publication = read("publication_submission_audit.csv")
    check("Publication audit totals strict pre and post", publication.models.sum() ==
          len(panel[panel.strict_open & panel.type_group.isin(["pretrained", "posttrained"])])
          and publication.publication_after_submission.le(publication.models).all())
    pub_lag = (pd.to_datetime(panel.submission) -
               pd.to_datetime(panel.epoch_publication, errors="coerce")).dt.days
    check("Publication lag independently recomputed", np.allclose(
          pub_lag.fillna(-99999), panel.publication_lag_days.fillna(-99999)))

    links = read("c1_c4_link_audit.csv")
    accepted = links[links.accepted_link]
    rule_cols = ["C4_key_unique", "source_identity_supported", "N_consistent",
                 "date_consistent", "language_domain",
                 "source_confident", "weight_agreement"]
    check("Every accepted link passes all rules", accepted[rule_cols].all(axis=None))
    check("Accepted C4 records are one-to-one", accepted.Model_C4.is_unique and
          accepted.Model_C1.is_unique)
    has_dev = accepted[accepted.developer_id_available]
    check("Known developer id agrees with repository namespace",
          has_dev.developer_id_matches.all())
    usable = read("c4_compute_linked_subset.csv")
    check("Compute subset has positive recorded compute", usable.usable_compute.all() and
          pd.to_numeric(usable["Training compute (FLOP)"], errors="coerce").gt(0).all())
    check("Compute subset only strict pretrained", usable.strict_open.all() and
          usable.type_group.eq("pretrained").all())
    review = read("c4_manual_review_queue.csv")
    raw_candidates = links[links.review_candidate]
    check("Source review queue retains original-label candidates",
          len(review) == 29 and len(raw_candidates) == 29 and
          set(review.Model_C1) == set(raw_candidates.Model_C1) and review.Model_C4.is_unique)
    check("Checkpoint and training telemetry review not falsely marked complete",
          review.manual_review_status.eq(
          "pending_checkpoint_and_training_telemetry_check").all() and
          review.repository_revision_evidence.isna().all())
    raw_c4 = pd.read_csv(DATA / "epoch_all_ai_models.csv", low_memory=False)
    original = review.merge(raw_c4[["Model", "Link", "Training compute notes"]],
                            left_on="Model_C4", right_on="Model", suffixes=("_saved", "_raw"))
    normalized_raw_link = original.Link_raw.astype("string").str.replace(
        r"[ \t]+(?=\r?\n|$)", "", regex=True)
    check("Manual review source fields copied from C4", len(original) == len(review) and
          original.Link_saved.fillna("").eq(normalized_raw_link.fillna("")).all() and
          original["Training compute notes_saved"].fillna("").eq(
              original["Training compute notes_raw"].fillna("")).all())
    source_review = read("c4_external_source_review.csv")
    check("External source review covers every original candidate once", len(source_review) == len(review)
          and source_review.Model_C1.is_unique and
          set(source_review.Model_C1) == set(review.Model_C1) and
          source_review.primary_evidence_url.str.startswith("https://").all())
    source_counts = source_review.source_review_status.value_counts().to_dict()
    check("Source review keeps three evidence classes distinct", source_counts == {
        "source_supported_estimate": 15, "not_comparable": 5,
        "insufficient_evidence": 9})
    approximate = source_review[source_review.source_review_status.eq(
        "source_supported_estimate")]
    check("Source-derived FLOPs numerically reconcile with C4 approximate values",
          approximate.source_derived_flops.notna().all() and
          approximate.source_vs_c4_relative_difference.le(.05).all() and
          np.allclose(approximate.source_vs_c4_relative_difference,
              (approximate["Training compute (FLOP)"].astype(float) -
               approximate.source_derived_flops).abs() / approximate.source_derived_flops))
    not_comparable = {"01-ai/Yi-1.5-34B", "01-ai/Yi-1.5-9B",
        "Qwen/Qwen2-57B-A14B", "microsoft/phi-1", "microsoft/phi-4"}
    check("Stage-mismatched records excluded from supported subset",
          set(source_review.loc[source_review.source_review_status.eq(
              "not_comparable"), "Model_C1"]) == not_comparable)

    holdout = read("future_holdout_predictions.csv")
    metrics = read("model_holdout_metrics.csv")
    fit_metrics = metrics[metrics.status.isin(["fit", "baseline"])]
    cv = read("family_group_cv.csv")
    selected_ok = []
    for item in metrics[metrics.status.eq("fit")].itertuples():
        candidate = cv[(cv.stratum == item.stratum) & (cv.variant == item.variant)]
        selected_ok.append(np.isclose(item.alpha, candidate.loc[
            candidate.group_cv_family_balanced_rmse.idxmin(), "alpha"]))
    check("Ridge alpha selected by family-balanced CV", all(selected_ok))
    row_ok = []; family_ok = []
    for item in fit_metrics.itertuples():
        rows = holdout[(holdout.stratum == item.stratum) & (holdout.variant == item.variant)]
        sq = (rows.predicted - rows.observed) ** 2
        row_ok.append(len(rows) == item.future_holdout_n and np.isclose(
            np.sqrt(sq.mean()), item.future_holdout_rmse))
        family_mse = rows.assign(squared_error=sq).groupby("family").squared_error.mean()
        family_ok.append(len(family_mse) == item.future_holdout_families and np.isclose(
            np.sqrt(family_mse.mean()), item.future_holdout_family_balanced_rmse))
    check("Time holdout row RMSE independently recomputed", all(row_ok))
    check("Time holdout family-balanced RMSE independently recomputed", all(family_ok))
    temporal = read("temporal_model_comparison.csv")
    comparison_ok = []
    for item in temporal.itertuples():
        a = metrics[(metrics.stratum == item.stratum) & metrics.variant.eq("N_only")]
        b = metrics[(metrics.stratum == item.stratum) & metrics.variant.eq("N_plus_time")]
        comparison_ok.append(np.isclose(b.future_holdout_family_balanced_rmse.iloc[0] -
                                        a.future_holdout_family_balanced_rmse.iloc[0],
                                        item.delta_family_balanced_rmse_time_minus_N))
    check("Temporal model comparison agrees with saved holdout errors", all(comparison_ok))
    check("Temporal comparison bootstrap bounds valid", temporal.family_bootstrap_ci_low.le(
          temporal.family_bootstrap_ci_high).all())
    rolling = read("rolling_month_support_audit.csv")
    rolling_metrics = read("rolling_month_metrics.csv")
    rolling_predictions = read("rolling_month_predictions.csv")
    screen_ok = (rolling.descriptive_screen_pass.le(rolling.fit_possible).all() and
                 rolling.loc[rolling.descriptive_screen_pass, "test_models"].ge(10).all() and
                 rolling.loc[rolling.descriptive_screen_pass, "test_families"].ge(3).all() and
                 rolling.loc[rolling.descriptive_screen_pass,
                             "test_logN_within_train_minmax_fraction"].ge(.8).all() and
                 rolling.loc[rolling.descriptive_screen_pass, "full_calendar_month"].all())
    check("Rolling descriptive screen rules independently checked", screen_ok)
    month_count_ok = []
    for item in rolling.itertuples():
        group = "pretrained" if item.stratum == "strict_pretrained" else "posttrained"
        source = panel[panel.strict_open & panel.type_group.eq(group)]
        date = pd.to_datetime(source.submission)
        observed_train = source[date.le(item.train_end)]
        observed_test = source[date.gt(item.train_end) & date.le(item.test_end)]
        month_count_ok.append(len(observed_train) == item.train_models and
                              len(observed_test) == item.test_models and
                              observed_test.family.nunique() == item.test_families)
    check("Rolling train and test counts recomputed from panel", all(month_count_ok))
    rolling_error_ok = []
    for item in rolling_metrics.itertuples():
        rows = rolling_predictions[(rolling_predictions.stratum == item.stratum) &
            (rolling_predictions.target_month == item.target_month) &
            (rolling_predictions.variant == item.variant)]
        squared = (rows.predicted - rows.observed) ** 2
        family_mse = rows.assign(squared=squared).groupby("family").squared.mean()
        rolling_error_ok.append(len(rows) == item.test_models and
            np.isclose(np.sqrt(squared.mean()), item.rmse) and
            np.isclose(np.sqrt(family_mse.mean()), item.family_balanced_rmse))
    check("Rolling errors independently recomputed", all(rolling_error_ok))
    influence = read("family_influence_review_queue.csv")
    check("Family review queue is ranked and pending", influence.absolute_influence_points.is_monotonic_decreasing
          and influence.base_family_identity_review.eq("pending_source_evidence").all()
          and influence.family_heuristic.is_unique)

    compute = read("linked_compute_sensitivity.csv")
    check("Compute association bootstrap intervals valid", compute.families.ge(2).all() and
          compute.family_bootstrap_valid_replicates.ge(900).all() and
          compute.family_bootstrap_ci_low.le(compute.family_bootstrap_ci_high).all() and
          compute.family_bootstrap_ci_low.ge(-1).all() and
          compute.family_bootstrap_ci_high.le(1).all())
    source_compute = read("linked_compute_source_supported_sensitivity.csv")
    chosen = source_review.loc[source_review.source_review_status.eq(
        "source_supported_estimate"), "Model_C1"]
    chosen_rows = usable[usable.Model_C1.isin(chosen)].copy()
    chosen_rows = chosen_rows.merge(panel[["Model", "Y"]],
        left_on="Model_C1", right_on="Model", validate="one_to_one")
    chosen_rows["log_compute"] = np.log10(chosen_rows["Training compute (FLOP)"].astype(float))
    expected_rho = chosen_rows[["log_compute", "Y"]].corr(method="spearman").iloc[0, 1]
    check("Source-supported compute subset correlation independently recomputed",
          len(chosen_rows) == 15 and source_compute.n.eq(15).all() and
          np.isclose(source_compute.loc[source_compute.variable.eq("log_compute"),
                                       "spearman"].iloc[0], expected_rho))
    check("Source-supported compute uncertainty includes zero",
          source_compute.loc[source_compute.variable.eq("log_compute"),
                             "family_bootstrap_ci_low"].iloc[0] < 0 <
          source_compute.loc[source_compute.variable.eq("log_compute"),
                             "family_bootstrap_ci_high"].iloc[0])

    c8audit = read("c8_file_audit.csv")
    c8 = read("c8_bbh_leaf_aggregation.csv")
    check("C8 file count", len(c8audit) == 1954)
    check("C8 timestamps numeric", pd.to_numeric(c8audit.eval_timestamp,
          errors="coerce").notna().all())
    check("C8 unique latest model", c8.Model.is_unique)
    check("C8 parsed leaf average independently", all(
        np.isclose(100 * np.mean([v["acc_norm,none"] for k, v in
            json.loads((DATA / row.file).read_text(encoding="utf-8"))["results"].items()
            if k.startswith("leaderboard_bbh_") and "acc_norm,none" in v]),
            row.bbh_leaf_macro_raw_pct, atol=1e-10)
        for row in c8.head(20).itertuples()))
    check("C8 different scale explicit", json.loads((RESULTS / "c8_summary.json").read_text(
        encoding="utf-8"))["same_scale_as_C1_established"] is False)

    support = read("decomposition_support_audit.csv")
    decomp = read("conditional_decomposition.csv")
    check("No unsupported decomposition", set(decomp.stratum).issubset(
        set(support.loc[support.eligible, "stratum"])))
    check("Shapley additivity", np.allclose(decomp.scale_points +
          decomp.conditional_time_points, decomp.model_change_points, atol=1e-8))
    check("Observed change residual additivity", np.allclose(decomp.model_change_points +
          decomp.unexplained_points, decomp.observed_change_points, atol=1e-8))
    boots = read("decomposition_family_bootstrap.csv")
    check("Bootstrap replicate additivity", np.allclose(boots.scale_points +
          boots.conditional_time_points, boots.model_change_points, atol=1e-8))
    check("Strict pretrained gate failed", not bool(support.loc[
          support.stratum.eq("strict_pretrained"), "eligible"].iloc[0]))
    overlap = read("decomposition_overlap_fit_sensitivity.csv")
    check("Overlap-only fit decomposition additive", np.allclose(
          overlap.overlap_fit_scale_points + overlap.overlap_fit_time_points,
          overlap.overlap_fit_total_points, atol=1e-8))

    bridge = json.loads((RESULTS / "bridge_summary.json").read_text(encoding="utf-8"))
    check("Bridge high only seven and no frontier translation", bridge["high_n"] == 7
          and bridge["high_unique_family"] == 1
          and bridge["frontier_translation_identified"] is False)
    loo = read("bridge_high_loo.csv")
    check("Bridge LOO RMSE", np.isclose(np.sqrt(np.mean((loo.loo_monotone_prediction -
          loo.LB_Average) ** 2)), bridge["loo_monotone_rmse"]))

    scenario = read("conditional_frontier_scenarios.csv")
    calibration = read("conditional_scenario_calibration.csv").iloc[0]
    raw_c4["publication"] = pd.to_datetime(raw_c4["Publication date"], errors="coerce")
    raw_c4["compute"] = pd.to_numeric(raw_c4["Training compute (FLOP)"], errors="coerce")
    c4_eligible = raw_c4[raw_c4.Domain.fillna("").str.contains("Language", case=False) &
                         raw_c4["Open model weights?"].eq("Yes") &
                         raw_c4.Confidence.isin(["Confident", "Likely"]) &
                         raw_c4.publication.between("2019-01-01", "2025-03-13")]
    med = c4_eligible.groupby(c4_eligible.publication.dt.year).compute.median()
    check("Compute scenario anchor and growth ratios from C4",
          np.isclose(calibration.C4_anchor_median_compute_FLOPs, med.loc[2025]) and
          np.isclose(calibration.C4_2022_to_2023_median_ratio, med.loc[2023] / med.loc[2022]) and
          np.isclose(calibration.C4_2023_to_2024_median_ratio, med.loc[2024] / med.loc[2023]) and
          int(calibration.C4_anchor_models_with_compute) == int(c4_eligible.loc[
              c4_eligible.publication.dt.year.eq(2025), "compute"].gt(0).sum()))
    supported_names = set(source_review.loc[
        source_review.source_review_status.eq("source_supported_estimate"), "Model_C1"])
    supported_linked = read("c4_compute_linked_subset.csv")
    supported_linked = supported_linked[supported_linked.Model_C1.isin(supported_names)]
    check("Source-supported compute is below scenario reference",
          len(supported_linked) == int(calibration.source_supported_paired_models) == 15 and
          np.isclose(supported_linked["Training compute (FLOP)"].max(),
                     calibration.source_supported_max_compute_FLOPs) and
          bool(calibration.reference_exceeds_source_supported_compute_max) and
          calibration.C4_anchor_median_compute_FLOPs >
          calibration.source_supported_max_compute_FLOPs)
    check("Complete 12 and 24 month assumption grid", len(scenario) == 8 and
          set(scenario.horizon_months) == {12, 24} and scenario.groupby("horizon_months").scenario.nunique().eq(4).all())
    check("Scenario FLOPs reproduce conditional arithmetic", np.allclose(
          scenario.conditional_compute_FLOPs,
          calibration.C4_anchor_median_compute_FLOPs * np.exp(
              scenario.annual_log_compute_growth_assumption * scenario.horizon_months / 12)))
    strict_post = panel[panel.strict_open & panel.type_group.eq("posttrained")]
    check("Logical score bound uses observed cumulative record", np.allclose(
          scenario.cumulative_frontier_logical_lower_score, strict_post.Y.max()) and
          scenario.cumulative_frontier_logical_upper_score.eq(100).all() and
          scenario.bound_type.eq("deterministic_logical_bound_not_confidence_interval").all())
    check("No unsupported future score emitted", scenario.predicted_conditional_score.isna().all()
          and not scenario.same_horizon_backtest_available.any() and
          int(calibration.paired_strict_posttrained_compute_models) == 0)
    check("Scenario negative size trend and time gate recorded",
          scenario.monthly_q90_log10N_trend.lt(0).all() and
          not scenario.time_model_improves_future_holdout.any())

    lines = ["# 问题四独立验收", "", f"通过 {sum(ok for _, ok, _ in checks)}/{len(checks)} 项。", "",
             "| 检查 | 结果 | 说明 |", "| --- | --- | --- |"]
    lines += [f"| {name} | {'通过' if ok else '失败'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "## 验收范围", "",
              "这些检查独立读取原始 C2/C3/C4/C8 与保存产物，核对口径、守门条件和数值回代。C3 历史补录的分数口径不一致，因此只用于来源与趋势可比性审计。外部来源网页的具体文字由逐行来源审计记录，本脚本只检查审计文件的覆盖和结果一致性；它不证明 checkpoint 哈希一致，不把观察性分解升级为因果识别，也不证明未来外推可信。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    if not all(ok for _, ok, _ in checks):
        raise AssertionError("Question 4 verification failed: " +
                             ", ".join(name for name, ok, _ in checks if not ok))
    print(f"Verified {len(checks)} Question 4 checks")


if __name__ == "__main__":
    main()
