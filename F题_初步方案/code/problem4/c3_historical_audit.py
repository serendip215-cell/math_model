"""Audit the supplied C3 timeline without treating mixed protocols as one score series."""
from __future__ import annotations

import numpy as np
import pandas as pd


C3_TASKS = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]


def audit_c3(c1: pd.DataFrame, c3: pd.DataFrame, results) -> dict:
    required = {"Model", "Year", "Params_B", "Average", "Source", *C3_TASKS}
    if not required.issubset(c3.columns):
        raise ValueError(f"C3 columns missing: {sorted(required - set(c3.columns))}")
    board = c3[c3.Source.eq("Open LLM Leaderboard")].copy()
    historical = c3[c3.Source.eq("Historical (papers/reports)")].copy()
    if len(board) != 4573 or len(historical) != 26:
        raise ValueError("C3 source contract changed; re-audit before using results")
    c1_models = set(c1.Model)
    board["present_in_C1_by_name"] = board.Model.isin(c1_models)
    historical["present_in_C1_by_name"] = historical.Model.isin(c1_models)
    if not board.present_in_C1_by_name.all():
        raise ValueError("C3 leaderboard records are no longer a name subset of C1")
    missing_from_c3 = c1.loc[~c1.Model.isin(board.Model),
                             ["Model", "Submission Date", "#Params (B)", "Average ⬆️"]].copy()
    if len(missing_from_c3) != 3 or missing_from_c3["#Params (B)"].notna().any():
        raise ValueError("C1-to-C3 omitted rows changed; re-audit source relationship")
    missing_from_c3.to_csv(results / "c3_c1_omitted_rows.csv", index=False, encoding="utf-8-sig")
    c1_score_by_model = c1.groupby("Model")["Average ⬆️"].apply(
        lambda values: values.to_numpy(dtype=float)).to_dict()
    board["nearest_C1_average_difference"] = [
        float(np.min(np.abs(c1_score_by_model[name] - score)))
        for name, score in zip(board.Model, board.Average)]
    if board.nearest_C1_average_difference.gt(.01).any():
        raise ValueError("C3 leaderboard scores no longer match C1 to rounding precision")
    for frame in (board, historical):
        frame["six_task_mean"] = frame[C3_TASKS].mean(axis=1)
        frame["average_minus_six_task_mean"] = frame.Average - frame.six_task_mean
        frame["abs_average_discrepancy"] = frame.average_minus_six_task_mean.abs()
        frame["zero_task_count"] = frame[C3_TASKS].eq(0).sum(axis=1)
    historical["score_protocol_status"] = np.where(
        historical.abs_average_discrepancy.le(.01),
        "aggregate_numerically_consistent_protocol_unverified",
        "aggregate_inconsistent_with_C1_six_task_definition")
    historical[["Model", "Year", "Params_B", "Source", "Average", *C3_TASKS,
                "six_task_mean", "average_minus_six_task_mean",
                "abs_average_discrepancy", "zero_task_count", "present_in_C1_by_name",
                "score_protocol_status"]].to_csv(
        results / "c3_historical_row_audit.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([
        {"source": "Open LLM Leaderboard", "rows": len(board),
         "year_min": int(board.Year.min()), "year_max": int(board.Year.max()),
         "unique_models": board.Model.nunique(),
         "exact_name_overlap_with_C1": int(board.present_in_C1_by_name.sum()),
         "max_nearest_C1_average_difference": float(board.nearest_C1_average_difference.max()),
         "median_abs_average_minus_six_mean": float(board.abs_average_discrepancy.median()),
         "rows_abs_discrepancy_gt_0_01": int(board.abs_average_discrepancy.gt(.01).sum()),
         "rows_with_zero_task": int(board.zero_task_count.gt(0).sum())},
        {"source": "Historical (papers/reports)", "rows": len(historical),
         "year_min": int(historical.Year.min()), "year_max": int(historical.Year.max()),
         "unique_models": historical.Model.nunique(),
         "exact_name_overlap_with_C1": int(historical.present_in_C1_by_name.sum()),
         "max_nearest_C1_average_difference": np.nan,
         "median_abs_average_minus_six_mean": float(historical.abs_average_discrepancy.median()),
         "rows_abs_discrepancy_gt_0_01": int(historical.abs_average_discrepancy.gt(.01).sum()),
         "rows_with_zero_task": int(historical.zero_task_count.gt(0).sum())},
    ])
    summary.to_csv(results / "c3_source_comparability.csv", index=False, encoding="utf-8-sig")
    years = historical.groupby("Year", as_index=False).agg(
        records=("Model", "size"),
        median_reported_average=("Average", "median"),
        median_six_task_mean=("six_task_mean", "median"),
        median_abs_discrepancy=("abs_average_discrepancy", "median"),
        records_with_zero_task=("zero_task_count", lambda s: int(s.gt(0).sum())))
    years["usable_for_C1_protocol_trend"] = False
    years.to_csv(results / "c3_historical_year_coverage.csv", index=False, encoding="utf-8-sig")
    return {"summary": summary, "historical": historical, "years": years,
            "missing_from_c3": missing_from_c3}
