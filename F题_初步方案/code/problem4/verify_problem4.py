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
    panel = read("audited_leaderboard_panel.csv")
    flow = read("panel_attrition.csv")
    check("C2 raw row contract", len(raw) == 4576)
    check("Panel model unique", panel.Model.is_unique)
    check("Six-task average", np.allclose(panel[TASKS].mean(axis=1), panel.Y, atol=1e-10))
    check("Strict open evidence", panel.loc[panel.strict_open, "Epoch_AI_Open_Weights"].eq("Yes").all()
          and panel.loc[panel.strict_open, "Hub License"].isin(PERMISSIVE).all())
    check("Panel cutoff", pd.to_datetime(panel.submission).max() <= pd.Timestamp("2025-03-13"))
    check("Pretrained late count", len(panel[panel.strict_open &
          panel.type_group.eq("pretrained") & pd.to_datetime(panel.submission).ge("2025-01-01")]) == 5)
    check("Flow raw row count", int(flow.iloc[0].rows) == len(raw))

    links = read("c1_c4_link_audit.csv")
    accepted = links[links.accepted_link]
    rule_cols = ["C4_key_unique", "N_consistent", "date_consistent", "language_domain",
                 "source_confident", "weight_agreement"]
    check("Every accepted link passes all rules", accepted[rule_cols].all(axis=None))
    usable = read("c4_compute_linked_subset.csv")
    check("Compute subset has verified positive compute", usable.usable_compute.all() and
          pd.to_numeric(usable["Training compute (FLOP)"], errors="coerce").gt(0).all())
    check("Compute subset only strict pretrained", usable.strict_open.all() and
          usable.type_group.eq("pretrained").all())

    c8audit = read("c8_file_audit.csv")
    c8 = read("c8_bbh_leaf_aggregation.csv")
    check("C8 file count", len(c8audit) == 1954)
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

    bridge = json.loads((RESULTS / "bridge_summary.json").read_text(encoding="utf-8"))
    check("Bridge high only seven and no frontier translation", bridge["high_n"] == 7
          and bridge["high_unique_family"] == 1
          and bridge["frontier_translation_identified"] is False)
    loo = read("bridge_high_loo.csv")
    check("Bridge LOO RMSE", np.isclose(np.sqrt(np.mean((loo.loo_monotone_prediction -
          loo.LB_Average) ** 2)), bridge["loo_monotone_rmse"]))

    scenario = read("conditional_frontier_scenarios.csv")
    check("No unsupported future score emitted", scenario.predicted_conditional_score.isna().all()
          and not scenario.same_horizon_backtest_available.any())
    check("Scenario negative size trend recorded", scenario.monthly_q90_log10N_trend.lt(0).all())
    check("Scenario time model holdout gate failed", not scenario.time_model_improves_future_holdout.any())

    lines = ["# 问题四独立验收", "", f"通过 {sum(ok for _, ok, _ in checks)}/{len(checks)} 项。", "",
             "| 检查 | 结果 | 说明 |", "| --- | --- | --- |"]
    lines += [f"| {name} | {'通过' if ok else '失败'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "## 验收范围", "",
              "这些检查独立读取原始 C2/C8 与保存产物，核对口径、守门条件和数值回代。它们不把观察性分解升级为因果识别，也不证明未来外推可信。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    if not all(ok for _, ok, _ in checks):
        raise AssertionError("Question 4 verification failed: " +
                             ", ".join(name for name, ok, _ in checks if not ok))
    print(f"Verified {len(checks)} Question 4 checks")


if __name__ == "__main__":
    main()
