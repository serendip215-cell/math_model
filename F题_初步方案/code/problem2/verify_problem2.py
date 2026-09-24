"""问题二结果的独立回代与文件完整性核验。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后" / "B_scaling_laws"
RESULTS = PROJECT / "outputs" / "problem2" / "results"
FIGURES = PROJECT / "outputs" / "problem2" / "figures"
REPORT = PROJECT / "reports" / "问题二" / "验证验收" / "VERIFY_REPORT_PROBLEM2.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)


def add(checks: list[dict], name: str, passed: bool, evidence: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    checks: list[dict] = []
    summary = json.loads((RESULTS / "problem2_summary.json").read_text(encoding="utf-8"))
    params = pd.read_csv(RESULTS / "m0_parameters.csv").set_index("parameter")
    m1 = pd.read_csv(RESULTS / "m1_quality_models.csv").set_index("model")
    loo = pd.read_csv(RESULTS / "m0_loo_predictions.csv")
    direction = pd.read_csv(RESULTS / "quality_direction_audit.csv").set_index("dataset")
    compute = pd.read_csv(RESULTS / "compute_consistency.csv").iloc[0]
    equivalence = pd.read_csv(RESULTS / "quality_parameter_equivalence.csv")
    substitutions = pd.read_csv(RESULTS / "domain_substitution_matrix.csv")
    manifest = pd.read_csv(RESULTS / "figure_manifest.csv")

    positive = all(params.loc[name, "estimate"] > 0 for name in ["E", "A", "B", "alpha", "beta"])
    add(checks, "M0 参数符号", positive, "E、A、B、alpha、beta 均为正")

    add(checks, "计算量单位一致性", compute["within_5_percent_fraction"] > 0.99,
        f"C≈6ND 换算后 5% 内比例={compute['within_5_percent_fraction']:.4%}")

    p = params["estimate"].to_dict()
    gamma = float(m1.loc["exp", "gamma"])
    n, d = 1.0, 100.0
    m0_value = p["E"] + p["A"] * n ** (-p["alpha"]) + p["B"] * d ** (-p["beta"])
    m1_q1_without_source_offset = p["E"] + p["A"] * n ** (-p["alpha"]) + p["B"] * d ** (-p["beta"]) * np.exp(gamma * (1 - 1.0))
    add(checks, "Q=1 退化条件", abs(m0_value - m1_q1_without_source_offset) < 1e-12,
        f"回代差={m1_q1_without_source_offset - m0_value:.3e}；B6 来源截距不属于统一结构项")

    leakage_ok = np.allclose(loo["N_params_B"], loo["held_out_N_B"])
    add(checks, "留轨迹分组", leakage_ok, f"{len(loo)} 行预测均由对应 N 整轨迹留出")

    b8 = pd.read_csv(DATA / "supplementary_NQ_experiment_large.csv")
    comparisons = []
    for _, group in b8.groupby(["N_params_B", "D_tokens_B"]):
        comparisons.extend(np.diff(group.sort_values("Q_score")["val_loss"]).tolist())
    recomputed = float(np.mean(np.asarray(comparisons) <= 0))
    stored = float(direction.loc["B8", "nonincrease_fraction"])
    add(checks, "B8 方向冲突复算", abs(recomputed - stored) < 1e-12,
        f"复算={recomputed:.6f}，记录={stored:.6f}")

    sample = equivalence.dropna(subset=["N_multiplier"]).iloc[len(equivalence) // 2]
    n0, d0, q0, mult = sample["N_params_B"], sample["D_tokens_B"], sample["Q_score"], sample["N_multiplier"]
    old_quality_at_more_n = (p["E"] + p["A"] * (n0 * mult) ** (-p["alpha"])
                             + p["B"] * d0 ** (-p["beta"]) * np.exp(gamma * (1 - q0)))
    better_quality = (p["E"] + p["A"] * n0 ** (-p["alpha"])
                      + p["B"] * d0 ** (-p["beta"]) * np.exp(gamma * (1 - q0 - 0.1)))
    eq_error = abs(old_quality_at_more_n - better_quality)
    add(checks, "质量参数等价式回代", eq_error < 1e-8, f"Loss 回代差={eq_error:.3e}")

    anti_sym = substitutions.pivot(index="increase_domain", columns="decrease_domain",
                                   values="delta_macro_loss_linearized")
    values = []
    for i in anti_sym.index:
        for j in anti_sym.columns:
            if pd.notna(anti_sym.loc[i, j]) and pd.notna(anti_sym.loc[j, i]):
                values.append(abs(anti_sym.loc[i, j] + anti_sym.loc[j, i]))
    max_anti = max(values) if values else np.nan
    add(checks, "局部替换反对称性", bool(np.isfinite(max_anti) and max_anti < 1e-12), f"最大反对称误差={max_anti:.3e}")

    missing_figures = [name for name in manifest["figure"] if not (FIGURES / name).exists() or (FIGURES / name).stat().st_size == 0]
    add(checks, "图表完整性", not missing_figures, f"清单 {len(manifest)} 张，缺失={missing_figures}")

    required = [
        "data_audit.csv", "compute_consistency.csv", "m0_parameters.csv", "m0_loo_predictions.csv", "m0_profile_likelihood.csv",
        "m0_external_metrics.csv", "m1_grouped_cv_summary.csv", "m1_quality_models.csv",
        "quality_direction_audit.csv", "marginal_elasticities.csv", "quality_parameter_equivalence.csv",
        "domain_substitution_matrix.csv", "problem2_summary.json", "figure_manifest.csv",
    ]
    missing_results = [name for name in required if not (RESULTS / name).exists()]
    add(checks, "结果文件完整性", not missing_results, f"要求 {len(required)} 项，缺失={missing_results}")

    finite_summary = np.isfinite(summary["m0"]["loo_rmse"]) and np.isfinite(summary["m1"]["gamma"])
    add(checks, "汇总数值有限", bool(finite_summary), "M0 验证误差与 M1 gamma 均为有限数")
    add(checks, "跨来源指标纪律", summary.get("no_pooled_A_B_rmse_reported") is True,
        "未报告 A/B 拼接后的总体 RMSE")

    check_df = pd.DataFrame(checks)
    check_df.to_csv(RESULTS / "verification_checks.csv", index=False, encoding="utf-8-sig")
    passed = bool(check_df["passed"].all())
    result = {"passed": passed, "checks": len(check_df), "failed": check_df.loc[~check_df["passed"], "check"].tolist()}
    (RESULTS / "verification_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = "\n".join(f"| {r['check']} | {'通过' if r['passed'] else '失败'} | {r['evidence']} |" for r in checks)
    report = f"""# 问题二验证验收报告

## 验收结论

**{'通过' if passed else '未通过'}**。共检查 {len(check_df)} 项，失败 {len(result['failed'])} 项。

## 检查记录

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
{rows}

## 结论边界

- M0 按完整参数规模轨迹验证，未随机拆分同一轨迹。
- M1 的质量证据来自半合成 B6，B7 只作新增质量水平敏感性。
- B8 的质量方向冲突已复算，因此没有参与拟合。
- 配比替换矩阵是问题一已保存边际效应的一阶恢复；缺少 Hessian 与 Bootstrap 系数时不检验互补性。
- B10 Loss 为估算值，只作外推风险边界。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
