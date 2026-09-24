"""Independent numerical and evidence-boundary checks for Question 3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem3 import DATA, P1, P2, RESULTS, FIGURES, PROJECT, ETA, cost_21, g, load_inputs, loss

REPORT = PROJECT / "reports" / "问题三" / "验证验收" / "VERIFY_REPORT_PROBLEM3.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    checks: list[dict] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    data = load_inputs()
    summary = json.loads((RESULTS / "problem3_summary.json").read_text(encoding="utf-8"))
    main = pd.read_csv(RESULTS / "optimal_allocations.csv")
    path = pd.read_csv(RESULTS / "budget_path.csv")
    contexts = pd.read_csv(RESULTS / "context_sensitivity.csv")
    transitions = pd.read_csv(RESULTS / "active_set_transitions.csv")
    boot = pd.read_csv(RESULTS / "conditional_bootstrap_allocations.csv")
    grid = pd.read_csv(RESULTS / "independent_grid_checks.csv")
    global_check = pd.read_csv(RESULTS / "global_search_checks.csv")
    kkt = pd.read_csv(RESULTS / "marginal_kkt_checks.csv")
    structure = pd.read_csv(RESULTS / "m1_structure_sensitivity.csv")
    manifest = pd.read_csv(RESULTS / "figure_manifest.csv")
    recipe = pd.read_csv(RESULTS / "fixed_recipe.csv")
    recipe_audit = json.loads((RESULTS / "recipe_identifiability_audit.json").read_text(encoding="utf-8"))

    add("输入合同与支持域", data.nlo < data.nhi and data.dlo < data.dhi and
        summary["support"]["N_B"] == [data.nlo, data.nhi] and
        summary["support"]["D_B"] == [data.dlo, data.dhi],
        f"N_B=[{data.nlo:.6f},{data.nhi:.6f}], D_B=[{data.dlo:g},{data.dhi:.3f}]")
    add("固定配比单纯形", len(recipe) == 17 and abs(recipe.recommended_share.sum() - 1) < 1e-9,
        f"17 域份额和={recipe.recommended_share.sum():.12f}")
    add("跨问质量标尺边界", summary["no_QA_to_QB_calibration"] is True,
        "优化变量仅为 B6 原生 Q_B；不将问题一 Q_A 当作 Q_B")
    add("C7 离散窗口与临界长度", summary["C7_contexts"] == [2048, 4096, 8192, 32768, 131072]
        and abs(summary["L_crit"] - 30000) < 1e-9 and 8192 < 30000 < 32768,
        "Lcrit=6/eta=30000，处于 C7 的 8192 与 32768 之间")
    add("C7 最大容量证据边界", summary["C7_is_capacity_not_measured_training_length"] is True,
        "max_position_embeddings 仅作为假设情景，未当作实测训练序列长度")
    add("推荐配比跨尺度证据边界", summary["fixed_recipe_large_scale_transfer_unvalidated"] is True,
        "问题一较小尺度质心在本问大 N 下只作固定政策情景")
    add("配比未进入数值目标", recipe_audit["recipe_enters_loss_formula"] is False and
        recipe_audit["recipe_enters_cost_formula"] is False and
        recipe_audit["joint_optimum_over_recipe_identified"] is False and
        summary["recipe_influence_on_numeric_optimum_identified"] is False,
        "p 仅列为外生政策情景，未声称求得 N,D,Q_B,p 的联合最优")
    add("质量成本单调性", all(np.all(np.diff(g(np.linspace(.1, 1, 50), kind)) > 0)
                             for kind in ["exponential", "power", "logarithmic"]),
        "三种题设成本在 Q_B 支持域内严格递增")
    n, d, q, q0, L = 1.7, 42.0, .7, .4, 8192
    train, attn, quality = cost_21(n, d, q, q0, L, "exponential")
    raw_n, raw_d = n * 1e9, d * 1e9
    add("FLOPs 单位回代", max(abs(train * 1e21 - 6 * raw_n * raw_d),
                             abs(attn * 1e21 - ETA * raw_n * raw_d * L),
                             abs(quality * 1e21 - raw_d * (g(q, "exponential") - g(q0, "exponential")))) /
        (6 * raw_n * raw_d) < 1e-12, "十亿单位换算与原始 FLOPs 公式一致")

    errors = []
    losses = []
    bounds = []
    baselines = []
    for row in main.itertuples():
        components = cost_21(row.N_star_B, row.D_star_B, row.Q_star_B,
                             row.Q0_B, int(row.L_ctx), row.quality_cost)
        errors.append(abs(sum(components) * 1e21 - row.C_used_FLOPs) / row.budget_FLOPs)
        losses.append(abs(loss(row.N_star_B, row.D_star_B, row.Q_star_B, data) - row.predicted_loss))
        bounds.append(data.nlo - 1e-8 <= row.N_star_B <= data.nhi + 1e-8 and
                      data.dlo - 1e-8 <= row.D_star_B <= data.dhi + 1e-8 and
                      row.Q0_B - 1e-8 <= row.Q_star_B <= 1 + 1e-8 and
                      row.C_used_FLOPs <= row.budget_FLOPs * (1 + 1e-9))
        baselines.append(row.loss_gain_vs_fixed_Q0 >= -1e-7)
    add("成本回代", max(errors) < 1e-12, f"最大预算标准化差={max(errors):.3e}")
    add("Loss 回代", max(losses) < 1e-10, f"最大绝对差={max(losses):.3e}")
    add("可行性与支持域", all(bounds), f"主情景 {len(bounds)} 解均可行")
    add("不劣于固定基线 Q0", all(baselines),
        f"最小模型内改善={main.loss_gain_vs_fixed_Q0.min():.3e}")
    selected = structure[structure.model_structure == "selected_exp_both"].set_index("budget_FLOPs")
    reference = main[main.quality_cost == "exponential"].set_index("budget_FLOPs").loc[selected.index]
    add("模型结构敏感性口径", len(structure) == 6 and
        set(structure.model_structure) == {"selected_exp_both", "candidate_exp_D"} and
        np.allclose(selected["Q_star_B"], reference["Q_star_B"], atol=1e-5),
        "exp_D 仅作候选结构对照；主模型结果可回代")
    add("独立网格未找到更优解", grid.optimized_minus_grid.max() <= 1e-5,
        f"优化减密集网格 Loss 最大={grid.optimized_minus_grid.max():.3e}")
    add("差分进化未找到更优解", len(global_check) == len(main) and
        global_check.optimized_minus_global.max() < 1e-5,
        f"优化减独立全局搜索 Loss 最大={global_check.optimized_minus_global.max():.3e}")
    kkt_comparable = kkt.interior_relative_gap.dropna()
    add("内点边际条件", len(kkt_comparable) >= 3 and kkt_comparable.max() < 1e-3,
        f"{len(kkt_comparable)} 个可比较主解的最大相对差={kkt_comparable.max():.3e}")

    infeasible = contexts[contexts.status == "infeasible_lower_support"]
    min_costs = [sum(cost_21(data.nlo, data.dlo, row.Q0_B, row.Q0_B,
                             int(row.L_ctx), row.quality_cost)) * 1e21 for row in infeasible.itertuples()]
    add("长窗口低预算不可行性", len(infeasible) == 3 and all(
        c > b for c, b in zip(min_costs, infeasible.budget_FLOPs)),
        f"{len(infeasible)} 个 C7 情景连 N,D 下界都不可负担")

    high = main[main.budget_FLOPs == 1e24]
    add("高预算仅为支持域饱和", len(high) == 3 and high.support_saturated.astype(bool).all()
        and (high.C_unused_fraction > .9).all(),
        f"10^24 FLOPs 未用份额 {high.C_unused_fraction.min():.2%}–{high.C_unused_fraction.max():.2%}")
    add("活跃集合转折口径", len(transitions) > 0 and
        (transitions.budget_upper_FLOPs > transitions.budget_lower_FLOPs).all() and
        ((transitions.budget_upper_FLOPs / transitions.budget_lower_FLOPs) < 1.001).all() and
        (transitions.from_active_set != transitions.to_active_set).all(),
        f"{len(transitions)} 个转折区间经二分细化，均仅作模型内解释")
    common = set(pd.read_csv(P2 / "m0_bootstrap_parameters.csv").replicate) & set(
        pd.read_csv(P2 / "m1_bootstrap_parameters.csv").replicate)
    add("联合编号重抽样", boot.replicate.nunique() == 80 and set(boot.replicate).issubset(common)
        and boot.groupby("replicate").size().eq(3).all(),
        "80 个 B1/B6 同编号参数对，各计算三个预算")
    missing = [x for x in manifest.figure if not (FIGURES / x).exists() or (FIGURES / x).stat().st_size == 0]
    add("图表完整性", not missing, f"{len(manifest)} 张 PDF，缺失={missing}")

    result = pd.DataFrame(checks)
    result.to_csv(RESULTS / "verification_checks.csv", index=False, encoding="utf-8-sig")
    passed = bool(result.passed.all())
    (RESULTS / "verification_summary.json").write_text(json.dumps({
        "passed": passed, "checks": len(result),
        "failed": result.loc[~result.passed, "check"].tolist()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = "\n".join(f"| {r.check} | {'通过' if r.passed else '失败'} | {r.evidence} |"
                      for r in result.itertuples())
    REPORT.write_text(f"""# 问题三独立核验

**{'通过' if passed else '未通过'}**：{int(result.passed.sum())}/{len(result)} 项。

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
{lines}

## 解释范围

预算配置来自 B1 真实训练轨迹的规模响应与 B6 半合成质量响应；质量成本由题目附录 B 假设。第一问推荐配比仅列为外生情景，未进入第三问目标式，不能声称求得配比联合最优；较大尺度下是否仍最优也未验证。$Q_A$ 与 $Q_B$ 未校准。C7 的最大位置容量只是上下文假设情景，不是实际训练长度。高预算未用余额反映数据支持域饱和。Bootstrap 仅是当前模型内参数重抽样，不覆盖真实成本或跨数据源协议误差。
""", encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": len(result),
                      "failed": result.loc[~result.passed, "check"].tolist()}, ensure_ascii=False))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
