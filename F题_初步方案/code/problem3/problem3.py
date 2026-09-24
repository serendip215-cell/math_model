"""Question 3: conditional compute allocation inside the joint B1/B6 support.

Q means B6's synthetic Q_B. The fixed recipe is the problem-one recommended
centroid. No cross-source Q calibration or empirical quality cost is implied.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq, differential_evolution

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后"
P1 = PROJECT / "outputs" / "problem1"
P2 = PROJECT / "outputs" / "problem2" / "results"
ROOT = PROJECT / "outputs" / "problem3"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = PROJECT / "reports" / "问题三" / "结果分析" / "RESULTS_REPORT_PROBLEM3.md"
for folder in (RESULTS, FIGURES, REPORT.parent):
    folder.mkdir(parents=True, exist_ok=True)

ETA = 2e-4
Q0_MAIN = .4
BUDGETS = [1e19, 1e21, 1e22, 1e24]
COST_NAMES = ["exponential", "power", "logarithmic"]


def save(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")
    return frame


def table(frame: pd.DataFrame, digits: int = 4) -> str:
    view = frame.copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}g}")
    return "\n".join(["| " + " | ".join(view.columns) + " |",
                      "| " + " | ".join(["---"] * len(view.columns)) + " |",
                      *("| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy())])


def g(q: float | np.ndarray, kind: str) -> float | np.ndarray:
    q = np.asarray(q)
    if kind == "exponential": result = 1e7 * np.exp(6 * q)
    elif kind == "power": result = 5e9 * q ** 4
    elif kind == "logarithmic": result = 2e9 * np.log1p(10 * q)
    else: raise ValueError(kind)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class Inputs:
    m0: dict[str, float]
    gamma: float
    delta_b: float
    nlo: float
    nhi: float
    dlo: float
    dhi: float
    contexts: tuple[int, ...]
    placement: str = "both"


def load_inputs() -> Inputs:
    b1 = pd.read_csv(DATA / "B_scaling_laws" / "pythia_training_log_existing.csv")
    b6 = pd.read_csv(DATA / "B_scaling_laws" / "supplementary_NQ_experiment.csv")
    c7 = pd.read_csv(DATA / "C_efficiency_evolution" / "model_architecture_metadata.csv")
    recipe = pd.read_csv(P1 / "mixture" / "results" / "recommended_mixture.csv")
    model = json.loads((P2 / "m1_model_selection.json").read_text(encoding="utf-8"))
    if model["selected_model"] != "exp_both":
        raise ValueError("Question-three objective requires the selected exp_both model")
    if len(b1) != 1176 or len(b6) != 360 or len(c7) != 45 or len(recipe) != 17:
        raise ValueError("Input row contracts changed; audit before optimizing")
    if not math.isclose(recipe["recommended_share"].sum(), 1, abs_tol=1e-8):
        raise ValueError("Problem-one recipe does not sum to one")
    contexts = tuple(sorted(c7["max_position_embeddings"].dropna().astype(int).unique()))
    if contexts != (2048, 4096, 8192, 32768, 131072):
        raise ValueError(f"C7 context support changed: {contexts}")
    p = pd.read_csv(P2 / "m0_parameters.csv")
    m0 = dict(zip(p["parameter"], p["estimate"]))
    fit = pd.read_csv(P2 / "m1_quality_models.csv").set_index("model").loc["exp_both"]
    nlo = max(float(b1.N_params_B.min()), float(b6.N_params_B.min()))
    nhi = min(float(b1.N_params_B.max()), float(b6.N_params_B.max()))
    dlo = max(float(b1.D_tokens_B.min()), float(b6.D_tokens_B.min()))
    dhi = min(float(b1.D_tokens_B.max()), 300.0, float(b6.D_tokens_B.max()))
    save(pd.DataFrame([{"source": "B1 measured", "rows": len(b1), "N_min_B": b1.N_params_B.min(),
                        "N_max_B": b1.N_params_B.max(), "D_min_B": b1.D_tokens_B.min(), "D_max_B": b1.D_tokens_B.max()},
                       {"source": "B6 semi_synthetic", "rows": len(b6), "N_min_B": b6.N_params_B.min(),
                        "N_max_B": b6.N_params_B.max(), "D_min_B": b6.D_tokens_B.min(), "D_max_B": b6.D_tokens_B.max()},
                       {"source": "optimization intersection D<=300B", "rows": np.nan,
                        "N_min_B": nlo, "N_max_B": nhi, "D_min_B": dlo, "D_max_B": dhi}]), "input_support_audit.csv")
    save(c7.groupby("max_position_embeddings", as_index=False).agg(models=("model_name", "size"))
         .rename(columns={"max_position_embeddings": "L_ctx"}), "c7_context_support.csv")
    save(recipe[["domain", "recommended_share", "training_mean_share"]], "fixed_recipe.csv")
    if not .1 <= Q0_MAIN <= 1.0:
        raise ValueError("Q_B baseline outside B6 support")
    return Inputs(m0, float(fit["gamma"]), float(fit["delta"]), nlo, nhi, dlo, dhi, contexts)


def cost_21(n_b: float, d_b: float, q: float, q0: float, context: int, kind: str) -> tuple[float, float, float]:
    train = .006 * n_b * d_b
    attention = .001 * ETA * context * n_b * d_b
    quality = d_b * max(0.0, g(q, kind) - g(q0, kind)) / 1e12
    return train, attention, quality


def loss(n_b: float, d_b: float, q: float, data: Inputs) -> float:
    p = data.m0
    x = p["A"] * n_b ** (-p["alpha"])
    y = p["B"] * d_b ** (-p["beta"])
    factor = math.exp(data.gamma * (1 - q))
    if data.placement == "both": response = factor * (x + y)
    elif data.placement == "D": response = x + factor * y
    else: raise ValueError(data.placement)
    return data.delta_b + p["E"] + response


def optimize(budget: float, context: int, kind: str, q0: float, data: Inputs,
             q_grid_count: int = 81, n_grid_count: int = 70) -> dict:
    if budget <= 0 or not (0.1 <= q0 <= 1): raise ValueError("budget/Q0 outside allowed range")
    budget_21 = budget / 1e21
    a = .001 * (6 + ETA * context)

    def at_q(q: float) -> tuple[float, float, float]:
        b = max(0.0, g(q, kind) - g(q0, kind)) / 1e12
        max_n = min(data.nhi, (budget_21 / data.dlo - b) / a)
        if max_n < data.nlo * (1 - 1e-10): return math.inf, math.nan, math.nan
        max_n = max(data.nlo, max_n)

        def value(log_n: float) -> float:
            n = math.exp(log_n)
            d = min(data.dhi, budget_21 / (a * n + b))
            return loss(n, d, q, data) if d >= data.dlo * (1 - 1e-8) else math.inf

        logs = np.linspace(math.log(data.nlo), math.log(max_n), n_grid_count)
        vals = np.array([value(x) for x in logs])
        if not np.isfinite(vals).any(): return math.inf, math.nan, math.nan
        candidates = [(float(vals[k]), float(logs[k])) for k in np.argsort(vals)[:3]]
        for k in np.argsort(vals)[:3]:
            lo, hi = logs[max(0, k - 1)], logs[min(len(logs) - 1, k + 1)]
            if hi > lo:
                res = minimize_scalar(value, bounds=(lo, hi), method="bounded",
                                      options={"xatol": 1e-12})
                candidates.append((float(res.fun), float(res.x)))
        best_value, best_log = min(candidates)
        n = math.exp(best_log)
        d = min(data.dhi, budget_21 / (a * n + b))
        return best_value, n, d

    # The feasibility limit is itself data- and budget-dependent.
    if cost_21(data.nlo, data.dlo, q0, q0, context, kind)[0] + cost_21(
            data.nlo, data.dlo, q0, q0, context, kind)[1] > budget_21 * (1 + 1e-10):
        raise ValueError(f"Budget {budget:g} cannot support the joint N,D lower bounds")
    q_hi = 1.0
    if sum(cost_21(data.nlo, data.dlo, 1.0, q0, context, kind)) > budget_21:
        q_hi = brentq(lambda q: sum(cost_21(data.nlo, data.dlo, q, q0, context, kind)) - budget_21,
                      q0, 1.0)
    q_values = np.linspace(q0, q_hi, q_grid_count)
    evaluated = [(at_q(float(q)), float(q)) for q in q_values]
    candidates = [(v[0], v[1], v[2], q) for v, q in evaluated]
    best_idx = int(np.argmin([row[0] for row in candidates]))
    if 0 < best_idx < len(q_values) - 1:
        lo, hi = q_values[best_idx - 1], q_values[best_idx + 1]
        res = minimize_scalar(lambda q: at_q(float(q))[0], bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10})
        val, n, d = at_q(float(res.x))
        candidates.append((val, n, d, float(res.x)))
    objective, n, d, q = min(candidates)
    train, attention, quality = cost_21(n, d, q, q0, context, kind)
    used = (train + attention + quality) * 1e21
    q_active = "lower" if q - q0 < 1e-4 else ("upper" if 1 - q < 1e-4 else "interior")
    n_lower = n / data.nlo < 1 + 1e-4
    n_upper = n / data.nhi > 1 - 1e-4
    d_lower = d / data.dlo < 1 + 1e-4
    d_upper = d / data.dhi > 1 - 1e-4
    saturated = bool(n_upper and d_upper and q_active == "upper" and used < budget * (1 - 1e-6))
    return {"budget_FLOPs": budget, "L_ctx": context, "quality_cost": kind, "Q0_B": q0,
            "N_star_B": n, "D_star_B": d, "Q_star_B": q, "predicted_loss": objective,
            "C_train_FLOPs": train * 1e21, "C_attention_FLOPs": attention * 1e21,
            "C_quality_FLOPs": quality * 1e21, "C_used_FLOPs": used,
            "C_unused_fraction": max(0.0, 1 - used / budget),
            "train_fraction_of_used": train / max(train + attention + quality, 1e-20),
            "attention_fraction_of_used": attention / max(train + attention + quality, 1e-20),
            "quality_fraction_of_used": quality / max(train + attention + quality, 1e-20),
            "Q_active": q_active, "N_lower_active": n_lower, "N_upper_active": n_upper,
            "D_lower_active": d_lower, "D_upper_active": d_upper,
            "support_saturated": saturated,
            "active_set": f"Q:{q_active}|Nmin:{int(n_lower)}|Nmax:{int(n_upper)}|Dmin:{int(d_lower)}|Dmax:{int(d_upper)}"}


def independent_grid_check(sol: dict, data: Inputs, n_points: int = 180, q_points: int = 90) -> dict:
    budget = sol["budget_FLOPs"] / 1e21
    q0, context, kind = sol["Q0_B"], sol["L_ctx"], sol["quality_cost"]
    ns = np.geomspace(data.nlo, data.nhi, n_points)
    qs = np.linspace(q0, 1, q_points)
    best = math.inf
    for q in qs:
        b = max(0.0, g(q, kind) - g(q0, kind)) / 1e12
        ds = np.minimum(data.dhi, budget / (.001 * (6 + ETA * context) * ns + b))
        ok = ds >= data.dlo
        if not ok.any(): continue
        x = data.m0["A"] * ns[ok] ** (-data.m0["alpha"])
        y = data.m0["B"] * ds[ok] ** (-data.m0["beta"])
        factor = np.exp(data.gamma * (1 - q))
        values = data.delta_b + data.m0["E"] + (factor * (x + y) if data.placement == "both" else x + factor * y)
        best = min(best, float(values.min()))
    return {"budget_FLOPs": sol["budget_FLOPs"], "L_ctx": context, "quality_cost": kind,
            "optimized_loss": sol["predicted_loss"], "independent_grid_best_loss": best,
            "optimized_minus_grid": sol["predicted_loss"] - best,
            "grid_N_points": n_points, "grid_Q_points": q_points}


def global_search_check(sol: dict, data: Inputs) -> dict:
    """Independent two-dimensional global search, with D eliminated analytically."""
    budget = sol["budget_FLOPs"] / 1e21
    q0, context, kind = sol["Q0_B"], int(sol["L_ctx"]), sol["quality_cost"]
    a = .001 * (6 + ETA * context)

    def objective(x: np.ndarray) -> float:
        n, q = math.exp(float(x[0])), float(x[1])
        b = max(0.0, g(q, kind) - g(q0, kind)) / 1e12
        d = min(data.dhi, budget / (a * n + b))
        if d < data.dlo:
            return 1e3 + 1e3 * (data.dlo - d)
        return loss(n, d, q, data)

    result = differential_evolution(objective,
                                    [(math.log(data.nlo), math.log(data.nhi)), (q0, 1.0)],
                                    seed=20260924, popsize=12, maxiter=180, tol=1e-10,
                                    polish=True)
    return {"budget_FLOPs": sol["budget_FLOPs"], "quality_cost": kind,
            "optimized_loss": sol["predicted_loss"], "global_search_loss": result.fun,
            "optimized_minus_global": sol["predicted_loss"] - result.fun,
            "global_search_success": result.success, "global_search_evaluations": result.nfev}


def marginal_kkt_check(sol: dict, data: Inputs) -> dict:
    """Check interior equality and one-sided KKT inequalities at active bounds."""
    n, d, q = sol["N_star_B"], sol["D_star_B"], sol["Q_star_B"]
    kind, context = sol["quality_cost"], int(sol["L_ctx"])
    x = data.m0["A"] * n ** (-data.m0["alpha"])
    y = data.m0["B"] * d ** (-data.m0["beta"])
    f = math.exp(data.gamma * (1 - q))
    if data.placement == "both":
        gains = [f * data.m0["alpha"] * x / n,
                 f * data.m0["beta"] * y / d,
                 data.gamma * f * (x + y)]
    else:
        gains = [data.m0["alpha"] * x / n,
                 f * data.m0["beta"] * y / d,
                 data.gamma * f * y]
    if kind == "exponential": derivative = 6 * g(q, kind)
    elif kind == "power": derivative = 20e9 * q ** 3
    else: derivative = 20e9 / (1 + 10 * q)
    marginal_costs = [.001 * (6 + ETA * context) * d,
                      .001 * (6 + ETA * context) * n +
                      (g(q, kind) - g(sol["Q0_B"], kind)) / 1e12,
                      d * derivative / 1e12]
    efficiencies = np.asarray(gains) / np.asarray(marginal_costs)
    interior = [data.nlo * (1 + 1e-4) < n < data.nhi * (1 - 1e-4),
                data.dlo * (1 + 1e-4) < d < data.dhi * (1 - 1e-4),
                sol["Q0_B"] + 1e-4 < q < 1 - 1e-4]
    active = efficiencies[interior]
    gap = float((active.max() - active.min()) / active.max()) if len(active) >= 2 else math.nan
    budget_active = sol["C_unused_fraction"] < 1e-6
    multiplier = float(active.mean()) if len(active) and budget_active else 0.0
    lower = [bool(sol["N_lower_active"]), bool(sol["D_lower_active"]), sol["Q_active"] == "lower"]
    upper = [bool(sol["N_upper_active"]), bool(sol["D_upper_active"]), sol["Q_active"] == "upper"]
    violations = []
    for k in range(3):
        if interior[k]: violation = abs(efficiencies[k] - multiplier)
        elif lower[k]: violation = max(0.0, efficiencies[k] - multiplier)
        elif upper[k]: violation = max(0.0, multiplier - efficiencies[k])
        else: violation = math.nan
        violations.append(float(violation / max(multiplier, efficiencies[k], 1e-12)))
    return {"budget_FLOPs": sol["budget_FLOPs"], "quality_cost": kind,
            "N_interior": interior[0], "D_interior": interior[1], "Q_interior": interior[2],
            "N_gain_per_C21": efficiencies[0], "D_gain_per_C21": efficiencies[1],
            "Q_gain_per_C21": efficiencies[2], "interior_relative_gap": gap,
            "budget_active": budget_active, "shadow_gain_per_C21": multiplier,
            "N_one_sided_violation": violations[0],
            "D_one_sided_violation": violations[1],
            "Q_one_sided_violation": violations[2],
            "max_one_sided_violation": float(np.nanmax(violations))}


def p1_recipe_contrast() -> pd.DataFrame:
    """A within-A4 1M contrast only; it cannot be added to B6 absolute Loss."""
    recipe = pd.read_csv(P1 / "mixture" / "results" / "recommended_mixture.csv")
    fitted = np.load(P1 / "mixture" / "results" / "selected_response_surface.npz", allow_pickle=False)
    boot = np.load(P1 / "mixture" / "results" / "selected_response_bootstrap.npz", allow_pickle=False)
    if fitted["coefficients"].shape[1] != len(fitted["powers"]):
        raise ValueError("Problem-one response export dimensions changed")
    ordered = recipe.set_index("domain").loc[fitted["domains"]]
    basis, powers = fitted["basis"], fitted["powers"]

    def features(shares: np.ndarray) -> np.ndarray:
        shares = np.maximum(shares, 1e-6)
        shares = shares / shares.sum()
        ilr = np.log(shares) @ basis.T
        return np.prod(ilr[None, :] ** powers, axis=1)

    difference = features(ordered.recommended_share.to_numpy()) - features(
        ordered.training_mean_share.to_numpy())
    fitted_delta = float(difference @ fitted["coefficients"].mean(axis=0))
    replicates = boot["macro_coefficients"] @ difference
    save(pd.DataFrame({"replicate": boot["replicate"],
                       "recommended_minus_training_mean_1m_loss": replicates}),
         "p1_recipe_contrast_bootstrap.csv")
    metrics = pd.read_csv(P1 / "mixture" / "results" / "mixture_model_metrics.csv")
    test_rmse = float(metrics.query("split == 'test_1m' and target == 'macro_average'").rmse.iloc[0])
    return save(pd.DataFrame([{
        "contrast": "recommended_minus_training_mean",
        "A4_1m_fitted_macro_loss_difference": fitted_delta,
        "bootstrap_2p5": float(np.quantile(replicates, .025)),
        "bootstrap_50": float(np.median(replicates)),
        "bootstrap_97p5": float(np.quantile(replicates, .975)),
        "test_1m_macro_rmse": test_rmse,
        "abs_difference_over_test_rmse": abs(fitted_delta) / test_rmse,
        "transferable_to_B6_loss": False
    }]), "p1_recipe_contrast.csv")


def fixed_quality_loss(budget: float, context: int, kind: str, q0: float,
                       q: float, data: Inputs) -> float:
    """Optimize N and D at one observed B6 quality level; keep the same Q0 cost anchor."""
    c = budget / 1e21
    a = .001 * (6 + ETA * context)
    b = max(0.0, g(q, kind) - g(q0, kind)) / 1e12
    nmax = min(data.nhi, (c / data.dlo - b) / a)
    if nmax < data.nlo: return math.inf

    def value(log_n: float) -> float:
        n = math.exp(log_n)
        d = min(data.dhi, c / (a * n + b))
        return loss(n, d, q, data) if d >= data.dlo * (1 - 1e-9) else math.inf

    logs = np.linspace(math.log(data.nlo), math.log(max(data.nlo, nmax)), 150)
    values = np.asarray([value(float(x)) for x in logs])
    k = int(np.argmin(values))
    best = float(values[k])
    lo, hi = logs[max(0, k - 1)], logs[min(len(logs) - 1, k + 1)]
    if hi > lo:
        best = min(best, float(minimize_scalar(value, bounds=(lo, hi), method="bounded",
                                              options={"xatol": 1e-12}).fun))
    return best


def configure_plots() -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def figures(main: pd.DataFrame, path: pd.DataFrame, contexts: pd.DataFrame,
            sensitivity: pd.DataFrame, quality_summary: pd.DataFrame,
            floor_sensitivity: pd.DataFrame) -> pd.DataFrame:
    configure_plots()
    manifest = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, sub in path.groupby("quality_cost"):
        axes[0].plot(sub.budget_FLOPs, sub.Q_star_B, label=name)
        axes[1].plot(sub.budget_FLOPs, sub.N_star_B, label=name)
        axes[2].plot(sub.budget_FLOPs, sub.D_star_B, label=name)
    for ax, ylabel in zip(axes, ["最优 Q_B", "最优 N（十亿）", "最优 D（十亿 Token）"]):
        ax.set_xscale("log"); ax.set_xlabel("预算 C（FLOPs）"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    name = "budget_path.pdf"; fig.savefig(FIGURES / name, bbox_inches="tight"); plt.close(fig)
    manifest.append((name, "budget_path.csv", "支持域内最优配置；高预算平台为支持域饱和"))

    view = main[(main.quality_cost == "exponential") & (main.L_ctx == 2048)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(view)); bottom = np.zeros(len(view))
    for field, label, color in [("train_fraction_of_used", "基础训练", "#2563eb"),
                                ("attention_fraction_of_used", "注意力", "#7c3aed"),
                                ("quality_fraction_of_used", "质量处理", "#f59e0b")]:
        values = view[field].to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    ax.set_xticks(x, [f"10^{int(round(np.log10(v)))}" for v in view.budget_FLOPs])
    ax.set_xlabel("总预算（FLOPs）"); ax.set_ylabel("已使用算力中的份额")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); name = "compute_allocation.pdf"; fig.savefig(FIGURES / name, bbox_inches="tight"); plt.close(fig)
    manifest.append((name, "optimal_allocations.csv", "三部分成本占已使用预算比例；未用预算另见表"))

    context_view = sensitivity[(sensitivity.quality_cost == "exponential") &
                               (sensitivity.budget_FLOPs == 1e21)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].plot(context_view.L_ctx, context_view.N_star_B, marker="o")
    axes[1].plot(context_view.L_ctx, context_view.D_star_B, marker="o")
    axes[2].plot(context_view.L_ctx, context_view.predicted_loss, marker="o")
    for ax, ylabel in zip(axes, ["最优 N（十亿）", "最优 D（十亿 Token）", "条件预测 Loss"]):
        ax.set_xscale("log"); ax.set_xlabel("C7 最大位置容量情景"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
        ax.axvline(30000, color="#dc2626", ls="--", lw=1, label="临界长度 30000")
    axes[0].legend(fontsize=8); fig.tight_layout()
    name = "context_sensitivity.pdf"; fig.savefig(FIGURES / name, bbox_inches="tight"); plt.close(fig)
    manifest.append((name, "context_sensitivity.csv", "以 C7 最大位置容量构造的假设情景；不是实测训练长度"))

    low = quality_summary[quality_summary.budget_FLOPs == 1e19]
    d_floor = floor_sensitivity[floor_sensitivity.floor_type == "D_min_B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    axes[0].bar(low.quality_cost, low.snap_loss_penalty, color="#2563eb")
    axes[0].set_ylabel("离散 Q 档位的预测 Loss 增量")
    axes[0].set_xlabel("质量成本函数")
    axes[0].grid(axis="y", alpha=.2)
    axes[1].plot(d_floor.floor_value_B, d_floor.predicted_loss, marker="o", color="#d97706")
    axes[1].set_xlabel("D 下界（十亿 Token）")
    axes[1].set_ylabel("低预算条件预测 Loss")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    name = "discrete_and_floor_sensitivity.pdf"
    fig.savefig(FIGURES / name, bbox_inches="tight"); plt.close(fig)
    manifest.append((name, "discrete_quality_summary.csv + support_floor_sensitivity.csv",
                     "10^19 FLOPs 的 B6 离散质量档位及观测域内 D 下界敏感性"))
    return save(pd.DataFrame(manifest, columns=["figure", "data_source", "interpretation"]), "figure_manifest.csv")


def main() -> None:
    data = load_inputs()
    context_main = 2048
    base_rows = []
    for budget in BUDGETS:
        for kind in COST_NAMES:
            result = optimize(budget, context_main, kind, Q0_MAIN, data)
            # The baseline truly fixes Q, rather than approximating it with a two-point grid.
            baseline = optimize_fixed_q(budget, context_main, kind, Q0_MAIN, data)
            result["fixed_Q0_loss"] = baseline["predicted_loss"]
            result["loss_gain_vs_fixed_Q0"] = baseline["predicted_loss"] - result["predicted_loss"]
            base_rows.append(result)
    main_df = save(pd.DataFrame(base_rows), "optimal_allocations.csv")
    (RESULTS / "recipe_identifiability_audit.json").write_text(json.dumps({
        "recipe_source": "problem1 recommended_mixture.csv",
        "recipe_role": "exogenous policy scenario only",
        "recipe_enters_loss_formula": False,
        "recipe_enters_cost_formula": False,
        "joint_optimum_over_recipe_identified": False,
        "reason": "No cross-protocol measured L(N,D,Q_B,p) or calibrated Q_A-to-Q_B mapping"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    path_rows = []
    for kind in COST_NAMES:
        for budget in np.geomspace(1e19, 1e24, 46):
            path_rows.append(optimize(float(budget), context_main, kind, Q0_MAIN, data,
                                      q_grid_count=61, n_grid_count=55))
    path = save(pd.DataFrame(path_rows), "budget_path.csv")
    transitions = []
    for kind, sub in path.groupby("quality_cost"):
        sub = sub.sort_values("budget_FLOPs")
        previous = None
        for _, row in sub.iterrows():
            if previous is not None and row.active_set != previous.active_set:
                cursor = float(previous.budget_FLOPs)
                endpoint = float(row.budget_FLOPs)
                from_state = previous.active_set
                for _event in range(5):
                    if from_state == row.active_set or cursor >= endpoint: break
                    lower, upper = cursor, endpoint
                    for _ in range(14):
                        mid = math.sqrt(lower * upper)
                        state = optimize(mid, context_main, kind, Q0_MAIN, data,
                                         q_grid_count=51, n_grid_count=45)["active_set"]
                        if state == from_state: lower = mid
                        else: upper = mid
                    probe = min(endpoint, upper * (1 + 1e-4))
                    new_state = optimize(probe, context_main, kind, Q0_MAIN, data,
                                         q_grid_count=51, n_grid_count=45)["active_set"]
                    if new_state == from_state:
                        new_state = row.active_set  # unresolved near-coincident changes
                    transitions.append({"quality_cost": kind, "budget_lower_FLOPs": lower,
                                        "budget_upper_FLOPs": upper,
                                        "budget_estimate_FLOPs": math.sqrt(lower * upper),
                                        "from_active_set": from_state, "to_active_set": new_state,
                                        "interpretation": "active-set bracket; model/support transition, not real-world technology transition"})
                    cursor = probe
                    from_state = new_state
            previous = row
    transitions_df = save(pd.DataFrame(transitions), "active_set_transitions.csv")

    context_rows = []
    for budget in BUDGETS:
        for ctx in data.contexts:
            for kind in COST_NAMES:
                try:
                    context_rows.append({**optimize(budget, ctx, kind, Q0_MAIN, data,
                                                    q_grid_count=61, n_grid_count=55), "status": "feasible"})
                except ValueError as exc:
                    context_rows.append({"budget_FLOPs": budget, "L_ctx": ctx, "quality_cost": kind,
                                         "Q0_B": Q0_MAIN, "status": "infeasible_lower_support",
                                         "reason": str(exc)})
    contexts = save(pd.DataFrame(context_rows), "context_sensitivity.csv")
    q0_rows = [optimize(budget, context_main, "exponential", q0, data, q_grid_count=61, n_grid_count=55)
               for budget in BUDGETS for q0 in [.2, .4, .6, .8]]
    save(pd.DataFrame(q0_rows), "quality_baseline_sensitivity.csv")

    d_candidate = pd.read_csv(P2 / "m1_quality_models.csv").set_index("model").loc["exp_D"]
    d_data = Inputs(data.m0, float(d_candidate["gamma"]), float(d_candidate["delta"]),
                    data.nlo, data.nhi, data.dlo, data.dhi, data.contexts, "D")
    structure_rows = []
    for budget in BUDGETS[:3]:
        for label, candidate in [("selected_exp_both", data), ("candidate_exp_D", d_data)]:
            structure_rows.append({"model_structure": label, **optimize(
                budget, context_main, "exponential", Q0_MAIN, candidate,
                q_grid_count=61, n_grid_count=55)})
    save(pd.DataFrame(structure_rows), "m1_structure_sensitivity.csv")

    checks = []
    for _, row in pd.concat([main_df, path, contexts[contexts.status == "feasible"]], ignore_index=True).iterrows():
        cost = sum(cost_21(row.N_star_B, row.D_star_B, row.Q_star_B,
                           row.Q0_B, int(row.L_ctx), row.quality_cost)) * 1e21
        checks.append({"budget_FLOPs": row.budget_FLOPs, "L_ctx": row.L_ctx,
                       "quality_cost": row.quality_cost,
                       "relative_budget_violation": max(0.0, cost / row.budget_FLOPs - 1),
                       "N_inside": data.nlo - 1e-8 <= row.N_star_B <= data.nhi + 1e-8,
                       "D_inside": data.dlo - 1e-8 <= row.D_star_B <= data.dhi + 1e-8,
                       "Q_inside": row.Q0_B - 1e-8 <= row.Q_star_B <= 1 + 1e-8,
                       "cost_recalculation_relative_error": abs(cost / row.budget_FLOPs -
                                                                   row.C_used_FLOPs / row.budget_FLOPs)})
    constraint_checks = save(pd.DataFrame(checks), "constraint_checks.csv")
    grid_check = save(pd.DataFrame([independent_grid_check(row, data) for row in base_rows
                                    if row["budget_FLOPs"] != 1e24]), "independent_grid_checks.csv")
    global_check = save(pd.DataFrame([global_search_check(row, data) for row in base_rows]),
                        "global_search_checks.csv")
    kkt_check = save(pd.DataFrame([marginal_kkt_check(row, data) for row in base_rows]),
                     "marginal_kkt_checks.csv")
    recipe_contrast = p1_recipe_contrast()
    b6_levels = sorted(pd.read_csv(DATA / "B_scaling_laws" / "supplementary_NQ_experiment.csv")
                       .Q_score.unique())
    quality_rows = []
    for row in base_rows:
        for q in b6_levels:
            if q < row["Q0_B"] - 1e-10: continue
            candidate = fixed_quality_loss(row["budget_FLOPs"], int(row["L_ctx"]),
                                           row["quality_cost"], row["Q0_B"], float(q), data)
            quality_rows.append({"budget_FLOPs": row["budget_FLOPs"],
                                 "quality_cost": row["quality_cost"], "Q_B_observed_level": q,
                                 "fixed_Q_loss": candidate, "feasible": np.isfinite(candidate)})
    quality_grid = save(pd.DataFrame(quality_rows), "discrete_quality_grid.csv")
    quality_summary_rows = []
    for row in base_rows:
        sub = quality_grid[(quality_grid.budget_FLOPs == row["budget_FLOPs"]) &
                           (quality_grid.quality_cost == row["quality_cost"]) &
                           quality_grid.feasible]
        best = sub.loc[sub.fixed_Q_loss.idxmin()]
        quality_summary_rows.append({"budget_FLOPs": row["budget_FLOPs"],
                                     "quality_cost": row["quality_cost"],
                                     "continuous_Q_star_B": row["Q_star_B"],
                                     "best_observed_Q_B": best.Q_B_observed_level,
                                     "continuous_loss": row["predicted_loss"],
                                     "observed_grid_loss": best.fixed_Q_loss,
                                     "snap_loss_penalty": best.fixed_Q_loss - row["predicted_loss"]})
    quality_summary = save(pd.DataFrame(quality_summary_rows), "discrete_quality_summary.csv")
    floor_rows = []
    for floor_type, values in [("D_min_B", [data.dlo, 12.0, 15.0, 20.0]),
                               ("N_min_B", [data.nlo, .10, .12])]:
        for value in values:
            candidate_data = replace(data, **({"dlo": value} if floor_type == "D_min_B"
                                              else {"nlo": value}))
            try:
                solution = optimize(1e19, context_main, "exponential", Q0_MAIN, candidate_data)
                floor_rows.append({"floor_type": floor_type, "floor_value_B": value,
                                   "status": "feasible", "N_star_B": solution["N_star_B"],
                                   "D_star_B": solution["D_star_B"],
                                   "Q_star_B": solution["Q_star_B"],
                                   "predicted_loss": solution["predicted_loss"]})
            except ValueError:
                floor_rows.append({"floor_type": floor_type, "floor_value_B": value,
                                   "status": "infeasible"})
    floor_sensitivity = save(pd.DataFrame(floor_rows), "support_floor_sensitivity.csv")

    m0boot = pd.read_csv(P2 / "m0_bootstrap_parameters.csv").set_index("replicate")
    m1boot = pd.read_csv(P2 / "m1_bootstrap_parameters.csv").set_index("replicate")
    common = sorted(set(m0boot.index) & set(m1boot.index))
    selected = np.asarray(common)[np.linspace(0, len(common) - 1, min(80, len(common)), dtype=int)]
    boot_rows = []
    for rep in selected:
        row0 = m0boot.loc[rep]
        row1 = m1boot.loc[rep]
        bdata = Inputs({k: float(row0[k]) for k in ["E", "A", "B", "alpha", "beta"]},
                       float(row1["gamma"]), float(row1["delta"]),
                       data.nlo, data.nhi, data.dlo, data.dhi, data.contexts)
        for budget in BUDGETS[:3]:
            out = optimize(budget, context_main, "exponential", Q0_MAIN, bdata,
                           q_grid_count=51, n_grid_count=45)
            boot_rows.append({"replicate": rep, **out})
    boot = save(pd.DataFrame(boot_rows), "conditional_bootstrap_allocations.csv")
    quant = boot.groupby("budget_FLOPs")[["N_star_B", "D_star_B", "Q_star_B", "predicted_loss"]].quantile([.025, .5, .975])
    save(quant.reset_index().rename(columns={"level_1": "quantile"}), "conditional_bootstrap_quantiles.csv")
    manifest = figures(main_df, path, contexts, contexts, quality_summary, floor_sensitivity)
    write_report(data, main_df, path, transitions_df, contexts, constraint_checks,
                 grid_check, global_check, kkt_check, recipe_contrast,
                 quality_summary, floor_sensitivity, boot, manifest)
    summary = {"support": {"N_B": [data.nlo, data.nhi], "D_B": [data.dlo, data.dhi],
                           "Q_B": [Q0_MAIN, 1.0]},
               "C7_contexts": [int(x) for x in data.contexts], "L_crit": 6 / ETA,
               "solutions": len(main_df), "path_points": len(path),
               "max_constraint_violation": float(constraint_checks.relative_budget_violation.max()),
               "max_optimized_minus_grid": float(grid_check.optimized_minus_grid.max()),
               "max_optimized_minus_global": float(global_check.optimized_minus_global.max()),
               "max_interior_kkt_gap": float(kkt_check.interior_relative_gap.max()),
               "max_one_sided_kkt_violation": float(kkt_check.max_one_sided_violation.max()),
               "max_discrete_quality_penalty": float(quality_summary.snap_loss_penalty.max()),
               "conditional_bootstrap_replicates": len(selected),
               "transitions": len(transitions_df), "no_QA_to_QB_calibration": True,
               "C7_is_capacity_not_measured_training_length": True,
               "fixed_recipe_large_scale_transfer_unvalidated": True,
               "recipe_influence_on_numeric_optimum_identified": False}
    (RESULTS / "problem3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def optimize_fixed_q(budget: float, context: int, kind: str, q0: float, data: Inputs) -> dict:
    """Independent one-dimensional baseline optimizer with Q fixed at Q0."""
    budget_21 = budget / 1e21
    a = .001 * (6 + ETA * context)
    max_n = min(data.nhi, budget_21 / (a * data.dlo))
    logs = np.linspace(np.log(data.nlo), np.log(max_n), 150)
    vals = []
    for log_n in logs:
        n = float(np.exp(log_n)); d = min(data.dhi, budget_21 / (a * n))
        vals.append(loss(n, d, q0, data))
    best = int(np.argmin(vals))
    lo, hi = logs[max(0, best - 1)], logs[min(len(logs) - 1, best + 1)]
    candidate = [(vals[best], float(logs[best]))]
    if hi > lo:
        result = minimize_scalar(lambda x: loss(math.exp(x), min(data.dhi, budget_21 /
                             (a * math.exp(x))), q0, data), bounds=(lo, hi), method="bounded")
        candidate.append((float(result.fun), float(result.x)))
    value, log_n = min(candidate)
    return {"predicted_loss": value, "N_star_B": math.exp(log_n)}


def write_report(data: Inputs, main_df: pd.DataFrame, path: pd.DataFrame,
                 transitions: pd.DataFrame, contexts: pd.DataFrame,
                 checks: pd.DataFrame, grid: pd.DataFrame,
                 global_check: pd.DataFrame, kkt_check: pd.DataFrame,
                 recipe_contrast: pd.DataFrame, quality_summary: pd.DataFrame,
                 floor_sensitivity: pd.DataFrame,
                 boot: pd.DataFrame,
                 manifest: pd.DataFrame) -> None:
    focus = main_df[main_df.quality_cost == "exponential"]
    focus_view = focus[["budget_FLOPs", "N_star_B", "D_star_B", "Q_star_B", "predicted_loss",
                        "C_unused_fraction", "loss_gain_vs_fixed_Q0", "active_set", "support_saturated"]]
    costs = main_df[main_df.budget_FLOPs == 1e21][["quality_cost", "N_star_B", "D_star_B",
                                                    "Q_star_B", "predicted_loss", "quality_fraction_of_used"]]
    structure = pd.read_csv(RESULTS / "m1_structure_sensitivity.csv")
    structure_view = structure[["budget_FLOPs", "model_structure", "N_star_B", "D_star_B", "Q_star_B"]]
    context_view = contexts[(contexts.budget_FLOPs == 1e21) &
                            (contexts.quality_cost == "exponential")][["L_ctx", "N_star_B", "D_star_B",
                                                                          "Q_star_B", "predicted_loss"]]
    q0_sensitivity = pd.read_csv(RESULTS / "quality_baseline_sensitivity.csv")
    q0_view = q0_sensitivity[q0_sensitivity.budget_FLOPs == 1e19][["Q0_B", "N_star_B",
                                                                      "D_star_B", "Q_star_B"]]
    recipe_view = pd.read_csv(RESULTS / "fixed_recipe.csv")[["domain", "recommended_share"]]
    transition_view = transitions[["quality_cost", "budget_estimate_FLOPs", "from_active_set", "to_active_set"]]
    q = boot.groupby("budget_FLOPs")[["N_star_B", "D_star_B", "Q_star_B", "predicted_loss"]].quantile([.025, .5, .975])
    q = q.reset_index().rename(columns={"level_1": "quantile"})
    report = fr"""# 问题三计算结果：支持域内的条件算力配置

## 1. 模型与证据边界

本问列出问题一的 17 域推荐质心配比，使用问题二经 B6 分组 CV 选中的 **exp_both** 质量响应。优化变量为 $N,D,Q_B$；$Q_A$ 与 $Q_B$ 未标定，也不把半合成 B6 解释为真实训练质量处理实验。主情景 $Q_{{B,0}}=0.4$ 是 B6 已列出的质量水平，**不是**问题一评分实测值。质量成本是题设参数情景，不是采购报价。**当前 Loss 与成本公式均不含配比 $p$；因此第一问推荐配比是外生政策情景，不影响表中的数值最优解，更不能声称已求出 $N,D,Q_B,p$ 的联合最优。** 推荐配比主要来自第一问较小尺度的配比实验；用于本问较大 $N$ 时也缺少跨尺度联合验证。该不可识别性单列于 `recipe_identifiability_audit.json`。

共同支持域为 $N_B\in[{data.nlo:.6f},{data.nhi:.6f}]$、$D_B\in[{data.dlo:g},{data.dhi:.3f}]$、$Q_B\in[0.4,1]$。其中 B1 是真实训练轨迹，B6 质量关系为半合成；连续最优 $Q_B$ 是模型在 B6 离散质量水平之间的插值。目标值包括 B6 来源截距 $\delta_B$；截距不影响决策。固定配比参见 `fixed_recipe.csv`，输入范围和 C7 窗口分别见 `input_support_audit.csv`、`c7_context_support.csv`。

三种成本按题面附录 B：$g_{{exp}}(Q_B)=10^7e^{{6Q_B}}$、$g_{{pow}}(Q_B)=5\times10^9Q_B^4$、$g_{{log}}(Q_B)=2\times10^9\ln(1+10Q_B)$，单位 FLOPs/Token；总成本为 $6ND+\eta NDL_{{ctx}}+D[g(Q_B)-g(Q_{{B,0}})]_+$，$\eta=2\times10^{{-4}}$。固定的 17 域配比在所有预算情景保持同一份额：

{table(recipe_view, 5)}

## 2. 预算最优解

主成本为题设指数型，外生上下文长度 2048；表中 $10^{{24}}$ 预算若出现未用余额，则是支持域饱和，不代表真实高预算最优策略。

{table(focus_view, 5)}

质量、训练、注意力三种成本均回代为原始 FLOPs；三种质量成本函数互斥。$Q_B$ 固定在基线 0.4 的最优方案作为可行对照；`loss_gain_vs_fixed_Q0` 仅为模型条件改善。

### 三种质量成本敏感性：$C=10^{{21}}$ FLOPs

{table(costs, 5)}

质量处理成本函数在题目中给出，但未由实验测得，故不同成本式的最优 $Q_B$ 应并列报告，不能挑有利于主结论的一种。

### 质量响应结构敏感性

以下仅比较所选 exp_both 与未入选的 exp_D 候选在同一成本下给出的决策；两者来源截距和质量参数不同，**不直接比较预测 Loss 数值**：

{table(structure_view, 5)}

完整数据见 `m1_structure_sensitivity.csv`。低预算两种结构给出的 $Q_B$ 分别约 0.557 与 0.545；中高预算均触及上界。该结论仍受 B6 半合成标尺与题设成本控制。

### 基线质量选择的敏感性：$C=10^{{19}}$ FLOPs

{table(q0_view, 5)}

其他预算的 $Q_{{B,0}}$ 扫描见 `quality_baseline_sensitivity.csv`。低预算下最优质量对基线假设敏感，故不把单一 $Q_{{B,0}}$ 的选择写成确定政策。

### 第一问配比差值与第三问不可识别性

利用第一问保存的 1M 二次 ILR 响应面，在**同一 A4 协议**下比较推荐质心与训练均值，所得差值及按 A4 配比行重抽样的区间如下（差值为推荐减训练均值，负值代表该模型预测 Loss 较低）：

{table(recipe_contrast, 6)}

这是第一问 1M 模型的配比对比，不是第三问的绝对 Loss 增益；A4 与 B6 缺少共同实验及标尺校准，无法把该差值加到 exp_both 目标式，也无法声称推荐配比在本问最优。自举区间仅反映 A4 拟合抽样，不覆盖跨协议和大尺度转移误差。500 个重抽样值见 `p1_recipe_contrast_bootstrap.csv`。

### 连续质量插值与低预算边界敏感性

B6 只有离散 $Q_B$ 水平。将 $Q_B$ 限定为 B6 实际列出的水平、同时重新优化 $N,D$ 后，$10^{{19}}$ FLOPs 的对照如下；`snap_loss_penalty` 是离散方案相对连续插值最优解的模型预测 Loss 增量：

{table(quality_summary[quality_summary.budget_FLOPs == 1e19], 6)}

全部预算及成本式见 `discrete_quality_grid.csv`、`discrete_quality_summary.csv`。这只检验连续插值对方案的影响，不表示实际能买到精确的质量档位。低预算下的 $D_{{min}}$ 与 $N_{{min}}$ 为支持域边界；把下界**向内收紧**后的情景见下表。向外放宽低于观测最小值需要外推，本报告不据此推荐新最优解。

{table(floor_sensitivity, 6)}

## 3. 上下文与结构性转移

题面 $C_{{train}}=6ND$、$C_{{attn}}=\eta NDL_{{ctx}}$ 给出临界值 $L_{{ctx}}^{{crit}}=6/\eta=30,000$。C7 的 `max_position_embeddings` 有 {[int(x) for x in data.contexts]} 五种**最大位置容量**，其中 32768 略高于临界值，131072 只有一条记录。这些数值用来构造假设的上下文情景，**不是模型实际训练序列长度的观测值**，也不能据此验证注意力成本。注意力与基础训练成本比为 $\eta L_{{ctx}}/6$，不依赖 $N,D$；在 32768 情景时约为 1.092。固定 $C=10^{{21}}$、指数质量成本时：

{table(context_view, 5)}

在部分低预算与长窗口组合下，共同的 $N,D$ 下界已不可负担；这类情景在 `context_sensitivity.csv` 标为 `infeasible_lower_support`，不填造最优解。

结构转移预先定义为 $Q_B$ 的下界/内点/上界或 $N,D$ 支持域下界/上界活跃集合变化。预算路径在 `budget_path.csv`，二分后的转折预算如下；完整左右区间在 `active_set_transitions.csv`：

{table(transition_view, 6) if len(transitions) else '本次预算网格内没有活跃集合变化。'}

这些只是**模型与支持域约束**的转折；连续成本份额变化不称质变。高预算进入 $N,D,Q_B$ 全部上界且预算有余的状态属于数据支持域饱和。

## 4. 结果核验与不确定性

共复核 {len(checks)} 个主解、预算路径和上下文情景；最大预算超限比例 {checks.relative_budget_violation.max():.3e}，所有 $N,D,Q_B$ 均在指定范围，逐解记录见 `constraint_checks.csv`。独立 {int(grid.grid_N_points.iloc[0])}×{int(grid.grid_Q_points.iloc[0])} 网格核查的最大“优化 Loss－网格最小 Loss”为 {grid.optimized_minus_grid.max():.3e}；另一种随机种子固定的差分进化全局搜索，对 12 个主解的最大“现有解 Loss－独立搜索 Loss”为 {global_check.optimized_minus_global.max():.3e}。两者均未找到明显更优解，逐解见 `independent_grid_checks.csv` 和 `global_search_checks.csv`。对预算活跃的方案，内点变量的 Loss 边际降低量/FLOP 应相等，下界变量的边际收益不得高于影子价格，上界变量不得低于影子价格；预算松弛时影子价格为零。可比较的内点最大相对差为 {kkt_check.interior_relative_gap.max():.3e}，**含角点单侧必要条件**的最大标准化违反量为 {kkt_check.max_one_sided_violation.max():.3e}，逐项见 `marginal_kkt_checks.csv`。这是必要条件和独立搜索核验，不是全局最优的数学证明。

来自问题二同编号 B1 轨迹/B6 单元重抽样的 {boot.replicate.nunique()} 个条件参数样本，其配置分位数如下（详见 `conditional_bootstrap_allocations.csv`、`conditional_bootstrap_quantiles.csv`）：

{table(q, 5)}

区间只覆盖当前模型与数据重抽样，不包含质量成本参数不确定性、A/B 标尺不一致、真实质量处理价格或更大模型尺度外推。B1 近乎公式化，不能把狭窄区间称作现实训练项目的完整认知不确定性。

## 5. 图表与复现

{table(manifest, 4)}

从仓库根目录运行：

```powershell
python F题_初步方案/code/problem3/problem3.py
python F题_初步方案/code/problem3/verify_problem3.py
```
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
