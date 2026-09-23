"""Generate the curated, paper-facing figures for Problem 1.

This module does not refit a model.  It reads the saved numerical outputs,
checks the columns used by every chart, removes superseded PDFs, and writes a
manifest that records the data source and the interpretation boundary of each
figure.  Run it after the numerical scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
P1 = PROJECT / "outputs" / "problem1"
QUALITY = P1 / "quality"
MIXTURE = P1 / "mixture"
ENHANCEMENT = P1 / "enhancement"
REFINEMENT = P1 / "refinement"

QUALITY_FIG = QUALITY / "figures"
MIXTURE_FIG = MIXTURE / "figures"
ENH_FIG = ENHANCEMENT / "figures"
for folder in (QUALITY_FIG, MIXTURE_FIG, ENH_FIG):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BLUE = "#2563eb"
ORANGE = "#f59e0b"
GREEN = "#16a34a"
RED = "#dc2626"
GRAY = "#6b7280"
LIGHT = "#d1d5db"


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def remove_superseded() -> list[str]:
    old = [
        MIXTURE_FIG / "mixture_prediction_by_scale.pdf",
        MIXTURE_FIG / "mixture_rmse_by_scale.pdf",
        MIXTURE_FIG / "scale_aware_model_comparison.pdf",
        QUALITY_FIG / "conflict_rate_by_domain.pdf",
        ENH_FIG / "conflict_sensitivity.pdf",
        ENH_FIG / "cross_scale_recommendation.pdf",
        ENH_FIG / "quality_vs_macro_loss.pdf",
    ]
    removed = [str(path.relative_to(PROJECT)) for path in old]
    for path in old:
        if path.exists():
            path.unlink()
    return removed


def plot_conflict_profile() -> None:
    summary_path = QUALITY / "results" / "quality_domain_summary.csv"
    cause_path = QUALITY / "results" / "conflict_cause_profile.csv"
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    causes = pd.read_csv(cause_path, encoding="utf-8-sig")
    require_columns(summary, ["dataset", "domain", "conflict_rate"], summary_path)
    require_columns(causes, ["domain", "pair", "share_among_domain_high_conflict"], cause_path)

    a1 = summary.loc[summary["dataset"].eq("A1_sample"), ["domain", "conflict_rate"]]
    a1 = a1.sort_values("conflict_rate", ascending=True).reset_index(drop=True)
    domains = a1["domain"].tolist()
    pairs = sorted(causes["pair"].dropna().unique())
    pivot = (
        causes.pivot_table(
            index="domain", columns="pair", values="share_among_domain_high_conflict", aggfunc="sum", fill_value=0
        )
        .reindex(index=domains, columns=pairs, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [0.85, 1.6]})
    y = np.arange(len(domains))
    axes[0].barh(y, a1["conflict_rate"], color=ORANGE)
    axes[0].set_yticks(y, domains)
    axes[0].set_xlabel("高冲突样本占比")
    axes[0].grid(axis="x", alpha=0.2)

    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, max(len(pairs), 1)))
    left = np.zeros(len(domains))
    for color, pair in zip(colors, pairs):
        values = pivot[pair].to_numpy(dtype=float)
        axes[1].barh(y, values, left=left, label=pair, color=color)
        left += values
    axes[1].set_yticks(y, [])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("高冲突样本中的冲突类型构成")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=8, frameon=False)
    axes[1].grid(axis="x", alpha=0.2)
    save(fig, QUALITY_FIG / "conflict_profile_by_domain.pdf")


def plot_prediction_1m() -> None:
    pred_path = MIXTURE / "results" / "mixture_predictions.csv"
    metrics_path = MIXTURE / "results" / "mixture_model_metrics.csv"
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    require_columns(pred, ["split", "actual_macro_loss", "predicted_macro_loss"], pred_path)
    require_columns(metrics, ["split", "target", "rmse", "spearman"], metrics_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    for ax, split, label in zip(axes, ["train_1m", "test_1m"], ["训练集（1M）", "独立检验集（1M）"]):
        sub = pred[pred["split"].eq(split)]
        row = metrics[(metrics["split"].eq(split)) & (metrics["target"].eq("macro_average"))].iloc[0]
        actual = sub["actual_macro_loss"].to_numpy(float)
        predicted = sub["predicted_macro_loss"].to_numpy(float)
        r2 = 1.0 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        ax.scatter(actual, predicted, s=14, alpha=0.55, color=BLUE, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=GRAY, linewidth=1)
        ax.text(
            0.04,
            0.96,
            f"{label}\nRMSE={row['rmse']:.3f}\nSpearman={row['spearman']:.3f}\n$R^2$={r2:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "alpha": 0.9},
        )
        ax.set_xlabel("实际平均 Loss")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("预测平均 Loss")
    save(fig, MIXTURE_FIG / "mixture_prediction_1m.pdf")


def plot_quality_ablation() -> None:
    path = MIXTURE / "results" / "quality_integration_ablation.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["model", "cv_mse", "test_1m_macro_rmse", "selected"], path)
    labels = {
        "linear_ilr_ridge": "线性 ILR",
        "quadratic_ilr_ridge": "二阶 ILR",
        "quadratic_ilr_ridge_quality": "二阶 ILR + Q\n（可观测映射 6/17）",
    }
    names = [labels.get(x, x) for x in df["model"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    axes[0].bar(x, df["cv_mse"], color=[GRAY, BLUE, ORANGE])
    axes[0].set_ylabel("训练集五折 CV MSE")
    axes[1].bar(x, df["test_1m_macro_rmse"], color=[GRAY, BLUE, ORANGE])
    axes[1].set_ylabel("独立 1M 检验集平均 Loss RMSE")
    for ax in axes:
        ax.set_xticks(x, names)
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", alpha=0.2)
    selected = np.where(df["selected"].astype(bool).to_numpy())[0]
    for idx in selected:
        axes[1].text(idx, df.iloc[idx]["test_1m_macro_rmse"] + 0.002, "最终采用", ha="center", fontsize=8)
    save(fig, MIXTURE_FIG / "quality_integration_ablation.pdf")


def plot_cross_scale_transfer() -> None:
    main_path = MIXTURE / "results" / "mixture_model_metrics.csv"
    scale_path = MIXTURE / "results" / "scale_aware_metrics.csv"
    main = pd.read_csv(main_path, encoding="utf-8-sig")
    scale = pd.read_csv(scale_path, encoding="utf-8-sig")
    require_columns(main, ["split", "target", "centered_rmse", "spearman"], main_path)
    require_columns(scale, ["evaluation", "centered_rmse", "spearman"], scale_path)
    splits = ["test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
    labels = ["1M", "60M", "1B", "10B\n估计", "70B\n估计"]
    m = main[main["target"].eq("macro_average")].set_index("split").loc[splits]
    s = scale.set_index("evaluation").loc[splits]
    x = np.arange(len(splits))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(x, m["centered_rmse"], "o-", color=GRAY, label="1M 主响应面直接迁移")
    axes[0].plot(x, s["centered_rmse"], "o-", color=BLUE, label="多尺度补充模型（外部/留一尺度）")
    axes[0].set_ylabel("去均值平均 Loss RMSE")
    axes[1].plot(x, m["spearman"], "o-", color=GRAY, label="1M 主响应面直接迁移")
    axes[1].plot(x, s["spearman"], "o-", color=BLUE, label="多尺度补充模型（外部/留一尺度）")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("Spearman ρ")
    for ax in axes:
        ax.axvspan(2.5, 4.5, color=ORANGE, alpha=0.08)
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    save(fig, MIXTURE_FIG / "cross_scale_transfer.pdf")


def plot_recommendation() -> None:
    rec_path = MIXTURE / "results" / "recommended_mixture.csv"
    stable_path = MIXTURE / "results" / "recommendation_stability.csv"
    rec = pd.read_csv(rec_path, encoding="utf-8-sig")
    stable = pd.read_csv(stable_path, encoding="utf-8-sig")
    require_columns(rec, ["domain", "recommended_share", "model_optimal_share", "training_mean_share"], rec_path)
    require_columns(stable, ["domain", "bootstrap_ci_low", "bootstrap_ci_high"], stable_path)
    df = rec.merge(stable[["domain", "bootstrap_ci_low", "bootstrap_ci_high"]], on="domain", validate="one_to_one")
    df = df.sort_values("recommended_share", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    xerr = np.vstack(
        [df["recommended_share"] - df["bootstrap_ci_low"], df["bootstrap_ci_high"] - df["recommended_share"]]
    )
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    ax.errorbar(df["recommended_share"], y, xerr=xerr, fmt="o", color=BLUE, ecolor=BLUE, capsize=2.5,
                label="实测最优前 10% 质心及 Bootstrap 95% 区间")
    ax.scatter(df["training_mean_share"], y, marker="|", s=85, color=GRAY, label="训练配比均值")
    ax.scatter(df["model_optimal_share"], y, marker="^", s=30, facecolors="none", edgecolors=ORANGE,
               label="响应面诊断解（不作为推荐）")
    ax.set_yticks(y, df["domain"])
    ax.set_xlabel("配比")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, MIXTURE_FIG / "recommended_mixture.pdf")


def plot_marginal_effects() -> None:
    path = MIXTURE / "results" / "mixture_marginal_effects.csv"
    df = pd.read_csv(path, encoding="utf-8-sig").sort_values("delta_macro_loss")
    require_columns(df, ["domain", "delta_share", "delta_macro_loss"], path)
    colors = np.where(df["delta_macro_loss"] <= 0, GREEN, RED)
    fig, ax = plt.subplots(figsize=(8.3, 6.1))
    y = np.arange(len(df))
    ax.barh(y, df["delta_macro_loss"], color=colors)
    ax.set_yticks(y, df["domain"])
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("训练均值附近单域增加 1 个百分点时的模型预测平均 Loss 变化")
    ax.grid(axis="x", alpha=0.2)
    save(fig, MIXTURE_FIG / "mixture_marginal_effects.pdf")


def plot_observed_scale_centroids() -> None:
    path = ENHANCEMENT / "results" / "cross_scale_recommendation.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["scale", "domain", "centroid"], path)
    scales = ["train_1m", "test_60m", "test_1B"]
    labels = ["1M 训练样本", "60M 检验样本", "1B 检验样本"]
    pivot = df[df["scale"].isin(scales)].pivot(index="domain", columns="scale", values="centroid").loc[:, scales]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.9, 7.0))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=float(pivot.to_numpy().max()))
    ax.set_xticks(np.arange(len(scales)), labels)
    ax.set_yticks(np.arange(len(pivot)), pivot.index)
    ax.set_xlabel("每个尺度样本内按实测 Loss 选取最优前 10% 后取质心")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("质心占比")
    save(fig, ENH_FIG / "empirical_centroid_by_observed_scale.pdf")


def plot_domain_ablation() -> None:
    path = ENHANCEMENT / "results" / "domain_ablation.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["removed", "delta_rmse_pct", "delta_spearman"], path)
    df = df[~df["removed"].str.contains("baseline", na=False)].copy()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].bar(x, df["delta_rmse_pct"], color=np.where(df["delta_rmse_pct"] >= 0, RED, GREEN))
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("移除后标准化目标 RMSE 变化（%）")
    axes[1].bar(x, df["delta_spearman"], color=np.where(df["delta_spearman"] >= 0, GREEN, RED))
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("移除后 Spearman 变化")
    for ax in axes:
        ax.set_xticks(x, df["removed"], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2)
    save(fig, ENH_FIG / "domain_ablation.pdf")


def plot_effect_decay_check() -> None:
    path = P1 / "scaling_bridge" / "effect_variant_results.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["scale", "variant", "macro_rmse", "macro_spearman"], path)
    scales = ["est_10b", "est_70b"]
    variants = ["baseline_only", "frozen_1B", "extrap_trend", "decay_fusion"]
    names = ["纯规模基线", "冻结 1B 效应", "趋势外推", "置信衰减"]
    colors = [LIGHT, GRAY, ORANGE, BLUE]
    x = np.arange(len(scales))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    for i, (variant, name, color) in enumerate(zip(variants, names, colors)):
        sub = df[df["variant"].eq(variant)].set_index("scale").reindex(scales)
        offset = (i - 1.5) * width
        axes[0].bar(x + offset, sub["macro_rmse"], width, color=color, label=name)
        if sub["macro_spearman"].notna().any():
            axes[1].bar(x + offset, sub["macro_spearman"], width, color=color, label=name)
    axes[0].set_ylabel("平均 Loss RMSE")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    for ax in axes:
        ax.set_xticks(x, ["10B 估计表", "70B 估计表"])
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    save(fig, ENH_FIG / "estimated_scale_effect_decay_check.pdf")


def plot_regularization_path() -> None:
    path = REFINEMENT / "regularization_path.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["lam", "dist_to_centroid", "max_share", "pred_macro_original", "nearest_train_actual_loss"], path)
    labels = ["∞" if np.isinf(v) else f"{v:g}" for v in df["lam"].to_numpy(float)]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axes[0].plot(x, df["pred_macro_original"], "o-", color=ORANGE, label="响应面预测 Loss")
    axes[0].plot(x, df["nearest_train_actual_loss"], "o-", color=BLUE, label="最近实测配方 Loss")
    axes[0].set_ylabel("平均 Loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, df["max_share"], "o-", color=BLUE, label="最大单域占比")
    axes[1].set_ylabel("最大单域占比")
    ax2 = axes[1].twinx()
    ax2.plot(x, df["dist_to_centroid"], "s--", color=GRAY, label="到推荐质心的 ILR 距离")
    ax2.set_ylabel("ILR 距离")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25)
        ax.set_xlabel("质心惩罚系数 λ")
        ax.grid(alpha=0.2)
    save(fig, ENH_FIG / "recommendation_regularization_path.pdf")


def write_manifest(removed: list[str]) -> None:
    rows = [
        ("quality/figures/quality_by_domain.pdf", "quality_domain_summary.csv", "领域质量中位数及 Bootstrap 区间"),
        ("quality/figures/quality_score_distribution.pdf", "quality_sample_scores.csv", "同域扩展样本与 A1 分布比较"),
        ("quality/figures/quality_group_correlation.pdf", "quality_sample_scores.csv", "四个语义组的相关结构"),
        ("quality/figures/conflict_profile_by_domain.pdf", "quality_domain_summary.csv; conflict_cause_profile.csv", "冲突率与冲突类型构成；结构解释而非语义因果"),
        ("mixture/figures/mixture_prediction_1m.pdf", "mixture_predictions.csv; mixture_model_metrics.csv", "仅展示训练尺度和独立 1M 检验，R²按原始方差计算"),
        ("mixture/figures/quality_integration_ablation.pdf", "quality_integration_ablation.csv", "只使用 A16 可观测映射的 6/17 域 Q；最终未采用"),
        ("mixture/figures/cross_scale_transfer.pdf", "mixture_model_metrics.csv; scale_aware_metrics.csv", "去均值误差与排序；10B/70B 为估计压力测试"),
        ("mixture/figures/recommended_mixture.pdf", "recommended_mixture.csv; recommendation_stability.csv", "经验质心、Bootstrap 区间及响应面诊断解"),
        ("mixture/figures/mixture_marginal_effects.pdf", "mixture_marginal_effects.csv", "训练均值附近的模型局部预测，不作全局因果解释"),
        ("enhancement/figures/empirical_centroid_by_observed_scale.pdf", "cross_scale_recommendation.csv", "仅展示 1M/60M/1B 样本内经验质心，不含估计尺度"),
        ("enhancement/figures/domain_ablation.pdf", "domain_ablation.csv", "输入域移除敏感性；纵轴为标准化目标口径"),
        ("enhancement/figures/estimated_scale_effect_decay_check.pdf", "effect_variant_results.csv", "估计表诊断；纯规模基线 RMSE 更低且含配比变体排序接近零"),
        ("enhancement/figures/recommendation_regularization_path.pdf", "regularization_path.csv", "响应面外推与最近实测邻域的对照"),
    ]
    manifest = pd.DataFrame(rows, columns=["figure", "data_source", "interpretation_boundary"])
    manifest.to_csv(P1 / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    (P1 / "removed_figure_list.txt").write_text("\n".join(removed) + ("\n" if removed else ""), encoding="utf-8")


def main() -> None:
    removed = remove_superseded()
    plot_conflict_profile()
    plot_prediction_1m()
    plot_quality_ablation()
    plot_cross_scale_transfer()
    plot_recommendation()
    plot_marginal_effects()
    plot_observed_scale_centroids()
    plot_domain_ablation()
    plot_effect_decay_check()
    plot_regularization_path()
    write_manifest(removed)
    print(f"generated curated figures; removed {len(removed)} superseded PDFs")


if __name__ == "__main__":
    main()
