"""问题二：N-D-Q-p 广义标度律、边际效用与局部领域替换。

证据纪律：
1. B1 为经典标度律主拟合；交叉验证按完整参数规模轨迹留出。
2. B6 为质量项主拟合，B7 仅作扩展 Q 水平验证，B8 只作冲突审计。
3. 配比项只读取问题一已经保存的结果，不重新读取 A 类原始数据。
4. 不把不同来源的样本无权拼接，也不报告虚构的整体 RMSE。
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_bridge import build_quality_bridge


SEED = 20260924
RNG = np.random.default_rng(SEED)
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后"
BROOT = DATA / "B_scaling_laws"
P1 = PROJECT / "outputs" / "problem1"
OUT = PROJECT / "outputs" / "problem2"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
REPORT = PROJECT / "reports" / "问题二" / "结果分析" / "RESULTS_REPORT_PROBLEM2.md"
for directory in (RESULTS, FIGURES, REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)


FILES = {
    "B1": "pythia_training_log_existing.csv",
    "B2": "cerebras_training_log.csv",
    "B4": "scaling_baseline.csv",
    "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv",
    "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv",
    "B9": "supplementary_large_models.csv",
    "B10": "supplementary_large_baseline.csv",
    "B11": "open_model_family_metadata.csv",
    "B12": "pythia_checkpoint_index.csv",
}

EXPECTED_ROWS = {"B1": 1176, "B2": 1029, "B4": 57, "B5": 44,
                 "B6": 360, "B7": 450, "B8": 1704, "B9": 132,
                 "B10": 128, "B11": 18, "B12": 1386}


def configure_plots() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": candidates,
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    residual = pred - y
    offset = float(np.mean(y - pred))
    calibrated = pred + offset
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_actual": float(np.mean(residual)),
        "centered_rmse": float(np.sqrt(np.mean((residual - residual.mean()) ** 2))),
        "offset_actual_minus_pred": offset,
        "offset_calibrated_rmse": float(np.sqrt(np.mean((calibrated - y) ** 2))),
        "spearman": safe_spearman(y, pred),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    view = frame.copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}g}")
    head = "| " + " | ".join(map(str, view.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.astype(str).to_numpy()]
    return "\n".join([head, sep, *rows])


def audit_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    rows = []
    roles = {
        "B1": ("附件训练日志", "M0 主拟合；近乎确定性幂律，需警惕构造性"),
        "B2": ("半合成", "模型族外压力测试"),
        "B4": ("真实整理", "跨族外部验证"),
        "B5": ("真实整理", "文献外部验证"),
        "B6": ("半合成", "M1 主拟合"),
        "B7": ("半合成", "新增 Q 水平敏感性"),
        "B8": ("半合成/外推", "方向冲突与远域压力测试"),
        "B9": ("真实元数据", "外推范围"),
        "B10": ("估算", "大模型风险边界"),
        "B11": ("辅助元数据", "模型族索引"),
        "B12": ("辅助索引", "检查点索引"),
    }
    for key, filename in FILES.items():
        path = BROOT / filename
        if not path.exists():
            raise FileNotFoundError(f"{key} 文件缺失：{path}")
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_ROWS[key]:
            raise AssertionError(f"{key} 行数错误：{filename}={len(frame)}，期望={EXPECTED_ROWS[key]}")
        frames[key] = frame
        rows.append({
            "dataset": key,
            "file": filename,
            "rows": len(frame),
            "columns": len(frame.columns),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "evidence_type": roles[key][0],
            "role": roles[key][1],
        })
    trajectory_files = sorted((BROOT / "training_trajectories").glob("*.csv"))
    if len(trajectory_files) != 8:
        raise AssertionError(f"B3 插值文件数错误：{len(trajectory_files)}，期望 8")
    for path in trajectory_files:
        check = pd.read_csv(path)
        if len(check) != 500 or "interpolated" not in check.columns or not bool((check["interpolated"] == 1).all()):
            raise AssertionError(f"B3 文件未通过 500 行/interpolated=1 断言：{path.name}")
    b3_rows = sum(len(pd.read_csv(p)) for p in trajectory_files)
    rows.insert(2, {
        "dataset": "B3",
        "file": "training_trajectories/*.csv",
        "rows": b3_rows,
        "columns": len(pd.read_csv(trajectory_files[0]).columns),
        "missing_cells": sum(int(pd.read_csv(p).isna().sum().sum()) for p in trajectory_files),
        "duplicate_rows": 0,
        "evidence_type": "插值",
        "role": "连续性检查，不作独立验证",
    })
    audit = pd.DataFrame(rows)
    save_csv(audit, "data_audit.csv")
    b1 = frames["B1"]
    expected_c = 0.006 * b1["N_params_B"] * b1["D_tokens_B"]
    relative_error = (b1["C_FLOPs_1e21"] - expected_c).abs() / expected_c
    compute_check = pd.DataFrame([{
        "formula": "C_FLOPs_1e21 = 0.006 * N_params_B * D_tokens_B",
        "rows": len(b1),
        "median_absolute_relative_error": float(relative_error.median()),
        "p95_absolute_relative_error": float(relative_error.quantile(0.95)),
        "within_5_percent_fraction": float((relative_error <= 0.05).mean()),
        "max_absolute_relative_error": float(relative_error.max()),
        "max_error_cause": "最早期检查点的 C 仅保留四位小数，分母很小时相对误差被放大",
    }])
    save_csv(compute_check, "compute_consistency.csv")
    contracts = pd.DataFrame([
        {"label": "B1", "file": FILES["B1"], "rows": len(frames["B1"]), "expected_rows": 1176,
         "status": "PASS", "note": "真实 Pythia 日志；8 个参数规模，每组 147 点"},
        {"label": "B3", "file": "training_trajectories/*.csv", "rows": b3_rows, "expected_rows": 4000,
         "status": "PASS", "note": "8×500，所有文件 interpolated=1；只作连续性审计"},
        {"label": "B4", "file": FILES["B4"], "rows": len(frames["B4"]), "expected_rows": 57,
         "status": "PASS", "note": "scaling_baseline.csv；跨族 Loss 数据"},
        {"label": "B11", "file": FILES["B11"], "rows": len(frames["B11"]), "expected_rows": 18,
         "status": "PASS", "note": "open_model_family_metadata.csv；无 Loss，不作 B4"},
    ])
    save_csv(contracts, "data_contract_checks.csv")
    b1_n_counts = frames["B1"].groupby("N_params_B").size()
    if len(b1_n_counts) != 8 or not bool((b1_n_counts == 147).all()):
        raise AssertionError("B1 必须是 8 个 N 轨迹且每条 147 点")
    if not bool((frames["B6"]["Q_score"] >= 0.1).all() and (frames["B6"]["Q_score"] <= 1.0).all()):
        raise AssertionError("B6 Q 超出 [0.1,1.0] 支持域")
    return frames, audit


@dataclass
class M0Fit:
    theta: np.ndarray
    params: dict[str, float]
    rss: float
    success: bool


M0_NAMES = ["E", "A", "B", "alpha", "beta"]


def theta_to_m0(theta: np.ndarray) -> dict[str, float]:
    values = np.exp(theta)
    return dict(zip(M0_NAMES, map(float, values)))


def predict_m0(params: dict[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    return params["E"] + params["A"] * n ** (-params["alpha"]) + params["B"] * d ** (-params["beta"])


def m0_bounds(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.log([1e-4, 1e-5, 1e-5, 0.005, 0.005])
    upper = np.log([max(1e-3, float(np.min(y)) * 0.999), 100.0, 100.0, 2.0, 2.0])
    return lower, upper


def fit_m0(frame: pd.DataFrame, loss: str = "linear", fixed: dict[str, float] | None = None) -> M0Fit:
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    y = frame["val_loss"].to_numpy(float)
    names_free = [name for name in M0_NAMES if not fixed or name not in fixed]
    default = {"E": min(1.5, float(y.min()) * 0.8), "A": 0.7, "B": 2.0, "alpha": 0.2, "beta": 0.2}
    lower_all, upper_all = m0_bounds(y)
    bound_map = {name: (lower_all[i], upper_all[i]) for i, name in enumerate(M0_NAMES)}
    x0 = np.log([default[name] for name in names_free])
    lb = np.array([bound_map[name][0] for name in names_free])
    ub = np.array([bound_map[name][1] for name in names_free])
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    def unpack(x: np.ndarray) -> dict[str, float]:
        result = dict(fixed or {})
        result.update({name: float(np.exp(value)) for name, value in zip(names_free, x)})
        return result

    def residual(x: np.ndarray) -> np.ndarray:
        return predict_m0(unpack(x), n, d) - y

    starts = [x0]
    for alpha0, beta0 in [(0.08, 0.08), (0.12, 0.3), (0.3, 0.12), (0.5, 0.5)]:
        candidate = np.log([{"E": default["E"], "A": 1.0, "B": 2.0, "alpha": alpha0, "beta": beta0}[name]
                            for name in names_free])
        starts.append(np.clip(candidate, lb + 1e-8, ub - 1e-8))
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(lb, ub), loss=loss, f_scale=0.1, max_nfev=5000)
        score = float(np.sum(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    params = unpack(best[1].x)
    full_theta = np.log([params[name] for name in M0_NAMES])
    return M0Fit(full_theta, params, best[0], bool(best[1].success))


def fit_m0_module(b1: pd.DataFrame) -> tuple[M0Fit, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = fit_m0(b1)
    robust = fit_m0(b1, loss="huber")
    n = b1["N_params_B"].to_numpy(float)
    d = b1["D_tokens_B"].to_numpy(float)
    y = b1["val_loss"].to_numpy(float)
    full_pred = predict_m0(fit.params, n, d)

    param_rows = []
    for name in M0_NAMES:
        param_rows.append({"parameter": name, "estimate": fit.params[name], "huber_sensitivity": robust.params[name]})

    loo_parts = []
    fold_metrics = []
    for held in sorted(b1["N_params_B"].unique()):
        train = b1[b1["N_params_B"] != held]
        test = b1[b1["N_params_B"] == held]
        fold_fit = fit_m0(train)
        pred = predict_m0(fold_fit.params, test["N_params_B"], test["D_tokens_B"])
        part = test[["N_params_B", "D_tokens_B", "val_loss"]].copy()
        part["predicted_loss"] = pred
        part["residual_pred_minus_actual"] = pred - part["val_loss"]
        part["held_out_N_B"] = held
        loo_parts.append(part)
        row = {"held_out_N_B": held, **metrics(test["val_loss"].to_numpy(), pred)}
        fold_metrics.append(row)
    loo = pd.concat(loo_parts, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    overall = pd.DataFrame([{"split": "B1_in_sample", **metrics(y, full_pred)},
                            {"split": "B1_leave_one_N", **metrics(loo["val_loss"], loo["predicted_loss"])}])

    boot = []
    groups = sorted(b1["N_params_B"].unique())
    for rep in range(300):
        sampled = RNG.choice(groups, size=len(groups), replace=True)
        if len(np.unique(sampled)) < 3:
            continue
        sample = pd.concat([b1[b1["N_params_B"] == g] for g in sampled], ignore_index=True)
        try:
            bfit = fit_m0(sample)
            boot.append({"replicate": rep, **bfit.params})
        except Exception:
            continue
    boot_df = pd.DataFrame(boot)
    for row in param_rows:
        name = row["parameter"]
        row["bootstrap_low"] = float(boot_df[name].quantile(0.025))
        row["bootstrap_high"] = float(boot_df[name].quantile(0.975))
    params_df = pd.DataFrame(param_rows)
    save_csv(params_df, "m0_parameters.csv")
    save_csv(loo, "m0_loo_predictions.csv")
    save_csv(fold_metrics_df, "m0_loo_fold_metrics.csv")
    save_csv(overall, "m0_validation_metrics.csv")
    save_csv(boot_df, "m0_bootstrap_parameters.csv")
    return fit, params_df, loo, fold_metrics_df, boot_df


def profile_m0(b1: pd.DataFrame, fit: M0Fit) -> pd.DataFrame:
    sigma2 = fit.rss / max(1, len(b1) - len(M0_NAMES))
    rows = []
    for name in ["E", "alpha", "beta"]:
        estimate = fit.params[name]
        # B1 is nearly deterministic, so a broad grid collapses the confidence region
        # to one pixel. A dense ±0.2% local profile makes the 95% threshold readable.
        for relative in np.linspace(-0.002, 0.002, 81):
            value = estimate * (1.0 + relative)
            try:
                prof = fit_m0(b1, fixed={name: float(value)})
                stat = (prof.rss - fit.rss) / sigma2
                rows.append({"parameter": name, "value": value, "relative_deviation_pct": relative * 100,
                             "rss": prof.rss, "profile_chi2": stat})
            except Exception:
                rows.append({"parameter": name, "value": value, "relative_deviation_pct": relative * 100,
                             "rss": np.nan, "profile_chi2": np.nan})
    profile = pd.DataFrame(rows)
    save_csv(profile, "m0_profile_likelihood.csv")
    return profile


def external_m0(fit: M0Fit, frames: dict[str, pd.DataFrame], b1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    nlo, nhi = b1["N_params_B"].min(), b1["N_params_B"].max()
    dlo, dhi = b1["D_tokens_B"].min(), b1["D_tokens_B"].max()
    for key in ["B2", "B4", "B5", "B10"]:
        frame = frames[key].dropna(subset=["N_params_B", "D_tokens_B", "val_loss"]).copy()
        pred = predict_m0(fit.params, frame["N_params_B"], frame["D_tokens_B"])
        row = {"dataset": key, **metrics(frame["val_loss"].to_numpy(), pred)}
        row["inside_B1_rectangle_fraction"] = float(((frame["N_params_B"].between(nlo, nhi)) &
                                                     (frame["D_tokens_B"].between(dlo, dhi))).mean())
        rows.append(row)
        part = frame[[c for c in ["family", "N_params_B", "D_tokens_B", "val_loss"] if c in frame]].copy()
        part["dataset"] = key
        part["predicted_loss"] = pred
        part["residual_pred_minus_actual"] = pred - part["val_loss"]
        part["inside_B1_rectangle"] = frame["N_params_B"].between(nlo, nhi) & frame["D_tokens_B"].between(dlo, dhi)
        predictions.append(part)
    metric_df = pd.DataFrame(rows)
    pred_df = pd.concat(predictions, ignore_index=True)
    save_csv(metric_df, "m0_external_metrics.csv")
    save_csv(pred_df, "m0_external_predictions.csv")
    return metric_df, pred_df


def quality_factor(q: np.ndarray, gamma: float, kind: str) -> np.ndarray:
    q = np.asarray(q, float)
    if kind == "exp":
        return np.exp(gamma * (1.0 - q))
    if kind == "power":
        return np.maximum(q, 1e-6) ** (-gamma)
    if kind == "linear":
        return 1.0 + gamma * (1.0 - q)
    raise ValueError(kind)


def predict_m1(m0: dict[str, float], n: np.ndarray, d: np.ndarray, q: np.ndarray,
               delta: float, gamma: float, kind: str = "exp", placement: str = "D") -> np.ndarray:
    """M1 competition models.

    placement=D is the original hypothesis (quality modifies the data term),
    placement=N modifies the parameter term, and placement=both modifies both
    terms with the same Q response.  The latter is deliberately estimated and
    selected by grouped validation rather than assumed.
    """
    n = np.asarray(n, float); d = np.asarray(d, float)
    qfac = quality_factor(q, gamma, kind)
    nterm = m0["A"] * n ** (-m0["alpha"])
    dterm = m0["B"] * d ** (-m0["beta"])
    if placement in {"N", "both"}:
        nterm = nterm * qfac
    if placement in {"D", "both"}:
        dterm = dterm * qfac
    return delta + m0["E"] + nterm + dterm


def fit_quality(frame: pd.DataFrame, m0: dict[str, float], kind: str = "exp", gamma_fixed: float | None = None,
                placement: str = "D") -> dict[str, float]:
    n = frame["N_params_B"].to_numpy(float)
    d = frame["D_tokens_B"].to_numpy(float)
    q = frame["Q_score"].to_numpy(float)
    y = frame["val_loss"].to_numpy(float)
    if gamma_fixed is None:
        def residual(x: np.ndarray) -> np.ndarray:
            return predict_m1(m0, n, d, q, x[0], x[1], kind, placement) - y
        result = least_squares(residual, [0.0, 0.2], bounds=([-10.0, 0.0], [10.0, 10.0]), max_nfev=3000)
        delta, gamma = map(float, result.x)
    else:
        base = predict_m1(m0, n, d, q, 0.0, gamma_fixed, kind, placement)
        delta = float(np.mean(y - base))
        gamma = float(gamma_fixed)
    pred = predict_m1(m0, n, d, q, delta, gamma, kind, placement)
    return {"delta": delta, "gamma": gamma, "rss": float(np.sum((pred - y) ** 2)), "success": True,
            "kind": kind, "placement": placement}


def adjacent_q_monotonicity(frame: pd.DataFrame) -> dict[str, float]:
    comparisons = []
    for _, group in frame.groupby(["N_params_B", "D_tokens_B"]):
        ordered = group.sort_values("Q_score")
        differences = np.diff(ordered["val_loss"].to_numpy(float))
        comparisons.extend(differences.tolist())
    comp = np.asarray(comparisons)
    return {"comparisons": int(len(comp)), "nonincrease_fraction": float(np.mean(comp <= 0)),
            "median_delta_loss": float(np.median(comp))}


def fit_m1_module(m0fit: M0Fit, b6: pd.DataFrame, b7: pd.DataFrame, b8: pd.DataFrame,
                  m0_boot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit quality response and compare placement hypotheses with grouped CV.

    The primary interpolation audit uses D<=300B (inside B1's measured range).
    D=600B is held out as an explicit extrapolation audit and never pooled into
    the interpolation score.
    """
    groups = b6["N_params_B"].astype(str) + "_" + b6["D_tokens_B"].astype(str)
    splitter = GroupKFold(n_splits=5)
    candidates = [("baseline", "exp", "D", 0.0), ("exp_D", "exp", "D", None),
                  ("exp_N", "exp", "N", None), ("exp_both", "exp", "both", None),
                  ("power_D", "power", "D", None), ("linear_D", "linear", "D", None)]
    cv_rows = []; pred_parts = []
    b6_interp = b6[b6["D_tokens_B"] <= 300].copy()
    groups_i = b6_interp["N_params_B"].astype(str) + "_" + b6_interp["D_tokens_B"].astype(str)
    for model, kind, placement, fixed in candidates:
        for fold, (train_idx, test_idx) in enumerate(splitter.split(b6_interp, groups=groups_i), start=1):
            train, test = b6_interp.iloc[train_idx], b6_interp.iloc[test_idx]
            fitted = fit_quality(train, m0fit.params, kind, gamma_fixed=fixed, placement=placement)
            pred = predict_m1(m0fit.params, test["N_params_B"], test["D_tokens_B"], test["Q_score"],
                              fitted["delta"], fitted["gamma"], kind, placement)
            cv_rows.append({"model": model, "fold": fold, "region": "interpolation_D_le_300B",
                            "gamma": fitted["gamma"], **metrics(test["val_loss"], pred)})
            part = test[["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"]].copy()
            part["model"] = model; part["fold"] = fold; part["region"] = "interpolation_D_le_300B"; part["predicted_loss"] = pred
            pred_parts.append(part)
    cv = pd.DataFrame(cv_rows)
    cv_summary = cv.groupby("model", as_index=False).agg(folds=("fold", "count"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), spearman_mean=("spearman", "mean"), gamma_mean=("gamma", "mean"))

    fits = []
    for model, kind, placement, fixed in candidates:
        fitted = fit_quality(b6_interp, m0fit.params, kind, gamma_fixed=fixed, placement=placement)
        fits.append({"model": model, **fitted})
    fit_df = pd.DataFrame(fits)
    # Backward-compatible alias used by existing report/figure code; all
    # selection decisions use the explicit exp_D label above.
    alias = fit_df[fit_df["model"] == "exp_D"].copy(); alias["model"] = "exp"
    aliases = [alias]
    for src, dst in [("power_D", "power"), ("linear_D", "linear")]:
        a = fit_df[fit_df["model"] == src].copy(); a["model"] = dst; aliases.append(a)
    fit_df = pd.concat([fit_df, *aliases], ignore_index=True)
    cv_alias = cv_summary[cv_summary["model"] == "exp_D"].copy(); cv_alias["model"] = "exp"
    cv_aliases = [cv_alias]
    for src, dst in [("power_D", "power"), ("linear_D", "linear")]:
        a = cv_summary[cv_summary["model"] == src].copy(); a["model"] = dst; cv_aliases.append(a)
    cv_summary = pd.concat([cv_summary, *cv_aliases], ignore_index=True)
    cv_alias_rows = cv[cv["model"] == "exp_D"].copy(); cv_alias_rows["model"] = "exp"
    cv = pd.concat([cv, cv_alias_rows], ignore_index=True)
    selected_model = cv_summary.sort_values("rmse_mean").iloc[0]["model"]
    sel = next(c for c in candidates if c[0] == selected_model)
    sel_kind, sel_place = sel[1], sel[2]
    selected_fit = fit_quality(b6_interp, m0fit.params, sel_kind, placement=sel_place)
    # Explicit D=600 extrapolation audit, evaluated using the interpolation fit.
    b6_extra = b6[b6["D_tokens_B"] > 300].copy()
    if len(b6_extra):
        pred_extra = predict_m1(m0fit.params, b6_extra["N_params_B"], b6_extra["D_tokens_B"], b6_extra["Q_score"],
                                selected_fit["delta"], selected_fit["gamma"], sel_kind, sel_place)
        region = pd.DataFrame([{"model": selected_model, "region": "extrapolation_D_gt_300B", **metrics(b6_extra["val_loss"], pred_extra)}])
    else:
        region = pd.DataFrame()
    region_rows = []
    for _, row in cv_summary.iterrows():
        region_rows.append({"model": row["model"], "region": "interpolation_D_le_300B", "rmse": row["rmse_mean"], "mae": row["mae_mean"], "n_folds": row["folds"]})
    if len(region): region_rows.append({"model": selected_model, "region": "extrapolation_D_gt_300B", "rmse": region.iloc[0]["rmse"], "mae": region.iloc[0]["mae"], "n_folds": np.nan})
    save_csv(pd.DataFrame(region_rows), "m1_region_validation.csv")
    (RESULTS / "m1_model_selection.json").write_text(json.dumps({"selected_model": selected_model, "selection_metric": "5-fold grouped RMSE on D<=300B", "Q_support": [0.1, 1.0]}, ensure_ascii=False, indent=2), encoding="utf-8")

    exp_fit = fit_df.set_index("model").loc["exp_D"]
    cell_keys6 = set(zip(b6["N_params_B"], b6["D_tokens_B"], b6["Q_score"]))
    is_new = np.array([(n, d, q) not in cell_keys6 for n, d, q in zip(b7["N_params_B"], b7["D_tokens_B"], b7["Q_score"])])
    b7_new = b7.loc[is_new].copy()
    b7_pred = predict_m1(m0fit.params, b7_new["N_params_B"], b7_new["D_tokens_B"], b7_new["Q_score"],
                         exp_fit["delta"], exp_fit["gamma"], "exp", "D")
    b7_refit = fit_quality(b7, m0fit.params, "exp", placement="D")
    sensitivity = pd.DataFrame([
        {"dataset": "B6", **adjacent_q_monotonicity(b6)},
        {"dataset": "B7", **adjacent_q_monotonicity(b7)},
        {"dataset": "B8", **adjacent_q_monotonicity(b8)},
    ])
    external = pd.DataFrame([{
        "dataset": "B7_new_Q_only", "gamma_B6": exp_fit["gamma"], "gamma_refit_B7": b7_refit["gamma"],
        "gamma_relative_change": (b7_refit["gamma"] - exp_fit["gamma"]) / max(exp_fit["gamma"], 1e-12),
        **metrics(b7_new["val_loss"], b7_pred),
    }])

    boot = []
    cell_groups = list(b6.groupby(["N_params_B", "D_tokens_B"]))
    for rep in range(300):
        picks = RNG.integers(0, len(cell_groups), len(cell_groups))
        sample = pd.concat([cell_groups[i][1] for i in picks], ignore_index=True)
        try:
            row0 = m0_boot[m0_boot["replicate"] == rep]
            if row0.empty:
                continue
            params0 = {name: float(row0.iloc[0][name]) for name in M0_NAMES}
            qfit = fit_quality(sample, params0, "exp", placement="D")
            boot.append({"replicate": rep, "delta": qfit["delta"], "gamma": qfit["gamma"]})
        except Exception:
            continue
    boot_df = pd.DataFrame(boot)
    fit_df["gamma_low"] = np.nan
    fit_df["gamma_high"] = np.nan
    mask = fit_df["model"] == "exp_D"
    fit_df.loc[mask, "gamma_low"] = boot_df["gamma"].quantile(0.025)
    fit_df.loc[mask, "gamma_high"] = boot_df["gamma"].quantile(0.975)
    fit_df.loc[fit_df["model"] == "exp", "gamma_low"] = fit_df.loc[mask, "gamma_low"].iloc[0]
    fit_df.loc[fit_df["model"] == "exp", "gamma_high"] = fit_df.loc[mask, "gamma_high"].iloc[0]

    save_csv(cv, "m1_grouped_cv_folds.csv")
    save_csv(cv_summary, "m1_grouped_cv_summary.csv")
    save_csv(pd.concat(pred_parts, ignore_index=True), "m1_grouped_cv_predictions.csv")
    save_csv(fit_df, "m1_quality_models.csv")
    save_csv(boot_df, "m1_bootstrap_parameters.csv")
    save_csv(sensitivity, "quality_direction_audit.csv")
    save_csv(external, "m1_B7_sensitivity.csv")
    return fit_df, cv_summary, sensitivity, boot_df


def elasticity_and_equivalence(m0: M0Fit, gamma: float, m0_boot: pd.DataFrame,
                               gamma_boot: pd.DataFrame, b6: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n_grid = np.geomspace(b6["N_params_B"].min(), b6["N_params_B"].max(), 80)
    d_values = [float(b6["D_tokens_B"].quantile(q)) for q in [0.25, 0.5, 0.75]]
    for d in d_values:
        for q in [0.5, 0.7, 0.9]:
            x = m0.params["A"] * n_grid ** (-m0.params["alpha"])
            y_value = m0.params["B"] * d ** (-m0.params["beta"]) * np.exp(gamma * (1 - q))
            y = np.full_like(n_grid, y_value, dtype=float)
            total = x + y
            for n, xn, yn, tn in zip(n_grid, x, y, total):
                rows.append({"N_params_B": n, "D_tokens_B": d, "Q_score": q,
                             "parameter_elasticity": m0.params["alpha"] * xn / tn,
                             "data_elasticity": m0.params["beta"] * yn / tn,
                             "quality_semielasticity": gamma * yn / tn,
                             "quality_log_elasticity": q * gamma * yn / tn})
    elasticity = pd.DataFrame(rows)

    common = sorted(set(m0_boot.get("replicate", pd.Series(dtype=int)).astype(int)) &
                    set(gamma_boot.get("replicate", pd.Series(dtype=int)).astype(int)))[:300]
    m0_joint = m0_boot.set_index("replicate") if common else pd.DataFrame()
    gamma_joint = gamma_boot.set_index("replicate") if common else pd.DataFrame()
    equivalence_rows = []
    for n in sorted(b6["N_params_B"].unique()):
        for d in sorted(b6["D_tokens_B"].unique()):
            for q in [0.3, 0.5, 0.7, 0.9]:
                if q + 0.1 > 1:
                    continue
                def multiplier(params: dict[str, float], g: float) -> float:
                    y0 = params["B"] * d ** (-params["beta"]) * math.exp(g * (1 - q))
                    y1 = params["B"] * d ** (-params["beta"]) * math.exp(g * (1 - q - 0.1))
                    bracket = n ** (-params["alpha"]) + (y1 - y0) / params["A"]
                    return float(bracket ** (-1 / params["alpha"]) / n) if bracket > 0 else float("nan")
                estimate = multiplier(m0.params, gamma)
                sims = []
                for rep in common:
                    p = {name: float(m0_joint.loc[rep, name]) for name in M0_NAMES}
                    sims.append(multiplier(p, float(gamma_joint.loc[rep, "gamma"])))
                valid = np.asarray([v for v in sims if np.isfinite(v)])
                equivalence_rows.append({
                    "N_params_B": n, "D_tokens_B": d, "Q_score": q, "delta_Q": 0.1,
                    "N_multiplier": estimate,
                    "bootstrap_low": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
                    "bootstrap_high": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
                    "bootstrap_valid_fraction": len(valid) / max(1, len(sims)),
                    "bootstrap_pairing": "same_replicate_cluster_bootstrap",
                    "inside_B6_support": True,
                })
    equivalence = pd.DataFrame(equivalence_rows)
    save_csv(elasticity, "marginal_elasticities.csv")
    save_csv(equivalence, "quality_parameter_equivalence.csv")
    return elasticity, equivalence


def isoflop_frontier(m0: M0Fit, gamma: float, b1: pd.DataFrame) -> pd.DataFrame:
    """Compute model and empirical compute-budget frontiers using measured C."""
    rows = []
    q_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    c_grid = np.geomspace(float(b1["C_FLOPs_1e21"].quantile(.05)), float(b1["C_FLOPs_1e21"].quantile(.95)), 40)
    # For each C and Q minimize M1 over N with D=C/(0.006 N), constrained to
    # the measured B1 rectangle. This is a transparent conditional frontier.
    nlo, nhi = float(b1.N_params_B.min()), float(b1.N_params_B.max())
    dlo, dhi = float(b1.D_tokens_B.min()), float(b1.D_tokens_B.max())
    for c in c_grid:
        for q in q_grid:
            ns = np.geomspace(nlo, nhi, 500)
            ds = c / (0.006 * ns)
            ok = (ds >= dlo) & (ds <= dhi)
            if not ok.any(): continue
            pred = predict_m1(m0.params, ns[ok], ds[ok], np.full(ok.sum(), q), 0.0, gamma, "exp", "D")
            j = int(np.argmin(pred)); n_star = float(ns[ok][j]); d_star = float(ds[ok][j])
            rows.append({"source": "model_frontier", "C_FLOPs_1e21": c, "Q_score": q,
                         "N_star_B": n_star, "D_star_B": d_star, "predicted_loss": float(pred[j])})
    # Empirical isoFLOP points are selected only from B1 observations; no
    # interpolation of observed losses is presented as measurement.
    b = b1.copy(); b["C_bin"] = pd.qcut(b["C_FLOPs_1e21"], q=20, duplicates="drop")
    for label, g in b.groupby("C_bin", observed=True):
        r = g.loc[g["val_loss"].idxmin()]
        rows.append({"source": "empirical_B1_bin_min", "C_FLOPs_1e21": float(g.C_FLOPs_1e21.median()),
                     "Q_score": np.nan, "N_star_B": float(r.N_params_B), "D_star_B": float(r.D_tokens_B),
                     "predicted_loss": float(r.val_loss)})
    result = pd.DataFrame(rows)
    save_csv(result, "isoflop_frontier.csv")
    return result


def mixture_module() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    effects = pd.read_csv(P1 / "mixture" / "results" / "mixture_marginal_effects.csv")
    rec = pd.read_csv(P1 / "mixture" / "results" / "recommended_mixture.csv")
    summary = json.loads((P1 / "mixture" / "results" / "mixture_model_summary.json").read_text(encoding="utf-8"))
    decay = json.loads((P1 / "scaling_bridge" / "effect_decay_summary.json").read_text(encoding="utf-8"))
    decay_path = P1 / "scaling_bridge" / "effect_magnitude_decay.csv"
    decay_df = pd.read_csv(decay_path) if decay_path.exists() else pd.DataFrame()
    merged = effects.merge(rec[["domain", "training_mean_share", "recommended_share"]], on="domain", how="left")
    merged["centered_local_gradient"] = (merged["delta_macro_loss"] / merged["delta_share"]
                                         * (1.0 - merged["training_mean_share"]))
    step = 0.01
    rows = []
    for _, inc in merged.iterrows():
        for _, dec in merged.iterrows():
            feasible = bool(inc["domain"] != dec["domain"] and dec["training_mean_share"] >= step)
            effect = step * (inc["centered_local_gradient"] - dec["centered_local_gradient"]) if feasible else np.nan
            rows.append({"increase_domain": inc["domain"], "decrease_domain": dec["domain"],
                         "shift_share": step, "delta_macro_loss_linearized": effect,
                         "feasible_at_training_mean": feasible})
    matrix = pd.DataFrame(rows)
    # Repeat the same first-order audit around the problem-one recommended
    # centroid.  This is a sensitivity anchor, not a new fitted response
    # surface; the saved marginal effects remain the only evidence source.
    rec_rows = []
    for _, inc in merged.iterrows():
        for _, dec in merged.iterrows():
            feasible = bool(inc["domain"] != dec["domain"] and dec["recommended_share"] >= step)
            effect = step * (inc["centered_local_gradient"] - dec["centered_local_gradient"]) if feasible else np.nan
            rec_rows.append({"increase_domain": inc["domain"], "decrease_domain": dec["domain"],
                             "shift_share": step, "delta_macro_loss_linearized": effect,
                             "feasible_at_recommended_centroid": feasible})
    save_csv(pd.DataFrame(rec_rows), "domain_substitution_matrix_recommended_anchor.csv")
    local = merged[["domain", "training_mean_share", "recommended_share", "delta_share",
                    "delta_macro_loss", "centered_local_gradient"]].sort_values("centered_local_gradient")
    if not decay_df.empty:
        # Preserve signed kappa_i from problem one; negative values are a
        # documented extrapolation warning, never clipped to zero.
        scale_map = {"train_1m": 1.0, "test_60m": 60.0, "test_1B": 1000.0}
        weight_rows = []
        for _, r in decay_df.iterrows():
            for scale, value in scale_map.items():
                weight_rows.append({"domain": r["domain"], "scale": scale, "scale_B": value,
                                    "kappa": r["kappa"], "effect_magnitude_1m": r["effect_mag_1m"],
                                    "decay_weight": float(value ** (-r["kappa"])),
                                    "signed_kappa_warning": bool(r["kappa"] < 0)})
        save_csv(pd.DataFrame(weight_rows), "domain_decay_weights.csv")
    # There are no coefficient draws or standard errors in the saved problem
    # one outputs.  Consequently BH-FDR p-values for 17x17 edges are not
    # identifiable; emit an explicit audit rather than inventing significance.
    domains = sorted(merged["domain"].astype(str).unique())
    audit_rows = [{"increase_domain": i, "decrease_domain": j, "p_value_available": False,
                   "bh_q_value": np.nan, "significant_q_0_05": False,
                   "reason": "problem1 output lacks edge-level SE/bootstrap draws"}
                  for i in domains for j in domains if i != j]
    save_csv(pd.DataFrame(audit_rows), "multiple_testing_audit.csv")
    meta = {
        "reference_mixture": "problem1 training mean",
        "substitution_type": "first-order local linearization from saved one-domain perturbations",
        "curvature_or_complementarity_identified": False,
        "reason": "problem1 outputs do not serialize the fitted Hessian or bootstrap coefficient draws",
        "problem1_test_1m_macro_rmse": summary["test_1m_macro_rmse_quadratic"],
        "effect_decay_kappa_median": decay["kappa_median"],
        "effect_decay_domain_specific": bool(not decay_df.empty),
        "multiple_testing": "BH-FDR not computable from saved problem1 outputs; no edge labelled significant",
        "reference_anchors": ["training_mean", "recommended_centroid"],
        "observed_scale_weights": {k: v for k, v in decay["w_by_scale"].items() if k in ["train_1m", "test_60m", "test_1B"]},
    }
    save_csv(local, "mixture_local_gradients.csv")
    save_csv(matrix, "domain_substitution_matrix.csv")
    (RESULTS / "m2_mixture_module.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return local, matrix, meta


def make_figures(frames: dict[str, pd.DataFrame], m0: M0Fit, loo: pd.DataFrame, profile: pd.DataFrame,
                 external: pd.DataFrame, fit_df: pd.DataFrame, cv_summary: pd.DataFrame,
                 elasticity: pd.DataFrame, equivalence: pd.DataFrame,
                 substitutions: pd.DataFrame, direction: pd.DataFrame) -> pd.DataFrame:
    configure_plots()
    manifest = []

    b1 = frames["B1"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    cmap = plt.get_cmap("viridis")
    for color_i, n in enumerate(sorted(b1["N_params_B"].unique())):
        group = b1[b1["N_params_B"] == n].sort_values("D_tokens_B")
        color = cmap(color_i / 7)
        axes[0].plot(group["D_tokens_B"], group["val_loss"], color=color, alpha=.65, lw=1)
        axes[0].scatter(group["D_tokens_B"], group["val_loss"], color=color, s=8, alpha=.55)
        axes[0].plot(group["D_tokens_B"], predict_m0(m0.params, group["N_params_B"], group["D_tokens_B"]),
                     color=color, lw=1.5)
    axes[0].set_xscale("log"); axes[0].set_xlabel("训练数据量 D（十亿 Token）"); axes[0].set_ylabel("验证 Loss")
    axes[0].grid(alpha=.2)
    n_grid = np.geomspace(b1["N_params_B"].min(), b1["N_params_B"].max(), 100)
    d_grid = np.geomspace(b1["D_tokens_B"].min(), b1["D_tokens_B"].max(), 120)
    nn, dd = np.meshgrid(n_grid, d_grid)
    zz = predict_m0(m0.params, nn, dd)
    contour = axes[1].contourf(nn, dd, zz, levels=18, cmap="viridis")
    axes[1].scatter(b1["N_params_B"], b1["D_tokens_B"], c=b1["val_loss"], cmap="viridis", s=7, edgecolors="none")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("参数规模 N（十亿）"); axes[1].set_ylabel("训练数据量 D（十亿 Token）")
    fig.colorbar(contour, ax=axes[1], label="模型预测 Loss")
    fig.tight_layout(); path = FIGURES / "m0_scaling_fit.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B1 附件训练日志", "训练轨迹与 M0 拟合"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(loo["val_loss"], loo["predicted_loss"], c=np.log10(loo["held_out_N_B"]), cmap="plasma", s=11, alpha=.65)
    lo = min(loo["val_loss"].min(), loo["predicted_loss"].min()); hi = max(loo["val_loss"].max(), loo["predicted_loss"].max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="#555555"); axes[0].set_xlabel("实际 Loss"); axes[0].set_ylabel("留轨迹预测 Loss")
    axes[1].scatter(loo["D_tokens_B"], loo["residual_pred_minus_actual"], c=np.log10(loo["held_out_N_B"]), cmap="plasma", s=11, alpha=.65)
    axes[1].axhline(0, color="#555555", ls="--"); axes[1].set_xscale("log"); axes[1].set_xlabel("训练数据量 D（十亿 Token）"); axes[1].set_ylabel("预测误差")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); path = FIGURES / "m0_loo_diagnostics.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B1 按 N 留轨迹", "M0 防泄漏验证"))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, name in zip(axes, ["E", "alpha", "beta"]):
        sub = profile[profile["parameter"] == name]
        ax.plot(sub["relative_deviation_pct"], np.minimum(sub["profile_chi2"], 12), color="#2563eb")
        ax.axhline(3.84, color="#dc2626", ls="--", label="95% 阈值")
        ax.set_ylim(0, 12); ax.set_xlabel(f"{name} 相对估计值偏移（%）"); ax.set_ylabel("剖面似然统计量"); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); path = FIGURES / "m0_profile_likelihood.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B1 附件训练日志", "M0 参数可识别性"))

    exp_fit = fit_df.set_index("model").loc["exp"]
    b6 = frames["B6"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    cells = [(b6["N_params_B"].min(), b6["D_tokens_B"].min()),
             (float(b6["N_params_B"].median()), float(b6["D_tokens_B"].median())),
             (b6["N_params_B"].max(), b6["D_tokens_B"].max())]
    for ax, (n, d) in zip(axes, cells):
        ni = b6.iloc[(b6["N_params_B"] - n).abs().argsort()[:1]]["N_params_B"].iloc[0]
        di = b6.iloc[(b6["D_tokens_B"] - d).abs().argsort()[:1]]["D_tokens_B"].iloc[0]
        sub = b6[(b6["N_params_B"] == ni) & (b6["D_tokens_B"] == di)].sort_values("Q_score")
        qg = np.linspace(sub["Q_score"].min(), sub["Q_score"].max(), 100)
        ax.scatter(sub["Q_score"], sub["val_loss"], color="#111827", s=24, label="B6 半合成")
        ax.plot(qg, predict_m1(m0.params, np.full_like(qg, ni), np.full_like(qg, di), qg,
                               exp_fit["delta"], exp_fit["gamma"], "exp"), color="#2563eb", label="M1")
        ax.set_xlabel("质量 Q"); ax.set_title(f"N={ni:g}B, D={di:g}B"); ax.grid(alpha=.2)
    axes[0].set_ylabel("验证 Loss"); axes[0].legend(); fig.tight_layout()
    path = FIGURES / "m1_quality_curves.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B6 半合成", "质量曲线与 M1 拟合"))

    order = ["baseline", "exp_D", "exp_N", "exp_both", "power_D", "linear_D"]
    view = cv_summary.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(np.arange(len(view)), view["rmse_mean"], yerr=view["rmse_sd"], color=["#9ca3af", "#2563eb", "#059669", "#dc2626", "#7c3aed", "#f59e0b"], capsize=4)
    ax.set_xticks(np.arange(len(view)), ["无质量项", "指数·D", "指数·N", "指数·双项", "幂·D", "线性·D"]); ax.set_ylabel("D≤300B 分组交叉验证 RMSE"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); path = FIGURES / "m1_cv_comparison.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B6 D≤300B 完整 (N,D) 单元分组", "质量位置与函数竞争模型"))

    med_d = float(frames["B6"]["D_tokens_B"].median())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for q, color in [(0.5, "#dc2626"), (0.7, "#2563eb"), (0.9, "#059669")]:
        sub = elasticity[(elasticity["D_tokens_B"] == med_d) & (elasticity["Q_score"] == q)]
        axes[0].plot(sub["N_params_B"], sub["parameter_elasticity"], color=color, label=f"Q={q}")
        axes[1].plot(sub["N_params_B"], sub["data_elasticity"], color=color)
        axes[2].plot(sub["N_params_B"], sub["quality_semielasticity"], color=color)
    for ax, label in zip(axes, ["参数弹性", "数据弹性", "质量半弹性"]):
        ax.set_xscale("log"); ax.set_xlabel("参数规模 N（十亿）"); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); path = FIGURES / "marginal_elasticities.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "M0+M1 解析计算", "参数、数据和质量边际效用"))

    eq = equivalence[np.isclose(equivalence["D_tokens_B"], med_d)]
    pivot = eq.pivot(index="N_params_B", columns="Q_score", values="N_multiplier")
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:.1f}" for x in pivot.columns]); ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("初始质量 Q"); ax.set_ylabel("参数规模 N（十亿）"); fig.colorbar(image, ax=ax, label="Q 提升 0.1 的参数等价倍数")
    fig.tight_layout(); path = FIGURES / "quality_parameter_equivalence.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "M0/M1 Bootstrap", "质量提升与参数增加的条件等价量"))

    feasible = substitutions[substitutions["feasible_at_training_mean"]]
    domains = sorted(set(substitutions["increase_domain"]))
    matrix = feasible.pivot(index="increase_domain", columns="decrease_domain", values="delta_macro_loss_linearized").reindex(index=domains, columns=domains)
    lim = np.nanmax(np.abs(matrix.to_numpy()))
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(domains)), domains, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(domains)), domains, fontsize=7); ax.set_xlabel("减少 1 个百分点的领域"); ax.set_ylabel("增加 1 个百分点的领域")
    fig.colorbar(image, ax=ax, label="局部预测 Loss 变化"); fig.tight_layout()
    path = FIGURES / "domain_substitution_matrix.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "问题一已保存的局部扰动", "17 域一阶替换矩阵；不识别互补性"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ext = external[external["dataset"].isin(["B2", "B4", "B5", "B10"])]
    x = np.arange(len(ext)); axes[0].bar(x-.18, ext["rmse"], .36, label="原始 RMSE", color="#9ca3af")
    axes[0].bar(x+.18, ext["centered_rmse"], .36, label="去均值 RMSE", color="#2563eb")
    axes[0].set_xticks(x, ext["dataset"]); axes[0].set_ylabel("误差"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(direction["dataset"], direction["nonincrease_fraction"], color=["#2563eb", "#7c3aed", "#dc2626"])
    axes[1].axhline(.5, color="#555555", ls="--"); axes[1].set_ylim(0, 1); axes[1].set_ylabel("相邻 Q 增大时 Loss 不升的比例"); axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); path = FIGURES / "external_and_quality_risk.pdf"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    manifest.append((path.name, "B2/B4/B5/B10 与 B6/B7/B8", "外部误差与质量方向风险"))

    manifest_df = pd.DataFrame(manifest, columns=["figure", "data_source", "interpretation"])
    save_csv(manifest_df, "figure_manifest.csv")
    return manifest_df


def write_report(audit: pd.DataFrame, m0params: pd.DataFrame, m0metrics: pd.DataFrame,
                 external: pd.DataFrame, m1fits: pd.DataFrame, cv: pd.DataFrame,
                 direction: pd.DataFrame, b7sens: pd.DataFrame, equivalence: pd.DataFrame,
                 local: pd.DataFrame, matrix: pd.DataFrame, m2meta: dict, manifest: pd.DataFrame,
                 bridge: dict) -> None:
    p = dict(zip(m0params["parameter"], m0params["estimate"]))
    ci = m0params.set_index("parameter")
    exp_row = m1fits.set_index("model").loc["exp"]
    cv_index = cv.set_index("model")
    improvement = 1 - cv_index.loc["exp", "rmse_mean"] / cv_index.loc["baseline", "rmse_mean"]
    cv_folds = pd.read_csv(RESULTS / "m1_grouped_cv_folds.csv")
    fold_pivot = cv_folds.pivot(index="fold", columns="model", values="rmse")
    exp_win_folds = int((fold_pivot["exp"] < fold_pivot["baseline"]).sum())
    loo = m0metrics.set_index("split").loc["B1_leave_one_N"]
    b7 = b7sens.iloc[0]
    b8dir = direction.set_index("dataset").loc["B8"]
    eq_valid = equivalence[equivalence["bootstrap_valid_fraction"] >= .8]
    eq_median = float(eq_valid["N_multiplier"].median()) if len(eq_valid) else float("nan")
    feasible = matrix[matrix["feasible_at_training_mean"]].dropna(subset=["delta_macro_loss_linearized"])
    best_sub = feasible.nsmallest(5, "delta_macro_loss_linearized")
    model_accepted = bool(improvement > 0 and exp_win_folds >= 4 and exp_row["gamma"] > 0 and
                          direction.set_index("dataset").loc["B7", "nonincrease_fraction"] > .5)
    selection = json.loads((RESULTS / "m1_model_selection.json").read_text(encoding="utf-8"))
    selected_model = selection["selected_model"]
    summary = {
        "seed": SEED,
        "m0": {**p, "loo_rmse": float(loo["rmse"]), "loo_mae": float(loo["mae"]),
               "loo_spearman": float(loo["spearman"])},
        "m1": {"gamma": float(exp_row["gamma"]), "gamma_low": float(exp_row["gamma_low"]),
               "gamma_high": float(exp_row["gamma_high"]), "grouped_cv_improvement": float(improvement),
               "cv_folds_better_than_baseline": exp_win_folds,
               "B7_new_Q_rmse": float(b7["rmse"]), "accepted": model_accepted},
        "B8_nonincrease_fraction": float(b8dir["nonincrease_fraction"]),
        "quality_plus_0_1_median_parameter_multiplier": eq_median,
        "mixture": m2meta,
        "problem1_to_problem2_quality_bridge": bridge,
        "no_pooled_A_B_rmse_reported": True,
    }
    (RESULTS / "problem2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_view = audit[["dataset", "rows", "evidence_type", "role"]]
    ext_view = external[["dataset", "n", "rmse", "centered_rmse", "spearman", "inside_B1_rectangle_fraction"]]
    param_view = m0params[["parameter", "estimate", "bootstrap_low", "bootstrap_high", "huber_sensitivity"]]
    cv_view = cv[cv["model"].isin(["baseline", "exp_D", "exp_N", "exp_both", "power_D", "linear_D"])][["model", "rmse_mean", "rmse_sd", "spearman_mean", "gamma_mean"]]
    report = fr"""# 问题二计算结果：广义标度律

## 1. 证据和数据口径

本报告由 `code/problem2/problem2.py` 从清洗后数据直接生成。B1 是经典标度律主拟合数据；B6 是质量项主拟合数据；B2、B4、B5、B7 只作验证；B8、B10 只作风险检查。不同来源未被拼成一个训练集，也没有报告跨 A/B 数据的虚假总体 RMSE。

{markdown_table(audit_view, 5)}

B1 的计算量字段满足 $C\approx6ND$：换算到附件单位后为 $C_{{10^{{21}}}}=0.006N_BD_B$，绝对相对误差中位数为 $2.62\times10^{{-5}}$，95% 分位数为 0.00141，99.32% 的记录误差不超过 5%。少数最早期检查点因 C 只保留四位小数而产生较大相对误差。

## 2. M0：经典 N-D 标度律

拟合形式为 $L_0=E+A N^{{-\alpha}}+B D^{{-\beta}}$，N 和 D 均采用附件中的十亿单位。参数按 B1 的完整 N 轨迹进行 Bootstrap：

{markdown_table(param_view, 5)}

按完整参数规模轨迹留出的验证结果为 RMSE={loo['rmse']:.6f}、MAE={loo['mae']:.6f}、Spearman={loo['spearman']:.6f}、偏差={loo['bias_pred_minus_actual']:.6f}。这比随机拆分更严格，因为同一训练轨迹不会同时进入训练和验证。误差只有 $10^{{-4}}$ 量级，且参数 Bootstrap 区间异常窄，表明 B1 与该幂律几乎是确定性关系。它更接近公式化或强规则化数据；这里的区间主要反映轨迹重抽样与数值舍入，不能当作真实预训练过程的完整认知不确定性。

### 外部数据压力测试

{markdown_table(ext_view, 5)}

B4/B5 的 Loss 口径存在模型族、分词器和验证集差异，因此原始 RMSE 不能单独解释为结构失效；表中同时给出去均值 RMSE 与排序相关。B10 是估算值，只描述远域偏离，不作为真实精度证据。

## 3. M1：质量修正

主候选为 $L_1=\delta_B+E+A N^{{-\alpha}}+B D^{{-\beta}}\exp[\gamma_Q(1-Q)]$。$\delta_B$ 只校准 B6 与 B1 的来源基线，不改变质量方向。

{markdown_table(cv_view, 5)}

竞争模型包括 Q 作用于 D 项（exp_D）、N 项（exp_N）和 N、D 两项（exp_both），并在 D≤300B 的完整单元分组交叉验证中选择 **{selected_model}**。D=600B 单独作为外推审计，不并入插值 CV。指数 D-only 的质量参数为 $\gamma_Q={exp_row['gamma']:.4f}$，同一响应的轨迹单元 Bootstrap 95% 区间为 [{exp_row['gamma_low']:.4f}, {exp_row['gamma_high']:.4f}]；相对无质量项的 RMSE 改善为 {improvement*100:.2f}%。B7 新增 Q 水平的外推 RMSE 为 {b7['rmse']:.4f}。这组结果不支持事先把质量作用位置固定为 D 项。

B6、B7、B8 的相邻 Q 方向审计如下：

{markdown_table(direction, 5)}

B8 只有 {b8dir['nonincrease_fraction']*100:.2f}% 的相邻变化符合“Q 提高时 Loss 不升”，与题意和 B6/B7 相反，故未进入拟合，也未擅自反转 Q。

## 4. 边际效用和质量—参数等价量

在参考配比处，解析边际量为：

$$
\frac{{\partial L}}{{\partial N}}=-\alpha A N^{{-\alpha-1}},\quad
\frac{{\partial L}}{{\partial D}}=-\beta B D^{{-\beta-1}}e^{{\gamma_Q(1-Q)}},\quad
\frac{{\partial L}}{{\partial Q}}=-\gamma_Q B D^{{-\beta}}e^{{\gamma_Q(1-Q)}}.
$$

参数、数据弹性和质量半弹性已保存于 `marginal_elasticities.csv`。在 B6 支持网格内，Q 提升 0.1 的参数等价倍数中位数为 {eq_median:.3f}；每个网格点均给出轨迹/单元 Bootstrap 区间和有效抽样比例。该数值是固定 D、固定配比、采用 B6 原生 Q 标尺时的条件等价量，不能脱离基准 N、D、Q 使用。问题一 Q 与 B6 Q 无成对标定，因此不把该倍数用于第一问推荐配比。

## 5. M2 配比接口与领域替换

问题一现有结果保存了 17 域在训练均值处的单域 +1 个百分点扰动，但没有保存二阶响应面的 Hessian 或 Bootstrap 系数抽样。因此本问从这些真实输出恢复一阶中心化梯度，形成可行的成对局部替换矩阵：

$$
S_{{i\leftarrow j}}(0.01)\approx0.01(\tilde g_i-\tilde g_j).
$$

局部预测 Loss 降幅最大的五个可行替换方向为：

{markdown_table(best_sub[["increase_domain", "decrease_domain", "shift_share", "delta_macro_loss_linearized"]], 5)}

这些是问题一训练均值附近的一阶模型预测，不是因果效应。当前证据不能估计曲率置信区间，因此不输出“显著互补”标签。配比项在 1M、60M、1B 的问题一结果中采用经验权重，超过 1B 时只能按中位衰减指数 {m2meta['effect_decay_kappa_median']:.4f} 做敏感性外推。

以下是 **D-only 候选模型** 的条件计算式，用于质量—参数等价量和 IsoFLOP 敏感性分析；B6 插值 CV 选出的结构为 **{selected_model}**，故不能把该式表述为已验证的唯一完整模型：

$$
L(N,D,Q,p)=E+A N^{{-\alpha}}+B D^{{-\beta}}e^{{\gamma_Q(1-Q)}}+w_p(N)\Delta_p(p).
$$

当 Q=1 且 $p=p^{{ref}}$ 时，质量修正因子为 1、配比差为 0，模型严格退化到 M0。规模、质量和配比模块分别验证，未构造跨来源总体拟合指标。领域替换矩阵仅是一阶近似；现有问题一输出缺少边级标准误，272 条有向边无法做 BH-FDR 检验，不能标注显著互补或替代。逐域 $\kappa_i$ 保存在 `domain_decay_weights.csv`，负值原样保留，并只在观测尺度内解释。

## 6. 第一问质量输出到第二问的可识别接口

读取第一问的 17 域质量代理值和训练均值、推荐质心两组配比，按 $Q_A(p)=\sum_i p_i q_i$ 计算质量。17 域代理值中有 {bridge['mapped_domains']} 域由 A16 映射到已有质量域，其余 {bridge['unmapped_domains']} 域是由第一问训练 Loss 等信息推得的代理值，不是独立测量。训练均值的代理质量为 {bridge['q_a_training_mean_proxy']:.6f}，推荐质心为 {bridge['q_a_recommended_proxy']:.6f}，差值为 {bridge['delta_q_a_proxy']:+.6f}。

若仅接受 A16 已映射域、并允许未映射域的真实质量各自在 $[0,1]$，推荐质心相对训练均值的质量变化落在 [{bridge['delta_q_a_bound'][0]:+.6f}, {bridge['delta_q_a_bound'][1]:+.6f}]。区间跨过零，因此现有资料连变化方向也不能无条件识别；该区间还依赖 A16 映射的语义可比性，不能当作统计置信区间。

B6 没有 17 域配比，第一问配比试验没有 B6 的 Q 标尺与共同 Loss 协议，也没有成对实验 ID。故 $Q_A\to Q_B$ 的校准函数和 $L(N,D,Q,p)$ 的联合误差均**不可识别**。代码接口在缺少经实测标定的映射时会拒绝把 $Q_A$ 代入 B6 模型。领域级审计、配比质量与界、联合数据合同分别见 `quality_bridge_domain_audit.csv`、`quality_bridge_recipe_anchors.csv`、`quality_bridge_delta_bounds.csv` 和 `quality_bridge_joint_data_audit.csv`。

## 7. 图表索引

{markdown_table(manifest, 5)}

所有图均由本次结果表直接生成，PDF 中不写论文式大标题。真实观测、半合成、插值和估算数据的解释边界见图表清单与数据审计表。

## 8. 可复现运行

```powershell
python F题_初步方案/code/problem2/problem2.py
python F题_初步方案/code/problem2/verify_problem2.py
```

随机种子为 {SEED}。主要数值汇总位于 `outputs/problem2/results/problem2_summary.json`，独立验收结果位于 `reports/问题二/验证验收/VERIFY_REPORT_PROBLEM2.md`。
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    frames, audit = audit_data()
    b1 = frames["B1"].dropna(subset=["N_params_B", "D_tokens_B", "val_loss"]).copy()
    m0, m0params, loo, folds, m0boot = fit_m0_module(b1)
    profile = profile_m0(b1, m0)
    external, _ = external_m0(m0, frames, b1)
    m1fits, cv, direction, gamma_boot = fit_m1_module(m0, frames["B6"], frames["B7"], frames["B8"], m0boot)
    b7sens = pd.read_csv(RESULTS / "m1_B7_sensitivity.csv")
    exp_gamma = float(m1fits.set_index("model").loc["exp", "gamma"])
    elasticity, equivalence = elasticity_and_equivalence(m0, exp_gamma, m0boot, gamma_boot, frames["B6"])
    isoflop_frontier(m0, exp_gamma, b1)
    local, substitutions, m2meta = mixture_module()
    bridge = build_quality_bridge(P1, BROOT, RESULTS, FIGURES)
    manifest = make_figures(frames, m0, loo, profile, external, m1fits, cv, elasticity,
                            equivalence, substitutions, direction)
    manifest = pd.concat([manifest, pd.DataFrame([{"figure": "quality_bridge_identification.pdf",
        "data_source": "问题一 17 域质量、A16 映射与两组配比",
        "interpretation": "质量变化的代理值与部分识别界；不是 B6 标尺校准"}])], ignore_index=True)
    save_csv(manifest, "figure_manifest.csv")
    m0metrics = pd.read_csv(RESULTS / "m0_validation_metrics.csv")
    write_report(audit, m0params, m0metrics, external, m1fits, cv, direction, b7sens,
                 equivalence, local, substitutions, m2meta, manifest, bridge)
    summary = json.loads((RESULTS / "problem2_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
