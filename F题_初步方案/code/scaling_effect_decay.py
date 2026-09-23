"""问题一：配比效应标度律 + 置信衰减融合。

路线 1（extrap_trend）：在每个实测尺度（train_1m / test_60m / test_1B）分别
对 13 个 Loss 域拟合线性 ILR 配比效应系数，对每个 ILR 坐标拟合系数随
log10(D) 的趋势，用留一尺度交叉验证选出斜率收缩因子 lambda，再外推到
10B/70B。

路线 2（decay_fusion）：D 超过 1B 后按 w(D) = (D/1e9)^(-kappa) 把配比项向
尺度基线收缩，kappa 由实测尺度上效应幅值的衰减速度估计。

纪律：est_10b / est_70b 的 Loss 是估计量，绝不参与任何拟合，仅作外部检验。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT.parent / "F题_清洗后"
REGMIX = DATA / "A_data_value" / "regmix_tables"
OUT = PROJECT / "outputs" / "problem1" / "scaling_bridge"
OUT.mkdir(parents=True, exist_ok=True)

SCALE_TOKENS = {
    "train_1m": 1.0e6,
    "test_1m": 1.0e6,
    "test_60m": 6.0e7,
    "test_1B": 1.0e9,
    "est_10b": 1.0e10,
    "est_70b": 7.0e10,
}
COEF_SCALES = ["train_1m", "test_60m", "test_1B"]  # 效应系数拟合只用实测尺度
ALL_SCALES = ["train_1m", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
EXTERNAL = {"est_10b", "est_70b"}
ALPHA = 1.0  # 每尺度岭回归正则
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CLIP_MULT = 3.0  # 外推系数幅值安全上限（相对 1B 系数）


def load_scale(name: str) -> pd.DataFrame:
    prefix, tag = name.split("_", 1)
    mix = pd.read_csv(REGMIX / f"{prefix}_mixture_{tag}.csv")
    loss = pd.read_csv(REGMIX / f"{prefix}_pile_loss_{tag}.csv")
    return mix.merge(loss, on="index", validate="one_to_one")


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def main() -> None:
    H = helmert(17, full=False)  # (16, 17)

    def ilr(P: np.ndarray) -> np.ndarray:
        P = np.maximum(np.asarray(P, dtype=float), 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        return np.log(P) @ H.T

    # ---- 载入所有尺度 ----
    store: dict[str, dict] = {}
    domain_names = None
    for name in ALL_SCALES:
        df = load_scale(name)
        mcols = [c for c in df.columns if c.startswith("train_the_pile_")]
        lcols = [c for c in df.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
        if domain_names is None:
            domain_names = [c.removeprefix("metric/the_pile_").removesuffix("_val_loss") for c in lcols]
        store[name] = {
            "phi": ilr(df[mcols].to_numpy()),
            "L": df[lcols].to_numpy(),
            "logd": float(np.log10(SCALE_TOKENS[name])),
        }
    nd, nc = len(domain_names), H.shape[0]

    # ---- 每尺度逐域岭回归系数（目标按尺度均值中心化，原始 Loss 单位）----
    logd_fit = np.array([store[s]["logd"] for s in COEF_SCALES])
    means_arr = np.stack([store[s]["L"].mean(axis=0) for s in COEF_SCALES])  # (3, nd)
    B = np.zeros((nd, 3, nc))
    for si, s in enumerate(COEF_SCALES):
        phi, L = store[s]["phi"], store[s]["L"]
        G = phi.T @ phi + ALPHA * np.eye(nc)
        for d in range(nd):
            y = L[:, d] - L[:, d].mean()
            B[d, si] = np.linalg.solve(G, phi.T @ y)

    # ---- 逐域基线：log10(Loss均值) 对 log10(D) 线性拟合（3 点 2 参数）----
    base_a = np.zeros(nd)
    base_b = np.zeros(nd)
    for d in range(nd):
        base_b[d], base_a[d] = np.polyfit(logd_fit, np.log10(means_arr[:, d]), 1)

    def baseline_matrix(logd: float) -> np.ndarray:
        return np.power(10.0, base_a + base_b * logd)  # (nd,) -> 广播到 (n, nd)

    # ---- 三点趋势斜率（每个域每个 ILR 坐标）----
    slopes = np.zeros((nd, nc))
    X = np.vstack([logd_fit, np.ones(3)]).T
    for d in range(nd):
        coef, *_ = np.linalg.lstsq(X, B[d], rcond=None)  # X(3,2) @ coef(2,nc) ≈ B[d](3,nc)
        slopes[d] = coef[0]

    # ---- 留一尺度交叉验证选 lambda ----
    loo_rows = []
    for lam in LAMBDA_GRID:
        sq_all = []
        for si_held in range(3):
            keep = [i for i in range(3) if i != si_held]
            ld2 = logd_fit[keep]
            B2 = B[:, keep, :]
            denom = ld2[1] - ld2[0]
            slope2 = (B2[:, 1, :] - B2[:, 0, :]) / denom
            near = 0 if abs(logd_fit[si_held] - ld2[0]) <= abs(logd_fit[si_held] - ld2[1]) else 1
            beta_pred = B2[:, near, :] + lam * slope2 * (logd_fit[si_held] - ld2[near])
            # 留一基线：两点 log-log 直线
            ly = np.log10(means_arr[keep, :])
            b2 = (ly[1] - ly[0]) / denom
            a2 = ly[0] - b2 * ld2[0]
            base_pred = np.power(10.0, a2 + b2 * logd_fit[si_held])
            s_held = COEF_SCALES[si_held]
            phi = store[s_held]["phi"]
            L = store[s_held]["L"]
            mix_pred = phi @ beta_pred.T  # (n, nd)
            full_pred = base_pred[None, :] + mix_pred
            sq_all.append((full_pred - L) ** 2)
        rmse = float(np.sqrt(np.mean(np.concatenate(sq_all, axis=0))))
        loo_rows.append({"lambda": lam, "loo_rmse": rmse})
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(OUT / "effect_lambda_selection.csv", index=False)
    lam_star = float(loo_df.loc[loo_df["loo_rmse"].idxmin(), "lambda"])

    # ---- kappa：效应幅值随 log10(D) 的衰减速度（逐域斜率取中位数）----
    mags = np.linalg.norm(B, axis=2)  # (nd, 3)
    kappa_list = []
    for d in range(nd):
        sl = np.polyfit(logd_fit, np.log(mags[d] + 1e-12), 1)[0]
        kappa_list.append(-sl)
    kappa = float(np.median(kappa_list))
    kappa_per_domain = pd.DataFrame({
        "domain": domain_names,
        "effect_mag_1m": mags[:, 0],
        "effect_mag_60m": mags[:, 1],
        "effect_mag_1B": mags[:, 2],
        "kappa": kappa_list,
    })
    kappa_per_domain.to_csv(OUT / "effect_magnitude_decay.csv", index=False)

    si_1b = COEF_SCALES.index("test_1B")
    beta_1b = B[:, si_1b, :]

    # ---- 各尺度 × 各变体预测 ----
    variants = ["baseline_only", "frozen_1B", "extrap_trend", "decay_fusion"]
    pred_store: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
    w_store = {}
    for s in ALL_SCALES:
        phi = store[s]["phi"]
        logd = store[s]["logd"]
        base = np.broadcast_to(baseline_matrix(logd), (phi.shape[0], nd))
        mix_frozen = phi @ beta_1b.T
        beta_ext = beta_1b + lam_star * slopes * (logd - 9.0)
        # 安全截断：外推系数幅值不超过 1B 系数的 CLIP_MULT 倍
        for d in range(nd):
            lim = CLIP_MULT * np.linalg.norm(beta_1b[d])
            if np.linalg.norm(beta_ext[d]) > lim:
                beta_ext[d] *= lim / np.linalg.norm(beta_ext[d])
        mix_extrap = phi @ beta_ext.T
        if kappa > 0:
            w = float(min(1.0, (SCALE_TOKENS[s] / 1e9) ** (-kappa)))
        else:
            w = 1.0  # 效应未观测到衰减，不收缩
        w_store[s] = w
        pred_store["baseline_only"][s] = base.copy()
        pred_store["frozen_1B"][s] = base + mix_frozen
        pred_store["extrap_trend"][s] = base + mix_extrap
        pred_store["decay_fusion"][s] = base + w * mix_frozen

    # ---- 指标表 ----
    metric_rows = []
    for s in ALL_SCALES:
        actual_macro = store[s]["L"].mean(axis=1)
        for v in variants:
            pred_macro = pred_store[v][s].mean(axis=1)
            metric_rows.append({
                "scale": s,
                "observed": s not in EXTERNAL,
                "variant": v,
                "macro_rmse": float(np.sqrt(np.mean((pred_macro - actual_macro) ** 2))),
                "macro_spearman": spearman_safe(pred_macro, actual_macro),
                "n": store[s]["L"].shape[0],
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "effect_variant_results.csv", index=False)

    # ---- 外部尺度逐域诊断：frozen vs extrap 的逐域 RMSE 差 ----
    diag_rows = []
    for s in sorted(EXTERNAL):
        for d in range(nd):
            actual = store[s]["L"][:, d]
            r_f = float(np.sqrt(np.mean((pred_store["frozen_1B"][s][:, d] - actual) ** 2)))
            r_e = float(np.sqrt(np.mean((pred_store["extrap_trend"][s][:, d] - actual) ** 2)))
            diag_rows.append({
                "scale": s, "domain": domain_names[d],
                "rmse_frozen": r_f, "rmse_extrap": r_e,
                "improvement": r_f - r_e,
                "spearman_frozen": spearman_safe(pred_store["frozen_1B"][s][:, d], actual),
                "spearman_extrap": spearman_safe(pred_store["extrap_trend"][s][:, d], actual),
            })
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "effect_per_domain_external.csv", index=False)

    # ---- 长表预测（供论文绘图）----
    long_rows = []
    for s in ALL_SCALES:
        actual_macro = store[s]["L"].mean(axis=1)
        for v in variants:
            for i in range(store[s]["L"].shape[0]):
                long_rows.append({
                    "scale": s, "variant": v, "recipe_i": i,
                    "actual_macro": float(actual_macro[i]),
                    "pred_macro": float(pred_store[v][s][i].mean()),
                })
    pd.DataFrame(long_rows).to_csv(OUT / "effect_predictions_long.csv", index=False)

    summary = {
        "lambda_selection": loo_rows,
        "lambda_star": lam_star,
        "kappa_median": kappa,
        "w_by_scale": w_store,
        "discipline": "est_10b/est_70b never used in fitting; external check only",
    }
    (OUT / "effect_decay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 控制台输出 ----
    print("=== 配比效应标度律 + 置信衰减 ===")
    print(f"lambda* = {lam_star}（留一尺度 CV 选出）")
    print(f"kappa（效应幅值衰减指数，中位数）= {kappa:.4f}")
    print("各尺度置信权重 w(D)：", {k: round(v, 4) for k, v in w_store.items()})
    print()
    print("效应幅值（前 5 域）：")
    print(kappa_per_domain.head(5).to_string(index=False))
    print()
    print("变体 × 尺度 指标：")
    pivot = metrics.pivot(index="scale", columns="variant", values="macro_spearman")
    print(pivot.round(4).to_string())
    print()
    rmse_pivot = metrics.pivot(index="scale", columns="variant", values="macro_rmse")
    print("macro RMSE：")
    print(rmse_pivot.round(4).to_string())
    print()
    print("外部尺度逐域改善 top 5（rmse_frozen - rmse_extrap）：")
    top = diag.sort_values("improvement", ascending=False).head(5)
    print(top.to_string(index=False))
    print()
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
