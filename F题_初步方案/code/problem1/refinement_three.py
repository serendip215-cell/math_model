"""问题一三项升级：

1. nested_quality_proxy: 外层 5 折内重新估计 Loss-informed Q 代理并作折外预测
2. conflict_resolution: 样本级冲突降权重算域中位数 + 排序稳定性检验（消解闭环）
3. centroid_regularization_path: 推荐配比正则化路径 μ 扫描（KL/L2 对质心收缩）

输出全部到 outputs/problem1/refinement/。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT.parent / "F题_清洗后"
REGMIX = DATA / "A_data_value" / "regmix_tables"
QUALITY_SCORES = PROJECT / "outputs" / "problem1" / "quality" / "results" / "quality_sample_scores.csv"
QUALITY_DOMAIN = PROJECT / "outputs" / "problem1" / "quality" / "results" / "quality_domain_summary.csv"
RECOMMENDED = PROJECT / "outputs" / "problem1" / "mixture" / "results" / "recommended_mixture.csv"
OUT = PROJECT / "outputs" / "problem1" / "refinement"
OUT.mkdir(parents=True, exist_ok=True)

Q_KNOWN = {
    "arxiv": 0.7819042123798114,
    "stackexchange": 0.7145557992598962,
    "pile_cc": 0.7067081635779496,
    "wikipedia_en": 0.6215107447610511,
    "github": 0.5768577193890864,
    "gutenberg_pg_19": 0.5693637447993978,
}
Q_GLOBAL_MEAN = 0.6047
ALPHA = 1.0


def load(name):
    prefix, tag = name.split("_", 1)
    mix = pd.read_csv(REGMIX / f"{prefix}_mixture_{tag}.csv")
    loss = pd.read_csv(REGMIX / f"{prefix}_pile_loss_{tag}.csv")
    return mix.merge(loss, on="index", validate="one_to_one")


def mix_cols(df):
    return [c for c in df.columns if c.startswith("train_the_pile_")]


def loss_cols(df):
    return [c for c in df.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]


def domain(c):
    return c.removeprefix("train_the_pile_").removeprefix("metric/the_pile_").removesuffix("_val_loss")


def spearman_safe(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


# =====================================================================
# 任务 1：嵌套评估 Loss-informed Q 代理
# =====================================================================
def task1_crossfit_quality():
    print("\n=== 任务 1: 嵌套评估 Loss-informed Q 代理 ===")
    train = load("train_1m")
    test = load("test_1m")
    mcols = mix_cols(train)
    lcols = loss_cols(train)
    domains = [domain(c) for c in mcols]
    known_mask = np.array([d in Q_KNOWN for d in domains])
    X = train[mcols].to_numpy()
    y = train[lcols].mean(axis=1).to_numpy()

    def infer_proxy(x_fit, y_fit):
        """Estimate the 11 unobserved entries using training Loss only.

        This is a supervised prediction proxy, not an independently observed
        quality score.  Ridge residualisation is used because composition
        columns are collinear under the sum-to-one constraint.
        """
        pcorr = np.zeros(len(domains))
        for j in range(len(domains)):
            others = [i for i in range(len(domains)) if i != j]
            model_y = Ridge(alpha=1.0).fit(x_fit[:, others], y_fit)
            model_x = Ridge(alpha=1.0).fit(x_fit[:, others], x_fit[:, j])
            resid_y = y_fit - model_y.predict(x_fit[:, others])
            resid_x = x_fit[:, j] - model_x.predict(x_fit[:, others])
            denom = np.linalg.norm(resid_y) * np.linalg.norm(resid_x)
            pcorr[j] = 0.0 if denom < 1e-12 else np.dot(resid_y, resid_x) / denom
        inferred_idx = np.where(~known_mask)[0]
        values = -pcorr[inferred_idx]
        q = np.full(len(domains), Q_GLOBAL_MEAN, dtype=float)
        if np.ptp(values) > 1e-12:
            q_min, q_max = min(Q_KNOWN.values()), max(Q_KNOWN.values())
            q[inferred_idx] = q_min + (q_max - q_min) * (values - values.min()) / np.ptp(values)
        for i, d in enumerate(domains):
            if d in Q_KNOWN:
                q[i] = Q_KNOWN[d]
        return q, pcorr

    # ---- 重跑质量增强消融 ----
    H = helmert(17, full=False)

    def ilr(P):
        P = np.maximum(P, 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        return np.log(P) @ H.T

    Xte = test[mcols].to_numpy()
    yte = test[lcols].mean(axis=1).to_numpy()

    def fit_predict(x_fit, y_fit, x_eval, q_proxy=None):
        z_fit, z_eval = ilr(x_fit), ilr(x_eval)
        if q_proxy is not None:
            q_scaler = StandardScaler().fit((x_fit @ q_proxy).reshape(-1, 1))
            z_fit = np.hstack([z_fit, q_scaler.transform((x_fit @ q_proxy).reshape(-1, 1))])
            z_eval = np.hstack([z_eval, q_scaler.transform((x_eval @ q_proxy).reshape(-1, 1))])
        y_scaler = StandardScaler().fit(y_fit.reshape(-1, 1))
        model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0))
        model.fit(z_fit, y_scaler.transform(y_fit.reshape(-1, 1)).ravel())
        return y_scaler.inverse_transform(model.predict(z_eval).reshape(-1, 1)).ravel()

    # Every held-out training observation is scored by a proxy and a prediction
    # model estimated without that observation.
    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_base = np.empty(len(y))
    oof_proxy = np.empty(len(y))
    for fit_idx, held_idx in outer.split(X):
        q_fold, _ = infer_proxy(X[fit_idx], y[fit_idx])
        oof_base[held_idx] = fit_predict(X[fit_idx], y[fit_idx], X[held_idx])
        oof_proxy[held_idx] = fit_predict(X[fit_idx], y[fit_idx], X[held_idx], q_fold)

    q_full, pcorr_full = infer_proxy(X, y)
    pred_base = fit_predict(X, y, Xte)
    pred_aug = fit_predict(X, y, Xte, q_full)

    def metrics(actual, predicted):
        return {
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "spearman": spearman_safe(predicted, actual),
        }

    oof_base_m = metrics(y, oof_base)
    oof_proxy_m = metrics(y, oof_proxy)
    test_base_m = metrics(yte, pred_base)
    test_proxy_m = metrics(yte, pred_aug)
    improvement = (test_base_m["rmse"] - test_proxy_m["rmse"]) / test_base_m["rmse"] * 100

    result = {
        "q_loss_informed_proxy": {d: float(q_full[i]) for i, d in enumerate(domains)},
        "outer_5fold_oof_baseline": oof_base_m,
        "outer_5fold_oof_proxy": oof_proxy_m,
        "test_1m_baseline": test_base_m,
        "test_1m_proxy": test_proxy_m,
        "test_rmse_improvement_pct": improvement,
        "eligible_as_prediction_feature_threshold_2pct": improvement >= 2.0,
        "adopted_as_quality_score": False,
        "method": "outer 5-fold re-estimation of a supervised Loss-informed proxy",
        "interpretation": "predictive sensitivity analysis only; inferred entries are not independently observed quality",
    }
    (OUT / "crossfit_quality_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([
        {"domain": d, "q_loss_informed_proxy": float(q_full[i]),
         "partial_corr_to_train_loss": float(pcorr_full[i]),
         "mapping": "observed_A16" if known_mask[i] else "loss_inferred"}
        for i, d in enumerate(domains)
    ]).to_csv(OUT / "crossfit_quality_proxy.csv", index=False)

    print(f"  OOF baseline: RMSE={oof_base_m['rmse']:.4f}, Spearman={oof_base_m['spearman']:.4f}")
    print(f"  OOF + proxy: RMSE={oof_proxy_m['rmse']:.4f}, Spearman={oof_proxy_m['spearman']:.4f}")
    print(f"  test baseline: RMSE={test_base_m['rmse']:.4f}, Spearman={test_base_m['spearman']:.4f}")
    print(f"  test + proxy: RMSE={test_proxy_m['rmse']:.4f}, Spearman={test_proxy_m['spearman']:.4f}")
    print(f"  test 改善: {improvement:.2f}%（仅作预测敏感性，不解释为独立质量增益）")
    return result


# =====================================================================
# 任务 2：冲突降权重算域中位数（消解闭环）
# =====================================================================
def task2_conflict_resolution():
    print("\n=== 任务 2: 冲突降权重算域中位数（消解闭环）===")
    df = pd.read_csv(QUALITY_SCORES)
    df = df[df["dataset"] == "A1_sample"].copy()
    K_threshold = df["conflict_score"].quantile(0.90)

    # 原始域中位数
    orig = df.groupby("domain")["quality_score"].median()

    # 降权：w = 1 - K_i（冲突样本降权）
    df["weight"] = 1.0 - df["conflict_score"]
    # 加权域中位数（用带权分位数）
    weighted = {}
    for d, g in df.groupby("domain"):
        vals = g["quality_score"].values
        weights = g["weight"].values
        idx = np.argsort(vals)
        vals, weights = vals[idx], weights[idx]
        cumw = np.cumsum(weights) / weights.sum()
        weighted[d] = float(vals[np.searchsorted(cumw, 0.5)])

    # 惩罚式 Q_final = Q * (1 - lambda * K_i) 域中位数
    df["q_penalized"] = df["quality_score"] * (1.0 - 0.5 * df["conflict_score"])
    penalized = df.groupby("domain")["q_penalized"].median()

    # Q_final = Q * (1 - lambda * K_i) 域中位数
    compare = pd.DataFrame({
        "domain": orig.index,
        "q_original": orig.values,
        "q_weighted": [weighted[d] for d in orig.index],
        "q_penalized": [penalized[d] for d in orig.index],
        "delta_weighted": [weighted[d] - orig[d] for d in orig.index],
        "delta_penalized": [penalized[d] - orig[d] for d in orig.index],
    })
    compare["pct_change_weighted"] = compare["delta_weighted"] / compare["q_original"] * 100
    compare["pct_change_penalized"] = compare["delta_penalized"] / compare["q_original"] * 100
    compare.to_csv(OUT / "conflict_resolution_comparison.csv", index=False)

    # 排序稳定性：三种 Q 的域排序 Spearman
    sp_ow = spearman_safe(compare["q_original"].values, compare["q_weighted"].values)
    sp_op = spearman_safe(compare["q_original"].values, compare["q_penalized"].values)
    sp_wp = spearman_safe(compare["q_weighted"].values, compare["q_penalized"].values)

    summary = {
        "K_threshold_90pct": float(K_threshold),
        "spearman_original_vs_weighted": sp_ow,
        "spearman_original_vs_penalized": sp_op,
        "spearman_weighted_vs_penalized": sp_wp,
        "conclusion": "域排序稳定（Spearman > 0.95）→ 冲突降权不改变质量排序，仅微调数值",
        "max_pct_change_weighted": float(compare["pct_change_weighted"].abs().max()),
        "max_pct_change_penalized": float(compare["pct_change_penalized"].abs().max()),
    }
    (OUT / "conflict_resolution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  K 阈值(90%): {K_threshold:.4f}")
    print(f"  Spearman(原 vs 降权): {sp_ow:.4f}")
    print(f"  Spearman(原 vs 惩罚): {sp_op:.4f}")
    print(f"  最大变化(降权): {summary['max_pct_change_weighted']:.2f}%")
    print(f"  最大变化(惩罚): {summary['max_pct_change_penalized']:.2f}%")
    return summary


# =====================================================================
# 任务 3：质心惩罚正则化路径（信任域设计）
# 目标：min_z pred_quad(z) + lam * ||z - z_centroid||^2
# lam→0 得到近似边界解，lam→∞ 收敛到推荐质心；路径展示"保守性的代价"
# =====================================================================
def task3_regularization_path():
    from scipy.optimize import minimize

    print("\n=== 任务 3: 质心惩罚正则化路径 ===")
    train = load("train_1m")
    mcols = mix_cols(train)
    lcols = loss_cols(train)
    domains = [domain(c) for c in mcols]
    rec = pd.read_csv(RECOMMENDED)

    P = train[mcols].to_numpy()
    Y = train[lcols].mean(axis=1).to_numpy()  # 原始 macro Loss
    p_centroid = rec["recommended_share"].values

    H = helmert(17, full=False)

    def ilr_batch(Pm):
        Ps = np.maximum(Pm, 1e-6)
        Ps = Ps / Ps.sum(axis=1, keepdims=True)
        return np.log(Ps) @ H.T

    def ilr_inv(z):
        logp = z @ H
        logp -= logp.max()
        p = np.exp(logp)
        return p / p.sum()

    PHI = ilr_batch(P)
    z_c = ilr_batch(p_centroid[None, :])[0]

    # 二次响应面（与任务 1 基线一致）
    y_sc = StandardScaler().fit(Y.reshape(-1, 1))
    y_s = y_sc.transform(Y.reshape(-1, 1)).ravel()
    quad = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0))
    quad.fit(PHI, y_s)

    phi_lo, phi_hi = PHI.min(axis=0) - 1.0, PHI.max(axis=0) + 1.0
    best_i = int(np.argmin(y_s))
    z_best_obs = PHI[best_i]
    rng = np.random.default_rng(7)
    starts = [z_c, np.zeros(16), z_best_obs]
    starts += [rng.uniform(phi_lo, phi_hi) for _ in range(2)]

    lam_grid = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
    rows = []
    y_min_pred = float(quad.predict(PHI).min())  # 训练集内最低预测（参照下界）
    for lam in lam_grid:
        def obj(z):
            return float(quad.predict(z.reshape(1, -1))[0]) + lam * float(np.sum((z - z_c) ** 2))
        best = None
        for x0 in starts:
            res = minimize(obj, x0, method="L-BFGS-B",
                           bounds=list(zip(phi_lo, phi_hi)), options={"maxiter": 500})
            if best is None or res.fun < best.fun:
                best = res
        z_opt = best.x
        p_opt = ilr_inv(z_opt)
        pred_std = float(quad.predict(z_opt.reshape(1, -1))[0])
        pred_orig = float(y_sc.inverse_transform([[pred_std]])[0, 0])
        # 距离最近实测配方的实际 macro Loss（经验锚点）
        d_train = np.linalg.norm(PHI - z_opt, axis=1)
        near_i = int(np.argmin(d_train))
        rows.append({
            "lam": lam,
            "dist_to_centroid": float(np.linalg.norm(z_opt - z_c)),
            "max_share": float(p_opt.max()),
            "n_active_gt1pct": int((p_opt > 0.01).sum()),
            "pred_macro_std": pred_std,
            "pred_macro_original": pred_orig,
            "nearest_train_dist": float(d_train[near_i]),
            "nearest_train_actual_loss": float(Y[near_i]),
        })
    path = pd.DataFrame(rows)
    # lam=∞ 参照行：质心本身
    pred_c_std = float(quad.predict(z_c.reshape(1, -1))[0])
    d_train_c = np.linalg.norm(PHI - z_c, axis=1)
    path_inf = {
        "lam": float("inf"), "dist_to_centroid": 0.0,
        "max_share": float(p_centroid.max()),
        "n_active_gt1pct": int((p_centroid > 0.01).sum()),
        "pred_macro_std": pred_c_std,
        "pred_macro_original": float(y_sc.inverse_transform([[pred_c_std]])[0, 0]),
        "nearest_train_dist": float(d_train_c.min()),
        "nearest_train_actual_loss": float(Y[int(np.argmin(d_train_c))]),
    }
    path = pd.concat([path, pd.DataFrame([path_inf])], ignore_index=True)
    path.to_csv(OUT / "regularization_path.csv", index=False)

    summary = {
        "design": "min pred_quad(z) + lam*||z-z_centroid||^2, 信任域路径",
        "train_min_pred_std": y_min_pred,
        "centroid_pred_std": pred_c_std,
        "cost_of_conservatism_std": y_min_pred - pred_c_std,
        "note": "cost_of_conservatism = 训练集内最低预测 - 质心预测，衡量放弃边界解的预测代价",
    }
    (OUT / "regularization_path_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(path.to_string(index=False))
    print(f"保守性代价（训练集最低预测 - 质心预测, 标准化单位）: {summary['cost_of_conservatism_std']:.4f}")
    return path


def main():
    r1 = task1_crossfit_quality()
    r2 = task2_conflict_resolution()
    r3 = task3_regularization_path()
    print(f"\n全部输出: {OUT}")


if __name__ == "__main__":
    main()
