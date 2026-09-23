"""
Scaling Law bridge for Problem 1.

Goal: explain why the 1M response surface fails to extrapolate to 10B/70B
(Spearman turns negative) by fitting a Chinchilla-style scaling law that
uses *effective data* D_eff = D_scale * sum(p_i * Q_i) instead of raw D.

Fit only on observed scales (train_1m, test_60m, test_1B).
Use est_10b / est_70b purely as external test (no fitting).
Output: predicted vs actual Loss, Spearman per scale, fit parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.linalg import helmert

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT.parent / "F题_清洗后"
REGMIX = DATA / "A_data_value" / "regmix_tables"
OUT = PROJECT / "outputs" / "problem1" / "scaling_bridge"
OUT.mkdir(parents=True, exist_ok=True)

# Total training tokens per scale (from problem statement scale labels)
SCALE_TOKENS = {
    "train_1m": 1.0e6,
    "test_1m": 1.0e6,
    "test_60m": 6.0e7,
    "test_1B": 1.0e9,
    "est_10b": 1.0e10,
    "est_70b": 7.0e10,
}

# Known domain quality medians (from quality_domain_summary.csv, A1)
# 6 direct/near_direct domains; the rest use global mean as placeholder
Q_KNOWN = {
    "arxiv": 0.7819042123798114,
    "stackexchange": 0.7145557992598962,
    "pile_cc": 0.7067081635779496,       # maps to commoncrawl
    "wikipedia_en": 0.6215107447610511,  # maps to wikipedia
    "github": 0.5768577193890864,
    "gutenberg_pg_19": 0.5693637447993978,  # maps to book
}
Q_GLOBAL_MEAN = 0.6047  # from run_summary quality mean


def load_scale(name: str) -> pd.DataFrame:
    prefix = name.split("_")[0]  # train / test / est
    scale_tag = name.split("_", 1)[1]  # 1m / 60m / 1B / 10b / 70b
    mix_path = REGMIX / f"{prefix}_mixture_{scale_tag}.csv"
    loss_path = REGMIX / f"{prefix}_pile_loss_{scale_tag}.csv"
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    df = mix.merge(loss, on="index", validate="one_to_one")
    return df


def mixture_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("train_the_pile_")]


def loss_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]


def domain_from_col(col: str) -> str:
    return col.removeprefix("train_the_pile_").removeprefix("metric/the_pile_").removesuffix("_val_loss")


def build_q_vector(mix_cols: list[str]) -> np.ndarray:
    q = np.full(len(mix_cols), Q_GLOBAL_MEAN, dtype=float)
    for i, c in enumerate(mix_cols):
        d = domain_from_col(c)
        if d in Q_KNOWN:
            q[i] = Q_KNOWN[d]
    return q


def chinchilla(D_eff, a, c, beta):
    return a + c / np.power(D_eff, beta)


def main():
    records = []
    scale_data = {}
    mix_cols_all = None
    for name in SCALE_TOKENS:
        df = load_scale(name)
        mcols = mixture_cols(df)
        lcols = loss_cols(df)
        if mix_cols_all is None:
            mix_cols_all = mcols
        q = build_q_vector(mcols)
        p = df[mcols].to_numpy()
        D_scale = SCALE_TOKENS[name]
        D_eff = D_scale * (p * q).sum(axis=1)
        macro_loss = df[lcols].mean(axis=1).to_numpy()
        scale_data[name] = {
            "df": df, "D_eff": D_eff, "macro_loss": macro_loss,
            "n": len(df), "q": q, "mcols": mcols, "lcols": lcols,
        }
        for i in range(len(df)):
            records.append({
                "scale": name, "index": int(df["index"].iloc[i]),
                "D_eff": D_eff[i], "macro_loss": macro_loss[i],
                "D_tokens": D_scale,
            })
    raw = pd.DataFrame(records)
    raw.to_csv(OUT / "scaling_raw.csv", index=False)

    # ---- Fit Chinchilla on observed scales only ----
    fit_scales = ["train_1m", "test_60m", "test_1B"]
    D_fit = np.concatenate([scale_data[s]["D_eff"] for s in fit_scales])
    L_fit = np.concatenate([scale_data[s]["macro_loss"] for s in fit_scales])
    # Initial guess: a ~ 2.0 (asymptotic loss), c ~ 1.0, beta ~ 0.1
    popt, pcov = curve_fit(
        chinchilla, D_fit, L_fit,
        p0=[2.0, 1.0, 0.1],
        bounds=([0.0, 0.0, 1e-4], [10.0, 100.0, 2.0]),
        maxfev=20000,
    )
    a_fit, c_fit, beta_fit = popt
    perr = np.sqrt(np.diag(pcov))
    fit_info = {
        "fit_scales": fit_scales,
        "n_fit": int(len(D_fit)),
        "params": {"a": float(a_fit), "c": float(c_fit), "beta": float(beta_fit)},
        "param_std": {"a": float(perr[0]), "c": float(perr[1]), "beta": float(perr[2])},
        "fit_rmse": float(np.sqrt(np.mean((chinchilla(D_fit, *popt) - L_fit) ** 2))),
    }

    # ---- Predict on all scales, compute Spearman ----
    result_rows = []
    for name, sd in scale_data.items():
        pred = chinchilla(sd["D_eff"], *popt)
        actual = sd["macro_loss"]
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        sp = spearmanr(pred, actual)
        result_rows.append({
            "scale": name,
            "n": sd["n"],
            "D_tokens": SCALE_TOKENS[name],
            "observed": name in fit_scales,
            "pred_rmse": rmse,
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
            "mean_actual_loss": float(actual.mean()),
            "mean_pred_loss": float(pred.mean()),
        })
        # Save per-recipe predictions
        pred_df = sd["df"][["index"]].copy()
        pred_df["scale"] = name
        pred_df["D_eff"] = sd["D_eff"]
        pred_df["actual_macro_loss"] = actual
        pred_df["predicted_macro_loss"] = pred
        pred_df["residual"] = pred - actual
        pred_df.to_csv(OUT / f"predictions_{name}.csv", index=False)

    res = pd.DataFrame(result_rows)
    res.to_csv(OUT / "scaling_results.csv", index=False)

    # ---- Combined model: scale baseline + mixture ratio effect ----
    # Step 1: fit baseline(D) = a + c / D^beta using per-scale mean Loss
    scale_means = {s: float(scale_data[s]["macro_loss"].mean()) for s in scale_data}
    D_list = np.array([SCALE_TOKENS[s] for s in fit_scales])
    L_list = np.array([scale_means[s] for s in fit_scales])
    popt_b, _ = curve_fit(
        chinchilla, D_list, L_list,
        p0=[1.5, 5.0, 0.1],
        bounds=([0.0, 0.0, 1e-4], [10.0, 100.0, 2.0]),
        maxfev=20000,
    )
    baseline = {s: float(chinchilla(SCALE_TOKENS[s], *popt_b)) for s in scale_data}

    # Step 2: train ratio model on train_1m (de-meaned Loss), predict relative effect
    H = helmert(17, full=False)  # (16, 17)
    def ilr(P):
        P = np.maximum(P, 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        return np.log(P) @ H.T  # (batch, 17) @ (17, 16) -> (batch, 16)

    mcols_train = scale_data["train_1m"]["mcols"]
    Xtr_ilr = ilr(scale_data["train_1m"]["df"][mcols_train].to_numpy())
    ytr_raw = scale_data["train_1m"]["macro_loss"]
    ytr_centered = ytr_raw - ytr_raw.mean()

    ratio_model = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0)
    )
    ratio_model.fit(Xtr_ilr, ytr_centered)

    # Step 3: combine: pred = baseline(D_scale) + ratio_model(p)
    combined_rows = []
    for name, sd in scale_data.items():
        p = sd["df"][sd["mcols"]].to_numpy()
        phi = ilr(p)
        rel = ratio_model.predict(phi)
        pred = baseline[name] + rel
        actual = sd["macro_loss"]
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        sp = spearmanr(pred, actual)
        combined_rows.append({
            "scale": name,
            "observed": name in fit_scales,
            "baseline_loss": baseline[name],
            "pred_rmse": rmse,
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
        })
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(OUT / "combined_model_results.csv", index=False)

    summary = {
        "fit_info": fit_info,
        "results": result_rows,
        "baseline_fit": {
            "params": {"a": float(popt_b[0]), "c": float(popt_b[1]), "beta": float(popt_b[2])},
            "scale_baselines": baseline,
        },
        "combined_model": combined_rows,
    }
    (OUT / "scaling_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Scaling Law Bridge ===")
    print(f"Fit params: a={a_fit:.4f}, c={c_fit:.4f}, beta={beta_fit:.6f}")
    print(f"Fit RMSE (observed scales): {fit_info['fit_rmse']:.4f}")
    print()
    print(res.to_string(index=False))
    print()
    print("=== Combined Model (scale baseline + ratio effect) ===")
    print(f"Baseline fit: a={popt_b[0]:.4f}, c={popt_b[1]:.4f}, beta={popt_b[2]:.6f}")
    print(combined.to_string(index=False))
    print()
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
