"""
Inferred domain quality proxy estimation for Problem 1.

Two methods:
  A) RegMix inversion: partial correlation between each domain's mixture share
     and macro Loss (controlling for all other domains) on train_1m.
     Negative partial corr => increasing that domain lowers Loss => higher implied quality.
  B) Semantic nearest-neighbor mapping from A16 domain_mapping_guide.csv.

Then re-run the quality-enhanced mixture ablation with the completed 17-domain Q
vector to check whether quality_features_adopted flips to true.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT.parent / "F题_清洗后"
REGMIX = DATA / "A_data_value" / "regmix_tables"
MAPPING = DATA / "A_data_value" / "domain_mapping_guide.csv"
QUAL_SUMMARY = PROJECT / "outputs" / "problem1" / "quality" / "results" / "quality_domain_summary.csv"
OUT = PROJECT / "outputs" / "problem1" / "inferred_quality"
OUT.mkdir(parents=True, exist_ok=True)

# Known quality medians from A1 (direct/near_direct only)
Q_KNOWN = {
    "arxiv": 0.7819042123798114,
    "stackexchange": 0.7145557992598962,
    "pile_cc": 0.7067081635779496,
    "wikipedia_en": 0.6215107447610511,
    "github": 0.5768577193890864,
    "gutenberg_pg_19": 0.5693637447993978,
}
Q_GLOBAL_MEAN = 0.6047

# Semantic nearest-neighbor mapping for the 11 inferred domains
# Based on A16 notes + Pile domain semantics
Q_SEMANTIC = {
    "freelaw": 0.569,             # legal books -> book-like
    "pubmed_central": 0.782,      # academic -> arxiv-like
    "pubmed_abstracts": 0.782,    # academic -> arxiv-like
    "dm_mathematics": 0.782,      # academic math -> arxiv-like
    "philpapers": 0.782,          # academic philosophy -> arxiv-like
    "nih_exporter": 0.782,        # medical academic -> arxiv-like
    "uspto_backgrounds": 0.707,   # technical docs -> commoncrawl-like
    "europarl": 0.622,            # formal multilingual -> wikipedia-like
    "enron_emails": 0.622,        # emails -> wikipedia/structured text
    "ubuntu_irc": 0.577,          # community chat -> github-like
    "hackernews": 0.577,          # community discussion -> github-like
}


def load(name: str) -> pd.DataFrame:
    prefix = name.split("_")[0]
    tag = name.split("_", 1)[1]
    mix = pd.read_csv(REGMIX / f"{prefix}_mixture_{tag}.csv")
    loss = pd.read_csv(REGMIX / f"{prefix}_pile_loss_{tag}.csv")
    return mix.merge(loss, on="index", validate="one_to_one")


def mix_cols(df):
    return [c for c in df.columns if c.startswith("train_the_pile_")]


def loss_cols(df):
    return [c for c in df.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]


def domain(c):
    return c.removeprefix("train_the_pile_").removeprefix("metric/the_pile_").removesuffix("_val_loss")


def partial_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Partial correlation of each column of X with y, controlling for the rest."""
    n, k = X.shape
    pcorr = np.zeros(k)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        # Regress y on others
        beta_y = np.linalg.lstsq(X[:, others], y, rcond=None)[0]
        resid_y = y - X[:, others] @ beta_y
        # Regress X[:, j] on others
        beta_x = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)[0]
        resid_x = X[:, j] - X[:, others] @ beta_x
        denom = (np.linalg.norm(resid_y) * np.linalg.norm(resid_x))
        if denom < 1e-12:
            pcorr[j] = 0.0
        else:
            pcorr[j] = float(np.dot(resid_y, resid_x) / denom)
    return pcorr


def normalize_to_quality(pcorr: np.ndarray, domain_list: list[str],
                         known_mask: np.ndarray) -> np.ndarray:
    """Map partial correlations to [0,1] quality proxy.
    Negative pcorr (domain lowers Loss) -> high quality.
    Use min-max within inferred domains only; known domains keep their real Q."""
    q = np.zeros(len(domain_list))
    inferred_idx = np.where(~known_mask)[0]
    if len(inferred_idx) == 0:
        return q
    pc = pcorr[inferred_idx]
    # Invert: more negative => higher quality
    pc_inv = -pc
    lo, hi = pc_inv.min(), pc_inv.max()
    if hi - lo < 1e-9:
        q[inferred_idx] = Q_GLOBAL_MEAN
    else:
        # Scale to [0.4, 0.8] range (reasonable for inferred quality)
        q[inferred_idx] = 0.4 + 0.4 * (pc_inv - lo) / (hi - lo)
    return q


def main():
    train = load("train_1m")
    test_1m = load("test_1m")
    mcols = mix_cols(train)
    lcols = loss_cols(train)
    domains = [domain(c) for c in mcols]
    known_mask = np.array([d in Q_KNOWN for d in domains])

    X_train = train[mcols].to_numpy()
    y_train = train[lcols].mean(axis=1).to_numpy()
    X_test = test_1m[mcols].to_numpy()
    y_test = test_1m[lcols].mean(axis=1).to_numpy()

    # ---- Method A: partial correlation inversion ----
    pcorr = partial_correlations(X_train, y_train)
    q_method_a = normalize_to_quality(pcorr, domains, known_mask)
    # Known domains override
    for i, d in enumerate(domains):
        if d in Q_KNOWN:
            q_method_a[i] = Q_KNOWN[d]

    # ---- Method B: semantic mapping ----
    q_method_b = np.array([Q_KNOWN.get(d, Q_SEMANTIC.get(d, Q_GLOBAL_MEAN)) for d in domains])

    # ---- Combined: average of A and B for inferred domains ----
    q_combined = q_method_a.copy()
    inferred_idx = np.where(~known_mask)[0]
    q_combined[inferred_idx] = 0.5 * q_method_a[inferred_idx] + 0.5 * q_method_b[inferred_idx]

    proxy_df = pd.DataFrame({
        "domain": domains,
        "mapping_type": ["direct/near_direct" if k else "inferred" for k in known_mask],
        "partial_corr_to_loss": pcorr,
        "q_method_a_inversion": q_method_a,
        "q_method_b_semantic": q_method_b,
        "q_combined": q_combined,
    })
    proxy_df.to_csv(OUT / "inferred_quality_proxy.csv", index=False)

    # ---- Re-run quality-enhanced ablation with completed Q ----
    # Build domain-level quality feature for each recipe: weighted average Q
    qvec = q_combined
    q_train = X_train @ qvec
    q_test = X_test @ qvec

    # Model 1: quadratic ILR ridge (baseline, no Q)
    from scipy.linalg import helmert
    H = helmert(17, full=False)  # (16, 17)
    def ilr(P):
        P = np.maximum(P, 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        return np.log(P) @ H.T

    Xtr_ilr = ilr(X_train)
    Xte_ilr = ilr(X_test)

    # Standardize target
    y_scaler = StandardScaler()
    y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_te_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Baseline: quadratic ILR ridge
    from sklearn.preprocessing import PolynomialFeatures
    base = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0))
    base.fit(Xtr_ilr, y_tr_s)
    pred_base = y_scaler.inverse_transform(base.predict(Xte_ilr).reshape(-1, 1)).ravel()
    rmse_base = float(np.sqrt(np.mean((pred_base - y_test) ** 2)))
    sp_base = spearmanr(pred_base, y_test).correlation

    # Quality-enhanced: quadratic ILR + Q feature
    q_scaler = StandardScaler()
    q_tr_s = q_scaler.fit_transform(q_train.reshape(-1, 1))
    q_te_s = q_scaler.transform(q_test.reshape(-1, 1))
    Xtr_aug = np.hstack([Xtr_ilr, q_tr_s])
    Xte_aug = np.hstack([Xte_ilr, q_te_s])
    aug = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0))
    aug.fit(Xtr_aug, y_tr_s)
    pred_aug = y_scaler.inverse_transform(aug.predict(Xte_aug).reshape(-1, 1)).ravel()
    rmse_aug = float(np.sqrt(np.mean((pred_aug - y_test) ** 2)))
    sp_aug = spearmanr(pred_aug, y_test).correlation

    ablation = {
        "baseline_quadratic_ilr_ridge": {"test_1m_rmse": rmse_base, "test_1m_spearman": float(sp_base)},
        "quality_enhanced_completed_q": {"test_1m_rmse": rmse_aug, "test_1m_spearman": float(sp_aug)},
        "rmse_improvement_pct": (rmse_base - rmse_aug) / rmse_base * 100,
        "quality_adopted_threshold_2pct": (rmse_base - rmse_aug) / rmse_base >= 0.02,
    }
    (OUT / "quality_ablation_completed_q.json").write_text(json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Inferred Domain Quality Proxy ===")
    print(proxy_df.to_string(index=False))
    print()
    print("=== Quality-Enhanced Ablation (completed 17-domain Q) ===")
    print(json.dumps(ablation, indent=2))
    print()
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
