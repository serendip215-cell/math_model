"""Problem 1 enhancement bundle — 6 modules, all driven by real data."""
from __future__ import annotations
import json, hashlib, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr, pearsonr

SEED = 20260923
RNG = np.random.default_rng(SEED)
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT.parent / "F题_清洗后"
A = DATA / "A_data_value"
P1 = PROJECT / "outputs" / "problem1"
QUAL_OUT = P1 / "quality" / "results"
MIX_OUT = P1 / "mixture" / "results"
ENH_OUT = P1 / "enhancement" / "results"
ENH_FIG = P1 / "enhancement" / "figures"
for d in (ENH_OUT, ENH_FIG):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "figure.dpi": 140})


# ============================================================
# Helper
# ============================================================
def config_plots():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "figure.dpi": 140})


def load_pair(name):
    # name examples: train_1m, test_60m, est_10b → files are train_mixture_1m.csv, train_pile_loss_1m.csv
    split, size = name.rsplit("_", 1)
    mix = pd.read_csv(A / "regmix_tables" / f"{split}_mixture_{size}.csv", encoding="utf-8-sig")
    loss = pd.read_csv(A / "regmix_tables" / f"{split}_pile_loss_{size}.csv", encoding="utf-8-sig")
    merged = mix.merge(loss, on="index", how="inner", validate="one_to_one")
    p_cols = [c for c in mix.columns if c.startswith("train_the_pile_")]
    y_cols = [c for c in loss.columns if c.startswith("metric/the_pile_")]
    p = merged[p_cols].to_numpy(dtype=float)
    y = merged[y_cols].to_numpy(dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    return merged, p, y, p_cols, y_cols


# Quality domain → mixture domain mapping (direct/near-direct, per A16)
QUAL_TO_MIX = {
    "arxiv": "arxiv",
    "stackexchange": "stackexchange",
    "wikipedia": "wikipedia_en",
    "github": "github",
    "commoncrawl": "pile_cc",  # near-direct
    "book": "gutenberg_pg_19",  # near-direct, per A16
}
# c4 has no corresponding mixture domain in the 17-domain table.

MIX_DOMAINS = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
               "dm_mathematics", "github", "philpapers", "stackexchange", "enron_emails",
               "gutenberg_pg_19", "pile_cc", "ubuntu_irc", "europarl", "hackernews",
               "pubmed_abstracts", "uspto_backgrounds"]

QUAL_DOMAINS = list(QUAL_TO_MIX.keys())


# ============================================================
# Module 1: External validity anchor — weighted Q vs macro Loss
# ============================================================
def module1_quality_external_validity():
    print("=== Module 1: Quality external validity anchor ===")
    scores = pd.read_csv(QUAL_OUT / "quality_sample_scores.csv", encoding="utf-8-sig")
    a1 = scores[scores["dataset"].eq("A1_sample")].copy()
    # Per-domain quality median (A1)
    q_by_quality = a1.groupby("domain")["quality_score"].median().to_dict()
    # Map → mixture domain
    q_by_mix = {}
    for qd, md in QUAL_TO_MIX.items():
        if qd in q_by_quality:
            q_by_mix[md] = q_by_quality[qd]
    print(f"  Q available for {len(q_by_mix)}/{len(MIX_DOMAINS)} mixture domains: {list(q_by_mix.keys())}")

    # Load train pairs
    train_idx, p_tr, y_tr, p_cols, y_cols = load_pair("train_1m")
    p_names = [c.replace("train_the_pile_", "") for c in p_cols]
    y_actual = y_tr.mean(axis=1)

    # Weighted Q per recipe: Σ p_i * Q_median_i for domains with known Q
    weighted_q = np.zeros(len(p_tr))
    covered_share = np.zeros(len(p_tr))
    for i, pname in enumerate(p_names):
        if pname in q_by_mix:
            weighted_q += p_tr[:, i] * q_by_mix[pname]
            covered_share += p_tr[:, i]

    df = pd.DataFrame({"weighted_quality": weighted_q, "covered_share": covered_share,
                       "macro_loss": y_actual, "n": len(p_tr)})
    # Only recipes with reasonable coverage (>30%)
    mask = df["covered_share"] > 0.30
    r_all, p_all = pearsonr(df["weighted_quality"], df["macro_loss"])
    r_cov, p_cov = pearsonr(df.loc[mask, "weighted_quality"], df.loc[mask, "macro_loss"])
    print(f"  ALL Pearson r(Q, Loss) = {r_all:.4f}, p = {p_all:.3e}")
    print(f"  (coverage>30%) r = {r_cov:.4f}, p = {p_cov:.3e}")

    df_out = pd.DataFrame({
        "domain": p_names,
        "quality_median": [q_by_mix.get(n, np.nan) for n in p_names],
    })
    df_out.to_csv(ENH_OUT / "quality_mix_domain_medians.csv", index=False, encoding="utf-8-sig")

    # Plot
    config_plots()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["weighted_quality"], df["macro_loss"], alpha=0.35, s=14, color="#3b82f6")
    # Fit line
    z = np.polyfit(df["weighted_quality"], df["macro_loss"], 1)
    x_line = np.linspace(df["weighted_quality"].min(), df["weighted_quality"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", lw=1.8,
            label=f"Pearson r={r_all:.4f}, p={p_all:.1e}")
    ax.set_xlabel("加权质量 Σ(p_i × Q_i)")
    ax.set_ylabel("13 域平均 Loss")
    ax.set_title("质量评分外部效度锚点")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ENH_FIG / "quality_vs_macro_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = [{"pair": "train_1m_all", "n": len(df), "pearson_r": r_all, "pearson_p": p_all},
            {"pair": "train_1m_covered>30pct", "n": int(mask.sum()), "pearson_r": r_cov, "pearson_p": p_cov}]

    # Also check test_1m (validation)
    t_idx, p_te, y_te, _, _ = load_pair("test_1m")
    weighted_q_te = np.zeros(len(p_te))
    for i, pname in enumerate(p_names):
        if pname in q_by_mix:
            weighted_q_te += p_te[:, i] * q_by_mix[pname]
    y_te_actual = y_te.mean(axis=1)
    r_te, p_te_ = pearsonr(weighted_q_te, y_te_actual)
    rows.append({"pair": "test_1m", "n": len(p_te), "pearson_r": r_te, "pearson_p": p_te_})
    print(f"  test_1m Pearson r(Q, Loss) = {r_te:.4f}, p = {p_te_:.3e}")

    pd.DataFrame(rows).to_csv(ENH_OUT / "quality_external_validity.csv", index=False, encoding="utf-8-sig")
    print(f"  → saved quality_external_validity.csv")
    return r_all, p_all, r_te, p_te_


# ============================================================
# Module 2: Δφ differential correlation + explicit R²
# ============================================================
def module2_dphi_and_r2():
    print("\n=== Module 2: Δφ differential correlation + explicit R² ===")
    basis = helmert(17)
    results = []
    for split in ("train_1m", "test_1m", "test_60m", "test_1B"):
        merged, p, y, p_cols, y_cols = load_pair(split)
        y_actual = y.mean(axis=1)
        # Actual delta
        d_actual = np.diff(y_actual)
        # ILR delta of mixture (actual)
        eps = 1e-6
        p_safe = np.maximum(p, eps); p_safe = p_safe / p_safe.sum(axis=1, keepdims=True)
        phi = np.log(p_safe) @ basis.T   # (n, 16)
        d_phi = np.diff(phi, axis=0)      # (n-1, 16)
        # Correlate mean |Δφ| (norm) with |ΔLoss|
        dphi_norm = np.linalg.norm(d_phi, axis=1)
        r_diff, p_diff = pearsonr(d_actual, dphi_norm)
        r_diff_abs, _ = pearsonr(np.abs(d_actual), dphi_norm)

        # Predicted macro loss — reuse predicted from mixture_predictions.csv if exists
        pred_file = MIX_OUT.parent / "mixture" / "results" / "mixture_predictions.csv"
        if pred_file.exists():
            preds = pd.read_csv(pred_file, encoding="utf-8-sig")
            sub = preds[preds["split"] == split].copy()
            if len(sub) == len(y_actual):
                y_pred = sub["predicted_macro_loss"].to_numpy(dtype=float)
                d_pred = np.diff(y_pred)
                r_pred_diff, _ = pearsonr(d_actual, d_pred)
            else:
                r_pred_diff = np.nan
        else:
            r_pred_diff = np.nan

        results.append({"split": split, "n": len(y_actual),
                        "pearson_dLoss_vs_dphi": float(r_diff),
                        "pearson_abs_dLoss_vs_dphi_norm": float(r_diff_abs),
                        "pearson_dLoss_vs_dPred": float(r_pred_diff)})
        print(f"  {split}: ΔL vs Δφ r = {r_diff:.4f}, |ΔL| vs ‖Δφ‖ r = {r_diff_abs:.4f}, "
              f"ΔL vs ΔPred r = {r_pred_diff:.4f}")

    # Explicit R² on the ORIGINAL-unit macro Loss: R² = 1 − MSE/Var(y),
    # with Var estimated from the raw data of each split (ddof=1).
    # NOTE: R² = 1 − RMSE² is only valid when Var(y)=1 (standardized target).
    # The RMSE columns in mixture_model_metrics.csv are in ORIGINAL units
    # (test_1m macro Loss variance ≈ 0.077), so feeding them into 1−RMSE²
    # severely overstates R² (e.g. test_1m 0.946 vs true 0.302). The
    # wrong_r2_1minus_rmse2 column is kept deliberately as a record of that
    # earlier mistake; it must never be quoted as a result.
    metrics = pd.read_csv(MIX_OUT / "mixture_model_metrics.csv", encoding="utf-8-sig")
    rows = []
    for _, r in metrics.iterrows():
        if r["target"] == "macro_average":
            split = r["split"]
            _, _, y_sp, _, _ = load_pair(split)
            macro_y = y_sp.mean(axis=1)
            # Standard coefficient of determination uses SSE/SST, which is
            # equivalent to MSE divided by the population mean squared
            # deviation (ddof=0), not the sample variance (ddof=1).
            var_y = float(macro_y.var(ddof=0))
            rmse_abs = float(r["rmse"])
            rmse_centered = float(r["centered_rmse"])
            rows.append({"split": split,
                         "rmse_raw": rmse_abs,
                         "centered_rmse": rmse_centered,
                         "var_macro_loss": var_y,
                         "true_r2_raw": 1 - rmse_abs ** 2 / var_y,
                         "true_r2_centered": 1 - rmse_centered ** 2 / var_y,
                         "wrong_r2_1minus_rmse2": 1 - rmse_abs ** 2})
    r2_df = pd.DataFrame(rows)
    print("\n  Explicit R² on raw-unit macro Loss (R² = 1 − SSE/SST; Var uses ddof=0):")
    print(r2_df.to_string(index=False))
    r2_df.to_csv(ENH_OUT / "explicit_r2.csv", index=False, encoding="utf-8-sig")
    r2_df.to_csv(ENH_OUT / "corrected_r2.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(ENH_OUT / "dphi_differential_correlation.csv", index=False, encoding="utf-8-sig")
    print(f"  → saved dphi_differential_correlation.csv, explicit_r2.csv")


# ============================================================
# Module 3: Ablation — remove a domain and refit
# ============================================================
def module3_ablation():
    print("\n=== Module 3: Ablation by removing domain ===")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    basis = helmert(17)
    train_idx, p_tr, y_tr, p_cols, y_cols = load_pair("train_1m")
    test_idx, p_te, y_te, _, _ = load_pair("test_1m")

    # Normalize targets by z-score (per Loss domain)
    y_mean = y_tr.mean(axis=0); y_std = y_tr.std(axis=0)
    y_tr_s = (y_tr - y_mean) / y_std
    y_te_s = (y_te - y_mean) / y_std

    def fit_eval(p_tr_sel, y_tr_sel, p_te_sel, y_te_sel, label="quadratic"):
        eps = 1e-6
        p_tr_sel = np.maximum(p_tr_sel, eps); p_tr_sel = p_tr_sel / p_tr_sel.sum(axis=1, keepdims=True)
        p_te_sel = np.maximum(p_te_sel, eps); p_te_sel = p_te_sel / p_te_sel.sum(axis=1, keepdims=True)
        z_tr = np.log(p_tr_sel) @ basis.T   # (512, 16)
        z_te = np.log(p_te_sel) @ basis.T
        macro_tr = y_tr_sel.mean(axis=1); macro_te = y_te_sel.mean(axis=1)
        # Per-domain fit → predict each of 13 Loss domains, then average
        preds = np.zeros_like(y_te_sel)
        for j in range(y_te_sel.shape[1]):
            target_j = y_tr_sel[:, j] - macro_tr  # Use centered target per domain? No — full target is already standardized
            # Actually fit each domain independently: target_j = y_tr_sel[:, j]
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
            model = Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge()),
            ])
            grid = GridSearchCV(model, {"ridge__alpha": alphas}, cv=cv, scoring="neg_mean_squared_error")
            grid.fit(z_tr, y_tr_sel[:, j])
            preds[:, j] = grid.predict(z_te)
        macro_pred_te = preds.mean(axis=1)
        macro_actual_te = y_te_sel.mean(axis=1)
        rmse = float(np.sqrt(np.mean((macro_pred_te - macro_actual_te) ** 2)))
        rho = float(spearmanr(macro_actual_te, macro_pred_te).statistic)
        return rmse, rho

    # Baseline (all 17)
    rmse_all, rho_all = fit_eval(p_tr, y_tr_s, p_te, y_te_s)
    print(f"  Baseline (all 17 domains): RMSE = {rmse_all:.4f}, Spearman = {rho_all:.4f}")

    remove_list = ["pile_cc", "wikipedia_en", "github", "arxiv", "stackexchange"]
    rows = [{"removed": "(none, baseline)", "rmse": rmse_all, "spearman": rho_all,
             "delta_rmse_pct": 0.0, "delta_spearman": 0.0}]
    for rm in remove_list:
        mask = [not c.replace("train_the_pile_", "").endswith("_" + rm)
                if not c.endswith(rm) else True for c in p_cols]
        # Actually simpler: find indices to keep
        idx_remove = [i for i, c in enumerate(p_cols)
                      if c.replace("train_the_pile_", "") == rm]
        idx_keep = [i for i in range(len(p_cols)) if i not in idx_remove]
        if len(idx_remove) == 0:
            print(f"  WARNING: {rm} not found in p_cols")
            continue
        p_tr_sub = p_tr[:, idx_keep]; p_te_sub = p_te[:, idx_keep]
        # Remaining 16 domains — refit ILR on 16 dims → use 15-dim Helmert
        basis_sub = helmert(16)
        eps = 1e-6
        p_tr_s = np.maximum(p_tr_sub, eps); p_tr_s = p_tr_s / p_tr_s.sum(axis=1, keepdims=True)
        p_te_s = np.maximum(p_te_sub, eps); p_te_s = p_te_s / p_te_s.sum(axis=1, keepdims=True)
        z_tr = np.log(p_tr_s) @ basis_sub.T
        z_te = np.log(p_te_s) @ basis_sub.T
        preds = np.zeros_like(y_te_s)
        for j in range(y_te_s.shape[1]):
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
            model = Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge()),
            ])
            grid = GridSearchCV(model, {"ridge__alpha": alphas}, cv=cv, scoring="neg_mean_squared_error")
            grid.fit(z_tr, y_tr_s[:, j])
            preds[:, j] = grid.predict(z_te)
        macro_pred = preds.mean(axis=1)
        macro_actual = y_te_s.mean(axis=1)
        rmse = float(np.sqrt(np.mean((macro_pred - macro_actual) ** 2)))
        rho = float(spearmanr(macro_actual, macro_pred).statistic)
        delta_rmse = (rmse - rmse_all) / rmse_all * 100
        delta_rho = rho - rho_all
        rows.append({"removed": rm, "rmse": rmse, "spearman": rho,
                     "delta_rmse_pct": delta_rmse, "delta_spearman": delta_rho})
        print(f"  remove {rm}: RMSE = {rmse:.4f} ({delta_rmse:+.2f}%), Spearman = {rho:.4f} ({delta_rho:+.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(ENH_OUT / "domain_ablation.csv", index=False, encoding="utf-8-sig")

    config_plots()
    fig, ax = plt.subplots(figsize=(7, 4))
    names = out["removed"].tolist()
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, out["rmse"], width, label="test_1m macro RMSE", color="#3b82f6")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, out["spearman"], width, label="Spearman", color="#f59e0b")
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("Spearman ρ")
    ax.set_title("域级消融：去掉指定域后 1M 检验集指标变化")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ENH_FIG / "domain_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved domain_ablation.csv, domain_ablation.pdf")


# ============================================================
# Module 4: Cross-scale recommendation (1M / 60M / 1B)
# ============================================================
def module4_cross_scale_recommendation():
    print("\n=== Module 4: Cross-scale recommendation ===")
    scales = ["train_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
    rows_all = []
    p_cols_names = None
    for scale in scales:
        merged, p, y, p_cols, y_cols = load_pair(scale)
        if p_cols_names is None:
            p_cols_names = [c.replace("train_the_pile_", "") for c in p_cols]
        macro_actual = y.mean(axis=1)
        # Best top 10% (by actual macro Loss, lower is better)
        idx_top = np.argsort(macro_actual)[: max(1, int(len(macro_actual) * 0.10))]
        centroid = p[idx_top].mean(axis=0)
        # Bootstrap CI on centroid
        n_boot = 200
        boot = np.empty((n_boot, p.shape[1]))
        for b in range(n_boot):
            sel = RNG.choice(idx_top, size=len(idx_top), replace=True)
            boot[b] = p[sel].mean(axis=0)
        ci_lo = np.percentile(boot, 2.5, axis=0)
        ci_hi = np.percentile(boot, 97.5, axis=0)
        for i, name in enumerate(p_cols_names):
            rows_all.append({"scale": scale, "domain": name,
                             "centroid": float(centroid[i]),
                             "ci_low": float(ci_lo[i]), "ci_high": float(ci_hi[i]),
                             "n_top": int(len(idx_top))})
    rec_df = pd.DataFrame(rows_all)
    rec_df.to_csv(ENH_OUT / "cross_scale_recommendation.csv", index=False, encoding="utf-8-sig")

    # Plot: stacked bar for the top-5 domains across scales
    p1_top5 = rec_df[rec_df["scale"].eq("train_1m")].nlargest(5, "centroid")["domain"].tolist()
    config_plots()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(scales))
    bottom = np.zeros(len(scales))
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]
    for idx, dom in enumerate(p1_top5):
        vals = rec_df[rec_df["domain"].eq(dom)].set_index("scale").loc[scales, "centroid"].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=dom, color=colors[idx])
        bottom += vals
    ax.set_xticks(x, scales)
    ax.set_ylabel("推荐占比")
    ax.set_title("算力约束下推荐配比的前 5 域占比随规模变化")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ENH_FIG / "cross_scale_recommendation.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  Top-5 domains of each scale's top-10% centroid:")
    for scale in scales:
        top = rec_df[rec_df["scale"].eq(scale)].nlargest(5, "centroid")
        print(f"    {scale}: " + ", ".join(f"{r['domain']}={r['centroid']:.3f}"
                                         for _, r in top.iterrows()))
    print(f"  → saved cross_scale_recommendation.csv, cross_scale_recommendation.pdf")


# ============================================================
# Module 5: Conflict sensitivity (threshold methods)
# ============================================================
def module5_conflict_sensitivity():
    print("\n=== Module 5: Conflict sensitivity ===")
    scores = pd.read_csv(QUAL_OUT / "quality_sample_scores.csv", encoding="utf-8-sig")
    a1 = scores[scores["dataset"].eq("A1_sample")].copy()

    # Method A: unified A1 90th percentile (current default)
    thr_unified = a1["conflict_score"].quantile(0.90)
    # Method B: per-domain 90th percentile
    thr_per_domain = a1.groupby("domain")["conflict_score"].quantile(0.90).to_dict()
    # Method C: absolute 0.10
    thr_abs_010 = 0.10
    # Method D: absolute 0.05
    thr_abs_005 = 0.05

    methods = {
        "统一A1 90%分位": lambda df: thr_unified,
        "各域内部90%分位": lambda df: thr_per_domain.get(df["domain"].iloc[0], thr_unified),
        "绝对阈值 0.10": lambda df: thr_abs_010,
        "绝对阈值 0.05": lambda df: thr_abs_005,
    }

    rows = []
    for mname, thr_fn in methods.items():
        for domain, sub in a1.groupby("domain"):
            thr = thr_fn(sub)
            rate = float((sub["conflict_score"] > thr).mean())
            rows.append({"method": mname, "domain": domain, "threshold": float(thr),
                         "conflict_rate": rate, "n": len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(ENH_OUT / "conflict_sensitivity.csv", index=False, encoding="utf-8-sig")

    config_plots()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    domains = sorted(a1["domain"].unique())
    x = np.arange(len(domains))
    width = 0.2
    for i, mname in enumerate(methods):
        vals = [out[(out["method"].eq(mname)) & (out["domain"].eq(d))]["conflict_rate"].values[0]
                for d in domains]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=mname)
    ax.set_xticks(x, domains)
    ax.set_ylabel("冲突率")
    ax.set_title("冲突率对阈值定义的敏感性")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ENH_FIG / "conflict_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved conflict_sensitivity.csv + conflict_sensitivity.pdf")


# ============================================================
# Module 6: Quality metric variability (CV) by domain
# ============================================================
def module6_metric_cv():
    print("\n=== Module 6: Quality metric CV by domain ===")
    cache_path = P1 / "quality" / "cache" / "quality_features.csv"
    if not cache_path.exists():
        print("  Cache missing — skipping. Run problem1.py first."); return
    df = pd.read_csv(cache_path, encoding="utf-8-sig")
    METRICS = [c for c in df.columns if c.startswith(("dsir", "fineweb", "fluency", "rps_", "modernbert", "qurater", "ad_en"))]
    a1 = df[df["dataset"].eq("A1_sample")]
    cv_rows = []
    for domain, sub in a1.groupby("domain"):
        for m in METRICS:
            vals = sub[m].dropna()
            if len(vals) < 2: continue
            mu = vals.mean(); sd = vals.std()
            cv = sd / abs(mu) if abs(mu) > 1e-12 else np.nan
            cv_rows.append({"domain": domain, "metric": m, "mean": mu, "std": sd, "cv": cv})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(ENH_OUT / "metric_cv_by_domain.csv", index=False, encoding="utf-8-sig")

    # Avg CV per domain
    avg_cv = cv_df.groupby("domain")["cv"].mean().sort_values(ascending=False)
    conflict = pd.read_csv(QUAL_OUT / "quality_domain_summary.csv", encoding="utf-8-sig")
    conflict = conflict[conflict["dataset"].eq("A1_sample")].set_index("domain")["conflict_rate"]
    merged = pd.DataFrame({"avg_cv": avg_cv, "conflict_rate": conflict}).dropna()
    if len(merged) > 1:
        r, p = pearsonr(merged["avg_cv"], merged["conflict_rate"])
        print(f"  Avg metric CV vs conflict rate: Pearson r = {r:.4f}, p = {p:.3e}")
    print(f"  Top-3 avg CV domains: {avg_cv.head(3).to_dict()}")
    print(f"  → saved metric_cv_by_domain.csv")


# ============================================================
# Main
# ============================================================
def main():
    summary = {}
    r_all, p_all, r_te, p_te_ = module1_quality_external_validity()
    summary["Q_vs_Loss_train_pearson"] = float(r_all)
    summary["Q_vs_Loss_train_p"] = float(p_all)
    summary["Q_vs_Loss_test_pearson"] = float(r_te)
    summary["Q_vs_Loss_test_p"] = float(p_te_)

    module2_dphi_and_r2()
    module3_ablation()
    module4_cross_scale_recommendation()
    module5_conflict_sensitivity()
    module6_metric_cv()

    (ENH_OUT / "enhancement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n======== Enhancement complete ========")
    print(f"  All results in: {ENH_OUT}")
    print(f"  All figures in: {ENH_FIG}")


if __name__ == "__main__":
    main()
