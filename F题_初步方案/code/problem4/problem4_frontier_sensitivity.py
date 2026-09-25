"""Supplementary, attachment-derived audits for Q4 frontier suggestions.

Run after problem4.py. This module does not revise the main Q4 model or
generate a 12/24-month score forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "outputs" / "problem4" / "results"
FIGURES = BASE / "outputs" / "problem4" / "figures"
TAU = 0.9
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
RNG = np.random.default_rng(20260925)


def pinball(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.where(observed >= predicted,
                    TAU * (observed - predicted),
                    (1 - TAU) * (predicted - observed))


def predict_quantile(train: pd.DataFrame, test: pd.DataFrame,
                     variant: str, alpha: float | None) -> np.ndarray:
    if variant == "constant":
        return np.full(len(test), float(train.Y.quantile(TAU)))
    cols = ["logN"] if variant == "N_only" else ["logN", "t_month"]
    scaler = StandardScaler().fit(train[cols])
    model = QuantileRegressor(quantile=TAU, alpha=float(alpha),
                              solver="highs")
    model.fit(scaler.transform(train[cols]), train.Y.to_numpy())
    return np.clip(model.predict(scaler.transform(test[cols])), 0, 100)


def family_cv(train: pd.DataFrame, variant: str,
              alpha: float | None) -> tuple[float, float]:
    folds = GroupKFold(n_splits=5)
    records = []
    for tr, va in folds.split(train, groups=train.family):
        fit, held = train.iloc[tr], train.iloc[va]
        predicted = predict_quantile(fit, held, variant, alpha)
        losses = pinball(held.Y.to_numpy(), predicted)
        records.extend((fam, loss, float(obs <= pred))
                       for fam, loss, obs, pred in
                       zip(held.family, losses, held.Y, predicted))
    frame = pd.DataFrame(records, columns=["family", "pinball", "covered"])
    return (float(frame.groupby("family").pinball.mean().mean()),
            float(frame.covered.mean()))


def family_bootstrap(frame: pd.DataFrame, column: str,
                     n: int = 1000) -> tuple[float, float]:
    groups = {name: rows[column].to_numpy(dtype=float)
              for name, rows in frame.groupby("family")}
    keys = list(groups)
    draws = []
    for _ in range(n):
        sampled = RNG.choice(keys, size=len(keys), replace=True)
        draws.append(float(np.mean([groups[key].mean() for key in sampled])))
    return tuple(np.quantile(draws, [0.025, 0.975]))


def frontier_audit(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel.loc[panel.strict_open & panel.type_group.eq("posttrained")].copy()
    train = sub[sub.submission.lt("2025-01-01")].copy()
    test = sub[sub.submission.ge("2025-01-01")].copy()
    assert len(train) == 74 and len(test) == 67
    assert train.family.nunique() == 14 and test.family.nunique() == 6

    rows, predictions = [], []
    for variant in ["constant", "N_only", "N_time"]:
        choices = [None] if variant == "constant" else ALPHAS
        cv = [(family_cv(train, variant, alpha), alpha) for alpha in choices]
        (cv_pin, cv_cover), alpha = min(cv, key=lambda item: item[0][0])
        pred = predict_quantile(train, test, variant, alpha)
        frame = test[["Model", "family", "submission", "logN", "Y"]].copy()
        frame["variant"] = variant
        frame["predicted_q90"] = pred
        frame["pinball"] = pinball(frame.Y.to_numpy(), pred)
        frame["covered"] = frame.Y.le(frame.predicted_q90).astype(float)
        predictions.append(frame)
        pin_lo, pin_hi = family_bootstrap(frame, "pinball")
        cov_lo, cov_hi = family_bootstrap(frame, "covered")
        rows.append({
            "variant": variant, "tau": TAU, "train_n": len(train),
            "train_families": train.family.nunique(), "holdout_n": len(test),
            "holdout_families": test.family.nunique(), "alpha_train_cv": alpha,
            "train_cv_family_pinball": cv_pin,
            "train_cv_row_coverage": cv_cover,
            "holdout_row_pinball": float(frame.pinball.mean()),
            "holdout_family_pinball":
                float(frame.groupby("family").pinball.mean().mean()),
            "holdout_row_coverage": float(frame.covered.mean()),
            "holdout_family_coverage":
                float(frame.groupby("family").covered.mean().mean()),
            "family_boot_pinball_low": pin_lo,
            "family_boot_pinball_high": pin_hi,
            "family_boot_coverage_low": cov_lo,
            "family_boot_coverage_high": cov_hi,
            "coverage_guarantee": False,
            "long_horizon_forecast_validated": False,
        })
    audit = pd.DataFrame(rows)
    audit["selected_by_train_cv"] = audit.train_cv_family_pinball.eq(
        audit.train_cv_family_pinball.min())
    audit.to_csv(RESULTS / "frontier_quantile_audit.csv",
                 index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        RESULTS / "frontier_quantile_holdout_predictions.csv",
        index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for variant, color in [("N_only", "#137c8b"), ("N_time", "#d57529")]:
        frame = next(x for x in predictions if x.variant.iloc[0] == variant)
        order = np.argsort(frame.logN.to_numpy())
        axes[0].plot(frame.logN.to_numpy()[order],
                     frame.predicted_q90.to_numpy()[order],
                     label=variant, color=color, linewidth=1.6)
    axes[0].scatter(test.logN, test.Y, s=14, color="black", alpha=.42,
                    label="Held-out models")
    axes[0].set(xlabel="log10(N in billions)", ylabel="Six-task mean score",
                title="2025 holdout: conditional 90th quantile")
    axes[0].legend(fontsize=8)
    axes[1].bar(audit.variant, audit.holdout_row_coverage,
                color=["#777777", "#137c8b", "#d57529"])
    axes[1].axhline(.9, color="black", linestyle="--", linewidth=1,
                    label="Nominal 0.90")
    axes[1].set(ylim=(0, 1.05), ylabel="Observed holdout coverage",
                title="Descriptive coverage; 6 families")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_frontier_quantile_holdout.pdf",
                bbox_inches="tight")
    plt.close(fig)
    return audit


def task_scale_audit(panel: pd.DataFrame) -> pd.DataFrame:
    sub = panel.loc[panel.strict_open & panel.type_group.eq("posttrained")].copy()
    train = sub[sub.submission.lt("2025-01-01")]
    early = sub[sub.submission.le("2024-08-31")]
    late = sub[sub.submission.ge("2025-01-01")]
    rows = []
    for task in TASKS:
        median = float(train[task].median())
        iqr = float(train[task].quantile(.75) - train[task].quantile(.25))
        if not np.isfinite(iqr) or iqr <= 0:
            raise ValueError(f"Nonpositive training IQR for {task}")
        raw_delta = float(late[task].mean() - early[task].mean())
        rows.append({"task": task, "train_median": median,
                     "train_iqr": iqr, "early_n": len(early),
                     "late_n": len(late), "raw_points_change": raw_delta,
                     "change_in_train_iqr_units": raw_delta / iqr,
                     "reference_period": "strict posttrained before 2025-01-01"})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "task_fixed_reference_standardization.csv",
               index=False, encoding="utf-8-sig")
    return out


def source_support_audit() -> dict:
    medium = pd.read_csv(RESULTS / "bridge_medium_source_audit.csv")
    linked = pd.read_csv(RESULTS / "c4_compute_linked_subset.csv")
    high = pd.read_csv(RESULTS / "bridge_high_loo.csv")
    counts = medium.groupby("Loss_Source").size()
    data_positive = pd.to_numeric(
        linked["Training dataset size (total)"], errors="coerce").gt(0)
    return {
        "bridge_high_rows": int(len(high)),
        "bridge_medium_rows": int(len(medium)),
        "bridge_medium_source_labels": int(len(counts)),
        "bridge_medium_median_rows_per_source": float(counts.median()),
        "bridge_medium_singleton_source_labels": int(counts.eq(1).sum()),
        "linked_compute_rows": int(len(linked)),
        "linked_compute_positive_dataset_size_rows": int(data_positive.sum()),
        "linked_compute_posttrained_rows":
            int(linked.type_group.eq("posttrained").sum()),
        "strict_open_definition_unchanged": True,
        "unknown_weight_layer_is_sensitivity_only": True,
    }



def compute_joint_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Exploratory family-held-out prediction in the nonrandom C4 linked sample."""
    from sklearn.linear_model import Ridge

    linked = pd.read_csv(RESULTS / "c4_compute_linked_subset.csv")
    review = pd.read_csv(RESULTS / "c4_external_source_review.csv")
    linked = linked.merge(
        panel[["Model", "Y", "family"]],
        left_on="Model_C1", right_on="Model", how="inner", validate="one_to_one")
    linked = linked.merge(
        review[["Model_C1", "source_review_status"]],
        on="Model_C1", how="left", validate="one_to_one")
    linked["logC"] = np.log10(pd.to_numeric(
        linked["Training compute (FLOP)"], errors="coerce"))
    linked["logN"] = np.log10(linked.N_B)
    samples = [
        ("attachment_linked", linked),
        ("source_supported_sensitivity",
         linked[linked.source_review_status.eq("source_supported_estimate")])
    ]
    rows = []
    for sample_name, data in samples:
        assert len(data) in (29, 15)
        n_folds = min(5, data.family.nunique())
        for variant, cols in [
            ("N_only", ["logN"]), ("C_only", ["logC"]),
            ("N_plus_C", ["logN", "logC"])
        ]:
            for alpha in [0.1, 1.0, 10.0]:
                records = []
                for tr, te in GroupKFold(n_splits=n_folds).split(
                        data, groups=data.family):
                    fit, held = data.iloc[tr], data.iloc[te]
                    scaler = StandardScaler().fit(fit[cols])
                    model = Ridge(alpha=alpha).fit(
                        scaler.transform(fit[cols]), fit.Y)
                    prediction = model.predict(scaler.transform(held[cols]))
                    records.extend((fam, float((y - pred) ** 2))
                                   for fam, y, pred in zip(
                                       held.family, held.Y, prediction))
                held = pd.DataFrame(records, columns=["family", "squared"])
                rmse = float(np.sqrt(
                    held.groupby("family").squared.mean().mean()))
                rows.append({
                    "sample": sample_name, "variant": variant,
                    "alpha": alpha, "n": len(data),
                    "families": data.family.nunique(),
                    "family_group_cv_rmse": rmse,
                    "inference": "exploratory_association_not_causal",
                })
    out = pd.DataFrame(rows)
    out["best_alpha_within_variant"] = out.groupby(
        ["sample", "variant"]).family_group_cv_rmse.transform(
            "min").eq(out.family_group_cv_rmse)
    out.to_csv(RESULTS / "compute_joint_family_cv_audit.csv",
               index=False, encoding="utf-8-sig")
    return out

def main() -> None:
    panel = pd.read_csv(RESULTS / "audited_leaderboard_panel.csv")
    quantile = frontier_audit(panel)
    task = task_scale_audit(panel)
    support = source_support_audit()
    compute_joint = compute_joint_audit(panel)
    selected = quantile.loc[quantile.selected_by_train_cv].iloc[0]
    support["quantile_cv_selected_variant"] = selected.variant
    support["task_raw_six_mean_change_points"] = float(task.raw_points_change.mean())
    support["task_robust_standardized_change_iqr_units"] = float(
        task.change_in_train_iqr_units.mean())
    support["task_robust_standardized_change_without_math_iqr_units"] = float(
        task.loc[task.task.ne("MATH Lvl 5"),
                 "change_in_train_iqr_units"].mean())
    (RESULTS / "frontier_suggestion_support.json").write_text(
        json.dumps(support, ensure_ascii=False, indent=2), encoding="utf-8")
    print(quantile[["variant", "alpha_train_cv", "train_cv_family_pinball",
                    "holdout_family_pinball", "holdout_row_coverage",
                    "selected_by_train_cv"]].to_string(index=False))
    print(compute_joint[compute_joint.best_alpha_within_variant][["sample", "variant", "family_group_cv_rmse"]].to_string(index=False))
    print(json.dumps(support, ensure_ascii=False))


if __name__ == "__main__":
    main()
