"""Reproducible checks prompted by an external Q4 plan.

The attachment's claimed refinement outputs are not inputs to this analysis.
All estimates below use the existing audited C1/C2 panel and retain the same
strict-open, early/late, and 2025 holdout definitions as the Q4 main pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = BASE / "outputs" / "problem4" / "results"
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
TAU = 0.9


def pinball(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.where(y >= pred, TAU * (y - pred), (1 - TAU) * (pred - y))


def predict(fit: pd.DataFrame, held: pd.DataFrame, variant: str,
            alpha: float, response: str) -> np.ndarray:
    cols = ["logN"] if variant == "N_only" else ["logN", "t_month"]
    scaler = StandardScaler().fit(fit[cols])
    y = fit.Y.to_numpy(dtype=float)
    if response == "logit":
        y = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    model = QuantileRegressor(quantile=TAU, alpha=alpha, solver="highs")
    model.fit(scaler.transform(fit[cols]), y)
    out = model.predict(scaler.transform(held[cols]))
    if response == "logit":
        return 100 * expit(out)
    return np.clip(out, 0, 100)


def quantile_common_split(panel: pd.DataFrame) -> pd.DataFrame:
    sample = panel.loc[panel.strict_open & panel.type_group.eq("posttrained")]
    train = sample.loc[sample.submission.lt("2025-01-01")].copy()
    held = sample.loc[sample.submission.ge("2025-01-01")].copy()
    assert (len(train), len(held), train.family.nunique(), held.family.nunique()) == (74, 67, 14, 6)
    rows = []
    for response in ("raw", "logit"):
        for variant in ("N_only", "N_time"):
            candidates = []
            for alpha in ALPHAS:
                oof = []
                for tr, va in GroupKFold(n_splits=5).split(train, groups=train.family):
                    fit, val = train.iloc[tr], train.iloc[va]
                    p = predict(fit, val, variant, alpha, response)
                    oof.extend(zip(val.family, pinball(val.Y.to_numpy(), p)))
                oof = pd.DataFrame(oof, columns=["family", "loss"])
                candidates.append((float(oof.groupby("family").loss.mean().mean()), alpha))
            cv, alpha = min(candidates)
            p = predict(train, held, variant, alpha, response)
            hold = held[["family", "Y"]].copy()
            hold["loss"] = pinball(hold.Y.to_numpy(), p)
            rows.append({"response_scale": response, "variant": variant,
                         "alpha_cv": alpha, "train_n": len(train),
                         "holdout_n": len(held), "train_families": train.family.nunique(),
                         "holdout_families": held.family.nunique(),
                         "cv_family_pinball_score_points": cv,
                         "holdout_family_pinball_score_points":
                             float(hold.groupby("family").loss.mean().mean()),
                         "holdout_row_coverage": float(np.mean(held.Y.to_numpy() <= p)),
                         "long_horizon_validated": False})
    result = pd.DataFrame(rows)
    result["selected_by_training_cv"] = result.cv_family_pinball_score_points.eq(
        result.cv_family_pinball_score_points.min())
    result.to_csv(RESULTS / "plan_quantile_common_split.csv", index=False, encoding="utf-8-sig")
    return result


def task_common_family(panel: pd.DataFrame) -> pd.DataFrame:
    sample = panel.loc[panel.strict_open & panel.type_group.eq("posttrained")]
    early = sample.loc[sample.submission.le("2024-08-31")]
    late = sample.loc[sample.submission.ge("2025-01-01")]
    common = sorted(set(early.family) & set(late.family))
    assert len(early) == 33 and len(late) == 67
    assert common == ["llama", "phi", "qwen"]
    details, rows = [], []
    for task in TASKS + ["Y"]:
        differences = []
        for fam in common:
            a, b = early.loc[early.family.eq(fam), task], late.loc[late.family.eq(fam), task]
            delta = float(b.mean() - a.mean())
            differences.append(delta)
            details.append({"task": task, "family": fam, "early_n": len(a),
                            "late_n": len(b), "early_mean": float(a.mean()),
                            "late_mean": float(b.mean()), "within_family_delta": delta})
        rows.append({"task": task, "all_early_n": len(early), "all_late_n": len(late),
                     "common_families": len(common),
                     "common_early_n": int(early.family.isin(common).sum()),
                     "common_late_n": int(late.family.isin(common).sum()),
                     "all_sample_delta": float(late[task].mean() - early[task].mean()),
                     "common_family_equal_weight_delta": float(np.mean(differences)),
                     "common_family_min_delta": float(min(differences)),
                     "common_family_max_delta": float(max(differences)),
                     "interpretation": "descriptive_common_families_only_not_causal_attribution"})
    pd.DataFrame(details).to_csv(RESULTS / "plan_task_common_family_detail.csv",
                                 index=False, encoding="utf-8-sig")
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "plan_task_common_family_summary.csv",
                  index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    panel = pd.read_csv(RESULTS / "audited_leaderboard_panel.csv")
    q = quantile_common_split(panel)
    t = task_common_family(panel)
    missing = [name for name in ("problem4_refinement.py", "problem4_refinement2.py",
                                "problem4_refinement3.py") if not (HERE / name).exists()]
    status = {"attachment_refinement_scripts_missing": missing,
              "attachment_extra_results_verified": False,
              "common_family_count": 3,
              "forecast_12_24_month_score_identified": False,
              "quantile_selected_training_model": q.loc[q.selected_by_training_cv,
                  ["response_scale", "variant"]].to_dict("records"),
              "math5_descriptive": t.loc[t.task.eq("MATH Lvl 5")].to_dict("records")[0]}
    (RESULTS / "plan_attachment_assessment.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
