"""Independent arithmetic checks for problem4_frontier_sensitivity.py outputs."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
R = BASE / "outputs" / "problem4" / "results"
TAU = 0.9
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def close(a: float, b: float, tol: float = 1e-9) -> None:
    if not np.isclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"{a} != {b}")


def main() -> None:
    audit = pd.read_csv(R / "frontier_quantile_audit.csv")
    held = pd.read_csv(R / "frontier_quantile_holdout_predictions.csv")
    panel = pd.read_csv(R / "audited_leaderboard_panel.csv")
    task = pd.read_csv(R / "task_fixed_reference_standardization.csv")
    compute = pd.read_csv(R / "compute_joint_family_cv_audit.csv")
    checks = 0

    assert set(audit.variant) == {"constant", "N_only", "N_time"}
    assert audit.selected_by_train_cv.sum() == 1
    checks += 2
    for row in audit.itertuples():
        sub = held[held.variant.eq(row.variant)]
        assert len(sub) == 67 and sub.family.nunique() == 6
        loss = np.where(sub.Y >= sub.predicted_q90,
                        TAU * (sub.Y - sub.predicted_q90),
                        (1 - TAU) * (sub.predicted_q90 - sub.Y))
        close(loss.mean(), row.holdout_row_pinball)
        close(sub.groupby("family").pinball.mean().mean(),
              row.holdout_family_pinball)
        close(np.mean(sub.Y <= sub.predicted_q90),
              row.holdout_row_coverage)
        assert not row.coverage_guarantee
        assert not row.long_horizon_forecast_validated
        checks += 6

    sub = panel[panel.strict_open & panel.type_group.eq("posttrained")]
    train = sub[sub.submission.lt("2025-01-01")]
    early = sub[sub.submission.le("2024-08-31")]
    late = sub[sub.submission.ge("2025-01-01")]
    assert len(train) == 74 and len(early) == 33 and len(late) == 67
    for row in task.itertuples():
        assert row.task in TASKS
        iqr = train[row.task].quantile(.75) - train[row.task].quantile(.25)
        delta = late[row.task].mean() - early[row.task].mean()
        close(row.train_iqr, iqr)
        close(row.raw_points_change, delta)
        close(row.change_in_train_iqr_units, delta / iqr)
        checks += 3

    assert set(compute["sample"]) == {
        "attachment_linked", "source_supported_sensitivity"}
    assert set(compute.variant) == {"N_only", "C_only", "N_plus_C"}
    assert compute.family_group_cv_rmse.gt(0).all()
    checks += 3
    print(f"PASS: {checks} arithmetic and evidence-boundary checks")


if __name__ == "__main__":
    main()
