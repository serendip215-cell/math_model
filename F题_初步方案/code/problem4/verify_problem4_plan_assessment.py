"""Independent checks for the external-plan assessment outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
R = BASE / "outputs" / "problem4" / "results"


def main() -> None:
    panel = pd.read_csv(R / "audited_leaderboard_panel.csv")
    sample = panel[panel.strict_open & panel.type_group.eq("posttrained")]
    early = sample[sample.submission.le("2024-08-31")]
    late = sample[sample.submission.ge("2025-01-01")]
    train = sample[sample.submission.lt("2025-01-01")]
    summary = pd.read_csv(R / "plan_task_common_family_summary.csv")
    detail = pd.read_csv(R / "plan_task_common_family_detail.csv")
    q = pd.read_csv(R / "plan_quantile_common_split.csv")
    status = json.loads((R / "plan_attachment_assessment.json").read_text(encoding="utf-8"))

    assert len(early) == 33 and len(late) == 67 and len(train) == 74
    assert set(early.family) & set(late.family) == {"llama", "phi", "qwen"}
    for row in summary.itertuples():
        np.testing.assert_allclose(row.all_sample_delta,
            late[row.task].mean() - early[row.task].mean(), atol=1e-10)
        d = detail[detail.task.eq(row.task)]
        assert len(d) == 3
        np.testing.assert_allclose(row.common_family_equal_weight_delta,
                                   d.within_family_delta.mean(), atol=1e-10)
        for r in d.itertuples():
            a = early[early.family.eq(r.family)][r.task]
            b = late[late.family.eq(r.family)][r.task]
            assert len(a) == r.early_n and len(b) == r.late_n
            np.testing.assert_allclose(r.within_family_delta, b.mean() - a.mean(), atol=1e-10)

    raw = pd.read_csv(R / "frontier_quantile_audit.csv")
    for r in raw.itertuples():
        v = q[q.response_scale.eq("raw") & q.variant.eq(r.variant)]
        if len(v) != 1:
            continue
        v = v.iloc[0]
        np.testing.assert_allclose(v.cv_family_pinball_score_points,
                                   r.train_cv_family_pinball, atol=1e-9)
        np.testing.assert_allclose(v.holdout_family_pinball_score_points,
                                   r.holdout_family_pinball, atol=1e-9)

    for r in q[q.response_scale.eq("logit")].itertuples():
        features = ["logN"] if r.variant == "N_only" else ["logN", "t_month"]
        scale = StandardScaler().fit(train[features])
        model = QuantileRegressor(quantile=0.9, alpha=r.alpha_cv, solver="highs")
        model.fit(scale.transform(train[features]),
                  logit(np.clip(train.Y.to_numpy() / 100, 1e-4, 1 - 1e-4)))
        predictions = 100 * expit(model.predict(scale.transform(late[features])))
        errors = np.where(late.Y.to_numpy() >= predictions,
                          .9 * (late.Y.to_numpy() - predictions),
                          .1 * (predictions - late.Y.to_numpy()))
        frame = pd.DataFrame({"family": late.family.to_numpy(), "loss": errors})
        np.testing.assert_allclose(frame.groupby("family").loss.mean().mean(),
                                   r.holdout_family_pinball_score_points, atol=1e-9)
        np.testing.assert_allclose(np.mean(late.Y.to_numpy() <= predictions),
                                   r.holdout_row_coverage, atol=1e-9)

    assert len(status["attachment_refinement_scripts_missing"]) == 3
    assert status["forecast_12_24_month_score_identified"] is False
    assert q.selected_by_training_cv.sum() == 1
    assert q.long_horizon_validated.eq(False).all()
    print("PASS: common-family arithmetic, same-split raw and logit scores, evidence boundaries")


if __name__ == "__main__":
    main()
