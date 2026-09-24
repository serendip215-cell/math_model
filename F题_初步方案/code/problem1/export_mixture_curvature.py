"""Export the selected A4 response surface and recipe-bootstrap uncertainty.

The export is diagnostic: pair effects are local fitted nonadditivity under a
specified simplex perturbation, not causal complementarity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1 import MIXTURE_OUT, ilr_transform, load_pair


def raw_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(x, y)
    scale = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    coef = ridge.coef_ / scale.scale_
    intercept = ridge.intercept_ - coef @ scale.mean_
    return np.asarray(coef), np.asarray(intercept)


def main() -> None:
    summary = json.loads((MIXTURE_OUT / "mixture_model_summary.json").read_text(encoding="utf-8"))
    if summary["selected_model"] != "quadratic_ilr_ridge" or summary["quality_features_adopted"]:
        raise ValueError("Selected problem-one model differs from the audited quadratic ILR surface")
    index, p, y, pcols, ycols, _ = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    basis = helmert(len(pcols), full=False)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x = poly.fit_transform(ilr_transform(p, basis))
    alpha = float(summary["quadratic_alpha"])
    coef, intercept = raw_coefficients(x, y, alpha)
    reference = pd.read_csv(MIXTURE_OUT / "mixture_predictions.csv")
    reference = reference[reference["split"] == "train_1m"].set_index("index").loc[index]
    prediction = x @ coef.T + intercept
    max_error = float(np.max(np.abs(prediction.mean(axis=1) - reference["predicted_macro_loss"].to_numpy())))
    if max_error > 1e-8:
        raise AssertionError(f"Refitted response disagrees with saved predictions: {max_error}")
    powers = poly.powers_
    hessian = np.zeros((len(ycols), basis.shape[0], basis.shape[0]))
    for k, power in enumerate(powers):
        dims = np.flatnonzero(power)
        if len(dims) == 1 and power[dims[0]] == 2:
            hessian[:, dims[0], dims[0]] += 2 * coef[:, k]
        elif len(dims) == 2:
            a, b = dims
            hessian[:, a, b] += coef[:, k]
            hessian[:, b, a] += coef[:, k]
    np.savez_compressed(MIXTURE_OUT / "selected_response_surface.npz",
                        coefficients=coef, intercept=intercept, basis=basis,
                        powers=powers, hessian_ilr=hessian,
                        domains=np.array([c.replace("train_the_pile_", "") for c in pcols]),
                        targets=np.array(ycols), alpha=np.array(alpha))
    rng = np.random.default_rng(20260924)
    draws = []
    for replicate in range(500):
        sample = rng.integers(0, len(p), len(p))
        boot_coef, _ = raw_coefficients(x[sample], y[sample].mean(axis=1), alpha)
        draws.append(boot_coef)
    np.savez_compressed(MIXTURE_OUT / "selected_response_bootstrap.npz",
                        macro_coefficients=np.asarray(draws),
                        replicate=np.arange(len(draws)), seed=np.array(20260924))
    (MIXTURE_OUT / "selected_response_export_audit.json").write_text(json.dumps({
        "selected_model": summary["selected_model"], "alpha": alpha,
        "train_rows": len(p), "targets": len(ycols), "bootstrap_replicates": len(draws),
        "max_macro_prediction_difference": max_error,
        "hessian_coordinates": "ILR logratio coordinates; not 17 raw mixture shares",
        "bootstrap_unit": "independent A4 recipe row",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported selected surface; max prediction difference={max_error:.3e}")


if __name__ == "__main__":
    main()
