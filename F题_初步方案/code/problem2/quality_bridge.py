"""Auditable interface from problem-one mixture quality to problem-two scaling.

No A-to-B quality calibration is estimated: the supplied datasets have no
paired experiment with both quality definitions and a common loss protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compose_problem1_quality(shares: pd.Series, domain_q: pd.Series) -> float:
    """Compute Q_A(p) only for a complete, aligned 17-domain simplex."""
    if shares.index.has_duplicates or domain_q.index.has_duplicates or set(shares.index) != set(domain_q.index):
        raise ValueError("领域质量与配比的 17 域键不一致")
    if len(shares) != 17 or not np.isfinite(shares.to_numpy(float)).all() or not np.isfinite(domain_q.to_numpy(float)).all():
        raise ValueError("17 域质量或配比不完整")
    if (shares < 0).any() or abs(float(shares.sum()) - 1.0) > 1e-8:
        raise ValueError("配比不满足非负及和为 1")
    if (domain_q < 0).any() or (domain_q > 1).any():
        raise ValueError("第一问质量分超出 [0,1]")
    return float(shares.sort_index() @ domain_q.sort_index())


def predict_with_problem1_quality(q_a: float, calibration=None) -> float:
    """Return B-scale Q only when an externally identified calibration is given."""
    if calibration is None:
        raise ValueError("A/B Q 标尺无成对实验标定，不能把问题一 Q 直接代入 B6 模型")
    q_b = float(calibration(float(q_a)))
    if not 0.1 <= q_b <= 1.0:
        raise ValueError("标定后的 Q_B 超出 B6 的 [0.1,1.0] 支持域")
    return q_b


def build_quality_bridge(p1: Path, b_root: Path, results: Path, figures: Path) -> dict:
    raw_mapping = pd.read_csv(b_root.parent / "A_data_value" / "domain_mapping_guide.csv")
    mapping = pd.read_csv(p1 / "mixture" / "results" / "quality_domain_mapping.csv")
    proxy = pd.read_csv(p1 / "inferred_quality" / "inferred_quality_proxy.csv")
    recipe = pd.read_csv(p1 / "mixture" / "results" / "recommended_mixture.csv")
    if any(t["domain"].duplicated().any() for t in [proxy, recipe]):
        raise AssertionError("问题一领域键不唯一")
    if mapping["mixture_domain"].duplicated().any():
        raise AssertionError("A16 映射领域键不唯一")
    check_mapping = mapping.merge(raw_mapping, on="mixture_domain", how="outer", suffixes=("_saved", "_A16"),
                                  validate="one_to_one", indicator=True)
    if len(check_mapping) != 17 or not (check_mapping["_merge"] == "both").all() or not (
        check_mapping["quality_domain_saved"].astype(str) == check_mapping["quality_domain_A16"].astype(str)
    ).all() or not (
        check_mapping["mapping_type_saved"].astype(str) == check_mapping["mapping_type_A16"].astype(str)
    ).all():
        raise AssertionError("问题一保存的域映射与原始 A16 不一致")
    domain = recipe.merge(proxy[["domain", "q_combined", "mapping_type"]], on="domain", validate="one_to_one")
    domain = domain.merge(mapping[["mixture_domain", "quality_domain", "quality_score", "mapping_available"]],
                          left_on="domain", right_on="mixture_domain", validate="one_to_one")
    if len(domain) != 17 or domain[["q_combined", "training_mean_share", "recommended_share"]].isna().any().any():
        raise AssertionError("17 域质量向量或配比不完整")
    if not np.all((domain["q_combined"] >= 0) & (domain["q_combined"] <= 1)):
        raise AssertionError("问题一 Q 代理值超出 [0,1]")
    for c in ["training_mean_share", "recommended_share"]:
        if abs(float(domain[c].sum()) - 1.0) > 1e-8 or (domain[c] < 0).any():
            raise AssertionError(f"{c} 不满足配比约束")
    domain["quality_provenance"] = np.where(domain["mapping_available"],
                                            "A16_mapped_domain_score", "loss_informed_inferred_proxy")
    domain["share_change"] = domain["recommended_share"] - domain["training_mean_share"]
    domain["proxy_delta_contribution"] = domain["share_change"] * domain["q_combined"]
    domain.to_csv(results / "quality_bridge_domain_audit.csv", index=False, encoding="utf-8-sig")

    anchor_rows = []
    known = domain["mapping_available"].astype(bool)
    for label, col in [("training_mean", "training_mean_share"), ("recommended_centroid", "recommended_share")]:
        shares = domain[col].to_numpy(float)
        mapped_contribution = float((domain.loc[known, col] * domain.loc[known, "quality_score"]).sum())
        unmapped_share = float(domain.loc[~known, col].sum())
        anchor_rows.append({"anchor": label, "share_sum": float(shares.sum()),
                            "mapped_domain_share": float(domain.loc[known, col].sum()),
                            "unmapped_domain_share": unmapped_share,
                            "q_a_loss_informed_proxy": compose_problem1_quality(
                                domain.set_index("domain")[col], domain.set_index("domain")["q_combined"]),
                            "q_a_lower_given_A16_mappings": mapped_contribution,
                            "q_a_upper_given_A16_mappings": mapped_contribution + unmapped_share})
    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(results / "quality_bridge_recipe_anchors.csv", index=False, encoding="utf-8-sig")

    mapped_delta = float((domain.loc[known, "share_change"] * domain.loc[known, "quality_score"]).sum())
    unknown_changes = domain.loc[~known, "share_change"].to_numpy(float)
    delta_low = mapped_delta + float(np.minimum(unknown_changes, 0).sum())
    delta_high = mapped_delta + float(np.maximum(unknown_changes, 0).sum())
    proxy_delta = float(domain["proxy_delta_contribution"].sum())
    if not delta_low - 1e-10 <= proxy_delta <= delta_high + 1e-10:
        raise AssertionError("代理值不在识别区间内")
    bounds = pd.DataFrame([{"comparison": "recommended_minus_training_mean",
                            "mapped_domain_contribution": mapped_delta,
                            "unmapped_share_change_negative_sum": float(np.minimum(unknown_changes, 0).sum()),
                            "unmapped_share_change_positive_sum": float(np.maximum(unknown_changes, 0).sum()),
                            "delta_q_a_lower_given_A16_mappings": delta_low,
                            "delta_q_a_upper_given_A16_mappings": delta_high,
                            "delta_q_a_loss_informed_proxy": proxy_delta,
                            "direction_identified": bool(delta_low > 0 or delta_high < 0),
                            "Q_B_calibration_identified": False}])
    bounds.to_csv(results / "quality_bridge_delta_bounds.csv", index=False, encoding="utf-8-sig")

    a_mix = pd.read_csv(b_root.parent / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv", nrows=1)
    a_loss = pd.read_csv(b_root.parent / "A_data_value" / "regmix_tables" / "train_pile_loss_1m.csv", nrows=1)
    b1 = pd.read_csv(b_root / "pythia_training_log_existing.csv", nrows=1)
    b6 = pd.read_csv(b_root / "supplementary_NQ_experiment.csv", nrows=1)
    contracts = pd.DataFrame([
        {"source": "A_mixture_1m", "has_17_domain_recipe": True, "has_loss": False, "has_Q_B": False,
         "has_N_D_grid": False, "has_common_AB_experiment_id": False},
        {"source": "A_pile_loss_1m", "has_17_domain_recipe": False, "has_loss": True, "has_Q_B": False,
         "has_N_D_grid": False, "has_common_AB_experiment_id": False},
        {"source": "B1_Pythia", "has_17_domain_recipe": False, "has_loss": "val_loss" in b1,
         "has_Q_B": False, "has_N_D_grid": {"N_params_B", "D_tokens_B"}.issubset(b1.columns),
         "has_common_AB_experiment_id": False},
        {"source": "B6_NDQ", "has_17_domain_recipe": False, "has_loss": "val_loss" in b6,
         "has_Q_B": "Q_score" in b6, "has_N_D_grid": {"N_params_B", "D_tokens_B"}.issubset(b6.columns),
         "has_common_AB_experiment_id": False},
    ])
    if not any(c.startswith("train_the_pile_") for c in a_mix) or not any(c.endswith("_val_loss") for c in a_loss):
        raise AssertionError("A 配比或 Loss 表结构变化")
    if any(c.startswith("train_the_pile_") for c in b6) or "Q_score" in a_mix:
        raise AssertionError("A/B 联合数据可用性发生变化，需重新评估标定")
    contracts.to_csv(results / "quality_bridge_joint_data_audit.csv", index=False, encoding="utf-8-sig")

    summary = {"mapped_domains": int(known.sum()), "unmapped_domains": int((~known).sum()),
               "q_a_training_mean_proxy": float(anchors.iloc[0]["q_a_loss_informed_proxy"]),
               "q_a_recommended_proxy": float(anchors.iloc[1]["q_a_loss_informed_proxy"]),
               "delta_q_a_proxy": proxy_delta, "delta_q_a_bound": [delta_low, delta_high],
               "direction_identified": bool(bounds.iloc[0]["direction_identified"]),
               "q_a_to_q_b_calibration_identified": False,
               "joint_A_B_loss_validation_possible": False,
               "reason": "B6 lacks 17-domain recipe; A recipe/loss data lack B6 Q and common training-loss protocol",
               "proxy_caution": "11 inferred domain scores are informed by A training Loss; not independent quality measurements"}
    (results / "quality_bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.hlines(0, delta_low, delta_high, color="#2563eb", linewidth=7, label="未映射域取 [0,1] 的识别区间")
    ax.plot(proxy_delta, 0, "o", color="#dc2626", markersize=9, label="17 域 Loss 信息代理值")
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks([]); ax.set_xlabel("推荐配比 − 训练均值的第一问质量变化")
    ax.set_xlim(min(delta_low, 0) - .01, max(delta_high, 0) + .01)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.28), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "quality_bridge_identification.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary
