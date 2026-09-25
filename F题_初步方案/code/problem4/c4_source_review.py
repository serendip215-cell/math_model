"""Source-level adjudication of the 29 rule-linked C1/C4 compute records.

This is a frozen, human-readable evidence ledger, not an automated web claim.
`source_supported_estimate` means the *order and construction* of C4's
approximate pretraining FLOPs are supported by a primary source; it does not
mean that a training operator reported or measured the precise FLOP number.
"""
from __future__ import annotations

import pandas as pd


PYTHIA = "https://arxiv.org/html/2304.01373"
QWEN2 = "https://arxiv.org/html/2407.10671"
FALCON = "https://arxiv.org/html/2311.16867"
YI15 = "https://huggingface.co/01-ai/Yi-1.5-34B"

# Model ID: (decision, primary evidence URL, short source observation).
EVIDENCE = {
    **{f"EleutherAI/pythia-{size}": (
        "source_supported_estimate", PYTHIA,
        "Paper reports 299,892,736,000 training tokens for the released Pythia suite; C4 uses 6 x rounded size x tokens, an approximation, not measured FLOPs."
    ) for size in ("410m", "1b", "1.4b", "2.8b", "6.9b", "12b")},
    "tiiuae/falcon-7b": (
        "source_supported_estimate", FALCON,
        "Author Table 1 reports 730 PF-days and 1,500B tokens; C4's 6.3e22 FLOPs agrees with PF-day conversion to rounding."
    ),
    "tiiuae/falcon-40b": (
        "source_supported_estimate", FALCON,
        "Author Table 1 reports 2,800 PF-days and 1,000B tokens; C4's 2.4e23 FLOPs approximates the 2.4192e23 PF-day conversion."
    ),
    "togethercomputer/RedPajama-INCITE-7B-Base": (
        "source_supported_estimate",
        "https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base",
        "Official base-model card reports 1.001T training tokens; C4 estimates about 6 x 6.9B x 1.001T."
    ),
    "HuggingFaceTB/SmolLM-1.7B": (
        "source_supported_estimate", "https://huggingface.co/blog/smollm",
        "Developer says base 1.7B model trained on 1T tokens; C4 computes 6 x 1.71B x 1T."
    ),
    "microsoft/phi-1_5": (
        "source_supported_estimate", "https://huggingface.co/microsoft/phi-1_5",
        "Official base-model card reports 150B training tokens; C4 uses 6 x 1.3B x 150B, not a measured FLOP count."
    ),
    "microsoft/phi-2": (
        "source_supported_estimate", "https://huggingface.co/microsoft/phi-2",
        "Official base-model card reports 1.4T training tokens; C4 uses 6 x 2.7B x 1.4T, not a measured FLOP count."
    ),
    **{f"Qwen/Qwen2-{size}": (
        "source_supported_estimate", QWEN2,
        f"Author Table 1 reports {'12T' if size == '0.5B' else '7T'} tokens for this dense size; C4 applies approximate 6ND."
    ) for size in ("0.5B", "1.5B", "7B")},
    "01-ai/Yi-1.5-34B": (
        "not_comparable", YI15,
        "Official card says Yi-1.5 continues training from Yi for 500B tokens; C4 applies 3.6T cumulative tokens as if all training belonged to this stage."
    ),
    "01-ai/Yi-1.5-9B": (
        "not_comparable", "https://huggingface.co/01-ai/Yi-1.5-9B",
        "Official card says Yi-1.5 continues training from Yi for 500B tokens; C4 applies 3.6T cumulative tokens as if all training belonged to this stage."
    ),
    "Qwen/Qwen2-57B-A14B": (
        "not_comparable", QWEN2,
        "Paper describes dense-to-MoE upcycling and an additional 4.5T tokens; C4's active-parameter 6ND omits ancestor training and is not total from-scratch compute."
    ),
    "microsoft/phi-1": (
        "not_comparable", "https://arxiv.org/html/2306.11644",
        "Author paper distinguishes phi-1-base after 51B tokens from phi-1 after additional code-exercise fine-tuning; C4 value mixes estimates and does not isolate the released checkpoint's full training."
    ),
    "microsoft/phi-4": (
        "not_comparable", "https://huggingface.co/microsoft/phi-4",
        "Official model card describes SFT and DPO on released phi-4; C4 geometric-mean estimate addresses pretraining-scale compute, not the released post-trained checkpoint's total."
    ),
    "EleutherAI/gpt-j-6b": (
        "insufficient_evidence", "https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/",
        "Developer reports 400B tokens, but the cited page does not directly report C4's exact 1.5e22 FLOPs; C4 labels it Reported."
    ),
    "EleutherAI/gpt-neox-20b": (
        "insufficient_evidence", "https://arxiv.org/html/2204.06745",
        "Paper describes training hardware and throughput; C4 uses correspondence plus an assumed 0.4 utilization, not a directly reported 9.316e22 FLOPs."
    ),
    "EleutherAI/gpt-neo-2.7B": (
        "insufficient_evidence", "https://github.com/EleutherAI/gpt-neo",
        "C4 cites a third-party 7.9e21 estimate, while its own 6ND calculation is 6.804e21; numerical discrepancy unresolved."
    ),
    "01-ai/Yi-34B": (
        "insufficient_evidence", "https://arxiv.org/abs/2403.04652",
        "C4 itself questions whether the approximately 3T tokens are dataset size or tokens actually seen; estimate remains uncertain."
    ),
    **{f"Qwen/Qwen2.5-{size}": (
        "insufficient_evidence", "https://qwenlm.github.io/blog/qwen2.5/",
        "Developer says up to 18T tokens across the family; the cited page does not establish exactly 18T for this individual checkpoint."
    ) for size in ("1.5B", "7B", "14B", "32B")},
    "mosaicml/mpt-7b": (
        "insufficient_evidence", "https://www.mosaicml.com/blog/mpt-7b",
        "C4 lists 440 A100s for about 9.5 days but does not document how its exact 4.2e22 FLOPs was obtained; model-specific FLOP estimate remains to reconcile."
    ),
}

# Recomputed only where the author's tokens or PF-days support the same scope.
# 6ND is an approximation for dense pretraining, not an instrument reading.
SOURCE_DERIVED_FLOPS = {
    **{f"EleutherAI/pythia-{size}": 6 * n * 299_892_736_000
       for size, n in (("410m", 410e6), ("1b", 1e9), ("1.4b", 1.4e9),
                       ("2.8b", 2.8e9), ("6.9b", 6.9e9), ("12b", 12e9))},
    "tiiuae/falcon-7b": 730 * 1e15 * 86_400,
    "tiiuae/falcon-40b": 2800 * 1e15 * 86_400,
    "togethercomputer/RedPajama-INCITE-7B-Base": 6 * 6.9e9 * 1.001e12,
    "HuggingFaceTB/SmolLM-1.7B": 6 * 1.71e9 * 1e12,
    "microsoft/phi-1_5": 6 * 1.3e9 * 150e9,
    "microsoft/phi-2": 6 * 2.7e9 * 1.4e12,
    "Qwen/Qwen2-0.5B": 6 * .5e9 * 12e12,
    "Qwen/Qwen2-1.5B": 6 * 1.5e9 * 7e12,
    "Qwen/Qwen2-7B": 6 * 7e9 * 7e12,
}


def adjudicate(queue: pd.DataFrame) -> pd.DataFrame:
    if len(queue) != 29 or set(queue.Model_C1) != set(EVIDENCE):
        raise ValueError("C4 source-review contract changed; re-adjudicate before using decisions")
    reviewed = queue[["Model_C1", "Model_C4", "Training compute (FLOP)",
                      "Training compute estimation method", "Link"]].copy()
    reviewed["source_review_status"] = reviewed.Model_C1.map(lambda m: EVIDENCE[m][0])
    reviewed["primary_evidence_url"] = reviewed.Model_C1.map(lambda m: EVIDENCE[m][1])
    reviewed["evidence_observation"] = reviewed.Model_C1.map(lambda m: EVIDENCE[m][2])
    reviewed["source_derived_flops"] = reviewed.Model_C1.map(SOURCE_DERIVED_FLOPS)
    reviewed["source_vs_c4_relative_difference"] = (
        (reviewed["Training compute (FLOP)"].astype(float) -
         reviewed.source_derived_flops).abs() / reviewed.source_derived_flops)
    supported = reviewed.source_review_status.eq("source_supported_estimate")
    if reviewed.loc[supported, "source_derived_flops"].isna().any() or (
            reviewed.loc[supported, "source_vs_c4_relative_difference"] > .05).any():
        raise ValueError("Primary-source approximate FLOPs disagree with C4 by over 5%")
    reviewed["review_scope"] = "source-page and C4-note audit; no checkpoint hash or measured-run telemetry"
    return reviewed
