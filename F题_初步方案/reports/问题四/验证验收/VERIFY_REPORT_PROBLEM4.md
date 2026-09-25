# 问题四独立验收

通过 55/55 项。

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| C3 all supplied rows classified | 通过 |  |
| C3 leaderboard rows are C1 duplicates up to rounding | 通过 |  |
| C1 records omitted from C3 have missing parameter size | 通过 |  |
| C3 historical score discrepancy independently recomputed | 通过 |  |
| C2 raw row contract | 通过 |  |
| Panel model unique | 通过 |  |
| Six-task average | 通过 |  |
| Strict open evidence | 通过 |  |
| Unknown weight stratum excludes explicit No | 通过 |  |
| Panel cutoff | 通过 |  |
| Main panel preserves attachment Phi type labels | 通过 |  |
| Pretrained late count | 通过 |  |
| Flow raw row count | 通过 |  |
| Publication audit totals strict pre and post | 通过 |  |
| Publication lag independently recomputed | 通过 |  |
| Every accepted link passes all rules | 通过 |  |
| Accepted C4 records are one-to-one | 通过 |  |
| Known developer id agrees with repository namespace | 通过 |  |
| Compute subset has positive recorded compute | 通过 |  |
| Compute subset only strict pretrained | 通过 |  |
| Source review queue retains original-label candidates | 通过 |  |
| Checkpoint and training telemetry review not falsely marked complete | 通过 |  |
| Manual review source fields copied from C4 | 通过 |  |
| External source review covers every original candidate once | 通过 |  |
| Source review keeps three evidence classes distinct | 通过 |  |
| Source-derived FLOPs numerically reconcile with C4 approximate values | 通过 |  |
| Stage-mismatched records excluded from supported subset | 通过 |  |
| Ridge alpha selected by family-balanced CV | 通过 |  |
| Time holdout row RMSE independently recomputed | 通过 |  |
| Time holdout family-balanced RMSE independently recomputed | 通过 |  |
| Temporal model comparison agrees with saved holdout errors | 通过 |  |
| Temporal comparison bootstrap bounds valid | 通过 |  |
| Rolling descriptive screen rules independently checked | 通过 |  |
| Rolling train and test counts recomputed from panel | 通过 |  |
| Rolling errors independently recomputed | 通过 |  |
| Family review queue is ranked and pending | 通过 |  |
| Compute association bootstrap intervals valid | 通过 |  |
| Source-supported compute subset correlation independently recomputed | 通过 |  |
| Source-supported compute uncertainty includes zero | 通过 |  |
| C8 file count | 通过 |  |
| C8 timestamps numeric | 通过 |  |
| C8 unique latest model | 通过 |  |
| C8 parsed leaf average independently | 通过 |  |
| C8 different scale explicit | 通过 |  |
| No unsupported decomposition | 通过 |  |
| Shapley additivity | 通过 |  |
| Observed change residual additivity | 通过 |  |
| Bootstrap replicate additivity | 通过 |  |
| Strict pretrained gate failed | 通过 |  |
| Overlap-only fit decomposition additive | 通过 |  |
| Bridge high only seven and no frontier translation | 通过 |  |
| Bridge LOO RMSE | 通过 |  |
| No unsupported future score emitted | 通过 |  |
| Scenario negative size trend recorded | 通过 |  |
| Scenario time model holdout gate failed | 通过 |  |

## 验收范围

这些检查独立读取原始 C2/C3/C4/C8 与保存产物，核对口径、守门条件和数值回代。C3 历史补录的分数口径不一致，因此只用于来源与趋势可比性审计。外部来源网页的具体文字由逐行来源审计记录，本脚本只检查审计文件的覆盖和结果一致性；它不证明 checkpoint 哈希一致，不把观察性分解升级为因果识别，也不证明未来外推可信。
