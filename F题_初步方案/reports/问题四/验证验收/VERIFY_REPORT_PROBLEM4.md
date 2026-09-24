# 问题四独立验收

通过 31/31 项。

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| C2 raw row contract | 通过 |  |
| Panel model unique | 通过 |  |
| Six-task average | 通过 |  |
| Strict open evidence | 通过 |  |
| Panel cutoff | 通过 |  |
| Pretrained late count | 通过 |  |
| Flow raw row count | 通过 |  |
| Every accepted link passes all rules | 通过 |  |
| Accepted C4 records are one-to-one | 通过 |  |
| Known developer id agrees with repository namespace | 通过 |  |
| Compute subset has verified positive compute | 通过 |  |
| Compute subset only strict pretrained | 通过 |  |
| Time holdout row RMSE independently recomputed | 通过 |  |
| Time holdout family-balanced RMSE independently recomputed | 通过 |  |
| Temporal model comparison agrees with saved holdout errors | 通过 |  |
| Temporal comparison bootstrap bounds valid | 通过 |  |
| Compute association bootstrap intervals valid | 通过 |  |
| C8 file count | 通过 |  |
| C8 unique latest model | 通过 |  |
| C8 parsed leaf average independently | 通过 |  |
| C8 different scale explicit | 通过 |  |
| No unsupported decomposition | 通过 |  |
| Shapley additivity | 通过 |  |
| Observed change residual additivity | 通过 |  |
| Bootstrap replicate additivity | 通过 |  |
| Strict pretrained gate failed | 通过 |  |
| Bridge high only seven and no frontier translation | 通过 |  |
| Bridge LOO RMSE | 通过 |  |
| No unsupported future score emitted | 通过 |  |
| Scenario negative size trend recorded | 通过 |  |
| Scenario time model holdout gate failed | 通过 |  |

## 验收范围

这些检查独立读取原始 C2/C8 与保存产物，核对口径、守门条件和数值回代。它们不把观察性分解升级为因果识别，也不证明未来外推可信。
