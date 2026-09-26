# Phase 2 model evaluation

## Scope and protocol

The production forecaster is Dixon-Coles Poisson model v2. It fits team attack and defence strengths, home advantage, a Dixon-Coles low-score correction, and exponentially time-weighted historical matches, then derives home/draw/away probabilities from its scoreline matrix. Elo is the principal simple-model baseline. Multinomial logistic regression and XGBoost are comparison models, not production candidates.

Evaluation is walk-forward over five held-out La Liga seasons: 2020/21 through 2024/25 (1,900 matches; 380 per season). Each season is forecast using only information available before the relevant match. Log loss is the primary metric; ranked probability score (RPS), multiclass Brier score, expected calibration error (ECE), and top-probability accuracy are also reported. The comparisons include a season base-rate forecaster, Elo, and de-vigged closing odds (the `market` benchmark).

## Headline result

Production v2 pooled the following walk-forward results:

| Model | Log loss | RPS | Brier | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dixon-Coles v2 | 0.9914 | 0.1983 | 0.5907 | 0.0176 | 52.21% |
| Elo | 0.9957 | 0.1979 | 0.5932 | 0.0291 | n/a |
| Base rate | 1.0736 | 0.2260 | 0.6493 | 0.0157 | n/a |
| De-vigged closing odds | 0.9693 | 0.1912 | 0.5756 | 0.0373 | n/a |

The model beat the base-rate baseline in log loss in all five held-out seasons, and beat Elo in all five. This result is season-level, not a pooled-only outcome:

| Season | Dixon-Coles log loss | Elo | Base rate | Closing odds | Dixon-Coles RPS | Brier | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020/21 | 1.0040 | 1.0063 | 1.0892 | 0.9797 | 0.1984 | 0.5999 | 0.0470 | 50.3% |
| 2021/22 | 1.0044 | 1.0103 | 1.0791 | 0.9920 | 0.1967 | 0.5993 | 0.0417 | 51.6% |
| 2022/23 | 0.9927 | 1.0013 | 1.0532 | 0.9764 | 0.2063 | 0.5910 | 0.0488 | 54.2% |
| 2023/24 | 0.9734 | 0.9762 | 1.0755 | 0.9513 | 0.1906 | 0.5792 | 0.0285 | 52.1% |
| 2024/25 | 0.9824 | 0.9846 | 1.0709 | 0.9470 | 0.1994 | 0.5843 | 0.0745 | 52.9% |

## Closing-line comparison

The model narrowly missed the closing-line acceptance tolerance. Its pooled log loss was +0.0221 above de-vigged closing odds, and its largest single-season gap was +0.0354 in 2024/25. The per-season gaps were +0.0243, +0.0124, +0.0163, +0.0221, and +0.0354 from 2020/21 to 2024/25.

This is a narrow miss against the sharpest available benchmark, not evidence that the benchmark was beaten. Consistently beating closing odds would itself be a leakage warning; this evaluation does not show that.

## Calibration

Isotonic calibration was fitted walk-forward using forecasts from earlier held-out seasons only. For Dixon-Coles, the calibration-eligible pooled window comprised 1,520 matches (2021/22-2024/25). With the job's default shrinkage of 0.30, ECE changed from 0.0186 to 0.0205 and log loss from 0.9883 to 0.9888. The job's verdict was **no help**: calibration did not produce a material improvement and was not supported by these results.

## Half-life tuning

The validation-only selection on 2019/20 chose a 1,095-day half-life (validation log loss 1.0383, versus 1.0454 for the 540-day baseline). On the untouched five-season test window, the selected setting recorded pooled log loss 0.9926 and a +0.0233 closing-line gap. It therefore did not close the closing-line gap; the 2024/25 gap remained +0.0349.

## Comparison models

Hyperparameters were selected by walk-forward validation on 2018/19 and 2019/20, before the five-season test window was scored.

| Model | Selected settings | Validation mean log loss | Test log loss | RPS | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | `C=0.3` | 1.0468 | 1.0241 | 0.2085 | 0.6136 | 0.0237 |
| XGBoost | `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=750` | 1.0427 | 1.0316 | 0.2107 | 0.6188 | 0.0232 |

Neither tuned comparison model beat Dixon-Coles v2 on the core evaluation metrics. Both had worse log loss, RPS, Brier score, and ECE; their season-level accuracies were also lower in each of the five test seasons.

## Known limitations

- No xG is available; shots on target are used as its proxy.
- There is no lineup or injury data.
- Fixture congestion is not modelled beyond rest days.
- Player statistics are out of scope for this phase.

## Reproducibility status

The registered production evaluation records fixed seed 42, but a fixed-seed, clean-checkout test reproducing this full evaluation has not been run. That remains an open item.
