| Benchmark | n | E1: few-shot | E2: compact | E3: GRPO | E3 − E2 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 200 | 100/200 (50.0%) | 104/200 (52.0%) | 110/200 (55.0%) | +3.0 pp |
| math500 | 200 | 79/200 (39.5%) | 70/200 (35.0%) | 74/200 (37.0%) | +2.0 pp |
| humaneval | 50 | 21/50 (42.0%) | 17/50 (34.0%) | 20/50 (40.0%) | +6.0 pp |
| mbpp | 100 | 32/100 (32.0%) | 27/100 (27.0%) | 33/100 (33.0%) | +6.0 pp |

| Experiment | Evidence |
|---|---|
| E1 | aggregate_only |
| E2 | per_example_logs |
| E3 | aggregate_only |

Point estimates only; aggregate-only conditions cannot support paired per-example significance tests.
