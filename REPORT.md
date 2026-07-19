# 2×2 vs 3×3 kernel ablation

Pre-specified non-inferiority margin: 1.00 percentage point.

## model2

- 2×2: n=3, mean=91.500%, 95% CI=[90.734, 92.266]
- 3×3: n=3, mean=93.321%, 95% CI=[91.377, 95.265]

Paired Δ(2×2 − 3×3): -1.821 pp; 95% CI [-4.281, +0.640] pp.
2×2 non-inferior within 1.00 pp: **NO**.
Zero lies in the 95% CI (no significant difference): **YES**.

## model3

- 2×2: n=3, mean=91.784%, 95% CI=[91.319, 92.249]
- 3×3: n=3, mean=93.031%, 95% CI=[92.238, 93.824]

Paired Δ(2×2 − 3×3): -1.247 pp; 95% CI [-1.796, -0.698] pp.
2×2 non-inferior within 1.00 pp: **NO**.
Zero lies in the 95% CI (no significant difference): **NO**.

## Existing 2×2 reference (not used in paired statistics)

- Model 2 int8 validation: 92.06%
- Model 3 int8+KD validation: 91.83%
