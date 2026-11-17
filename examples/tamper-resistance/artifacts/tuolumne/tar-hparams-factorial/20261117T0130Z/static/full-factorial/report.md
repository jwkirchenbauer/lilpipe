# Eight-cell controlled static comparison

R changes retain coefficient 2.5→4, I changes inner learning rate 1e-6→1e-5, and O changes outer learning rate 5e-6→1e-5. All 8 cells keep the original local sampler, K16/125 defense updates, source, data order and seeds fixed. Only defense endpoints and cached static grids are required; no new attacks or benchmarks enter this report.

These figures use the exact original BF16 random recipe and default attention: seed 42, the fixed set of 10 texts, truncation 512, batch size 2, independent globally normalized BF16 directions and sequential BF16 perturbations. They are separate from the newer FP32 orthonormal dynamic views. Each radius shares physical coordinates, orientation, absolute CE/z limits and camera across all 8 controlled cells. Center-relative views subtract each cell's measured center and share a symmetric color scale. Radii have separate scales. The earlier uploaded HF contextual comparison is outside this controlled factorial.

The rim/elevation names below refer only to finite sampled grids. These translated slices evaluate a fixed forget CE objective; they do not establish global barriers, Hessian curvature or robustness. One seed and no replication support descriptive contrasts only; there are no p-values.

## Prespecified metrics and formulas

- **center_ce**: L[4,4] on the 9x9 grid; CE in nats.
- **rim_rise**: max(L[4+di,4+dj] for di,dj in {-1,0,1} except di=dj=0) - L[4,4]. The 8 immediate neighbors have coordinate spacing radius/4. No clipping at zero.
- **center_elevation**: L[4,4] - mean(L[i,j] where i or j is 0 or 8). The perimeter contains 32 cells. No clipping at zero.
- **interpretation**: Finite-grid diagnostics of these sampled slices; not global barriers, Hessian curvature or attack robustness. One seed and no replication; no p-values.
- **factor_coding**: R/I/O are binary 0/1. R: retain 2.5/4; I: inner LR 1e-6/1e-5; O: outer LR 5e-6/1e-5.
- **anchored_coefficients**: For subset S, c_S=sum(T subset S) (-1)^(|S|-|T|) f(1_T,0_else). No divisor. The exact 0/1 multilinear expansion is f(r,i,o)=c_empty+c_R*r+c_I*i+c_O*o+c_RI*r*i+c_RO*r*o+c_IO*i*o+c_RIO*r*i*o.
- **conditional_main**: At each fixed combination of the other 2 factors, f(factor=1)-f(factor=0). No divisor; the marginal main contrast is the mean of 4 such differences, divisor 4.
- **conditional_pair**: At each fixed setting of the third factor, f11-f10-f01+f00. No divisor; the marginal pair contrast is the mean of 2 such differences, divisor 2.
- **three_way**: f111-f110-f101-f011+f100+f010+f001-f000. No divisor; also the difference between conditional pair contrasts at third-factor settings 1 and 0.

All reported differences have units of CE nats. Anchored coefficients and conditional differences have divisor 1. Marginal main contrasts average 4 conditional differences; marginal pair contrasts average 2. They therefore use distinct, explicitly recorded normalizations. The three-way coefficient is also the change of a conditional pair difference between the two settings of its third factor.

## Focus views

| Half-width | Absolute CE | Center-relative CE |
|---:|---|---|
| 125 | [PNG](radius-125-absolute.png) / [PDF](radius-125-absolute.pdf) | [PNG](radius-125-delta.png) / [PDF](radius-125-delta.pdf) |
| 100 | [PNG](radius-100-absolute.png) / [PDF](radius-100-absolute.pdf) | [PNG](radius-100-delta.png) / [PDF](radius-100-delta.pdf) |
| 1000 | [PNG](radius-1000-absolute.png) / [PDF](radius-1000-absolute.pdf) | [PNG](radius-1000-delta.png) / [PDF](radius-1000-delta.pdf) |
| 500 | [PNG](radius-500-absolute.png) / [PDF](radius-500-absolute.pdf) | [PNG](radius-500-delta.png) / [PDF](radius-500-delta.pdf) |
| 200 | [PNG](radius-200-absolute.png) / [PDF](radius-200-absolute.pdf) | [PNG](radius-200-delta.png) / [PDF](radius-200-delta.pdf) |
| 10 | [PNG](radius-10-absolute.png) / [PDF](radius-10-absolute.pdf) | [PNG](radius-10-delta.png) / [PDF](radius-10-delta.pdf) |
| 1 | [PNG](radius-1-absolute.png) / [PDF](radius-1-absolute.pdf) | [PNG](radius-1-delta.png) / [PDF](radius-1-delta.pdf) |
| 0.1 | [PNG](radius-0.1-absolute.png) / [PDF](radius-0.1-absolute.pdf) | [PNG](radius-0.1-delta.png) / [PDF](radius-0.1-delta.pdf) |

## Finite-grid diagnostics at half-width 125

| Cell | Retain | Inner LR | Outer LR | Center CE | Rim rise | Center elevation |
|---|---:|---:|---:|---:|---:|---:|
| B | 2.5 | 1e-06 | 5e-06 | 11.238433 | 0.039010 | 1.195921 |
| R | 4 | 1e-06 | 5e-06 | 10.957229 | 0.036222 | 0.937697 |
| I | 2.5 | 1e-05 | 5e-06 | 6.932543 | 0.345449 | -3.598975 |
| RI | 4 | 1e-05 | 5e-06 | 5.802483 | 0.286946 | -4.512189 |
| O | 2.5 | 1e-06 | 1e-05 | 11.256805 | 0.014800 | 1.257908 |
| RO | 4 | 1e-06 | 1e-05 | 10.931001 | 0.035761 | 0.641795 |
| IO | 2.5 | 1e-05 | 1e-05 | 12.510035 | 1.542543 | 0.887891 |
| RIO | 4 | 1e-05 | 1e-05 | 14.411371 | 0.630281 | 2.739132 |

Anchored polynomial coefficients (other factors at 0; raw differences, divisor 1):

| Term | Center CE | Rim rise | Center elevation |
|---|---:|---:|---:|
| R | -0.281204 | -0.002788 | -0.258224 |
| I | -4.305890 | 0.306439 | -4.794896 |
| O | 0.018372 | -0.024209 | 0.061987 |
| RI | -0.848857 | -0.055716 | -0.654989 |
| RO | -0.044601 | 0.023748 | -0.357889 |
| IO | 5.559120 | 1.221303 | 4.424879 |
| RIO | 3.075997 | -0.877506 | 3.122344 |

## Finite-grid diagnostics at half-width 100

| Cell | Retain | Inner LR | Outer LR | Center CE | Rim rise | Center elevation |
|---|---:|---:|---:|---:|---:|---:|
| B | 2.5 | 1e-06 | 5e-06 | 11.238433 | 0.028466 | 1.689740 |
| R | 4 | 1e-06 | 5e-06 | 10.957229 | 0.028418 | 1.535009 |
| I | 2.5 | 1e-05 | 5e-06 | 6.932543 | 0.181194 | -2.558846 |
| RI | 4 | 1e-05 | 5e-06 | 5.802483 | 0.166069 | -3.351025 |
| O | 2.5 | 1e-06 | 1e-05 | 11.256805 | 0.011917 | 1.643345 |
| RO | 4 | 1e-06 | 1e-05 | 10.931001 | 0.028694 | 0.663041 |
| IO | 2.5 | 1e-05 | 1e-05 | 12.510035 | 1.099821 | 1.122936 |
| RIO | 4 | 1e-05 | 1e-05 | 14.411371 | 0.533710 | 2.611313 |

Anchored polynomial coefficients (other factors at 0; raw differences, divisor 1):

| Term | Center CE | Rim rise | Center elevation |
|---|---:|---:|---:|
| R | -0.281204 | -0.000048 | -0.154731 |
| I | -4.305890 | 0.152728 | -4.248586 |
| O | 0.018372 | -0.016549 | -0.046395 |
| RI | -0.848857 | -0.015076 | -0.637448 |
| RO | -0.044601 | 0.016825 | -0.825573 |
| IO | 5.559120 | 0.935176 | 3.728177 |
| RIO | 3.075997 | -0.567812 | 3.106128 |

[All radii grid metrics](grid-metrics.csv) and [all anchored, conditional and marginal contrasts](factorial-contrasts.csv) retain the measured values. The normalization column is the divisor used for averaging conditional differences; anchored and conditional differences are unscaled. Signs describe differences at these two chosen levels, not a fitted dose response.
