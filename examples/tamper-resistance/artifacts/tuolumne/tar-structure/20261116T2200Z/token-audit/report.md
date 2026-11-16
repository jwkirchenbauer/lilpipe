# Embedding/head loading and native-basis token audit

Only embedding and output-head matrix slices were read. No model was instantiated.

Candidate loading map: BF16(source anchor). Exact equality is an algebraic measurement, not proof of a training cause. 'Unseen' includes all padded input IDs; defense uses only outer roles 0/1.

| First saved update / matrix | Source → saved dtype | Unseen rows | Raw L2 | Load-offset L2 | Adjusted L2 | Max residual | Exact raw=offset |
|---|---|---:|---:|---:|---:|---:|---|
| local-defense-attack-defense-1/embedding | BF16 → F32 | 31034 | 0 | 0 | 0 | 0 | True |
| local-defense-attack-defense-1/head | BF16 → F32 | 31034 | 0 | 0 | 0 | 0 | True |
| local-defense-attack-attack-1/embedding | F32 → F32 | 31770 | 0.033010568 | 0.033010568 | 0 | 0 | True |
| local-defense-attack-attack-1/head | F32 → F32 | 31770 | 0.081033809 | 0.080355685 | 0.01046795 | 1.0002404e-06 | False |
| collaborator-attack-attack-1/embedding | F32 → F32 | 31770 | 0.034257526 | 0.034257526 | 0 | 0 | True |
| collaborator-attack-attack-1/head | F32 → F32 | 31770 | 0.080410019 | 0.080249259 | 0.0050858054 | 1.0002404e-06 | False |

Counts below cover 90% of the named matrix's energy. Uniform-plane energy averages unit PC axes; variance-plane energy weights them by native uncentered displacement eigenvalues. The contribution of the first two axes normalized by the full spectrum is also preserved in JSON/CSV/NPZ. This does not estimate centered statistical variance.

| Case / matrix / method | Matrix energy | Rows for 90% | Columns for 90% | Top 10 column fraction | Top 100 column fraction |
|---|---:|---:|---:|---:|---:|
| local-defense-attack/embedding/pc1 | 0.0073294914 | 10603 | 3667 | 0.0055 | 0.0318 |
| local-defense-attack/embedding/pc2 | 0.021843466 | 9277 | 3673 | 0.0038 | 0.0281 |
| local-defense-attack/embedding/uniform_plane | 0.014586479 | 9779 | 3673 | 0.0043 | 0.0289 |
| local-defense-attack/embedding/variance_plane | 0.0086574745 | 10571 | 3669 | 0.0052 | 0.0309 |
| local-defense-attack/embedding/variance_full | 0.0082935452 | 10571 | 3669 | 0.0052 | 0.0309 |
| local-defense-attack/head/pc1 | 0.29550199 | 25537 | 2901 | 0.0097 | 0.0774 |
| local-defense-attack/head/pc2 | 0.39179771 | 21763 | 3193 | 0.0130 | 0.0877 |
| local-defense-attack/head/uniform_plane | 0.34364985 | 24370 | 3289 | 0.0090 | 0.0619 |
| local-defense-attack/head/variance_plane | 0.30431275 | 25410 | 3064 | 0.0091 | 0.0714 |
| local-defense-attack/head/variance_full | 0.29152053 | 25410 | 3064 | 0.0091 | 0.0714 |
| collaborator-attack/embedding/pc1 | 0.0014843329 | 10226 | 3669 | 0.0045 | 0.0300 |
| collaborator-attack/embedding/pc2 | 0.014472681 | 11643 | 3658 | 0.0078 | 0.0366 |
| collaborator-attack/embedding/uniform_plane | 0.0079785067 | 11635 | 3660 | 0.0075 | 0.0359 |
| collaborator-attack/embedding/variance_plane | 0.0018321651 | 10961 | 3669 | 0.0052 | 0.0313 |
| collaborator-attack/embedding/variance_full | 0.0018160052 | 10961 | 3669 | 0.0052 | 0.0313 |
| collaborator-attack/head/pc1 | 0.85789723 | 22778 | 2938 | 0.0052 | 0.0465 |
| collaborator-attack/head/pc2 | 0.53781404 | 24558 | 3292 | 0.0151 | 0.0912 |
| collaborator-attack/head/uniform_plane | 0.69785564 | 24504 | 3284 | 0.0066 | 0.0466 |
| collaborator-attack/head/variance_plane | 0.8493253 | 22886 | 2966 | 0.0051 | 0.0460 |
| collaborator-attack/head/variance_full | 0.84183419 | 22886 | 2966 | 0.0051 | 0.0460 |

[Column concentration](column-concentration.png) · [All numeric results](results.json) · [Loading rows](loading-offset-rows.csv) · [Basis rows](basis-rows.csv) · [Basis columns](basis-columns.csv) · [Top20 decoded rows per tensor/case](top-tokens.csv)

Algebraic agreement with a candidate BF16 loading map is measured, not assumed causal. Input embeddings and untied output head have different gradient support; output softmax can update classes absent from input IDs. No semantic biology-token inference is made.
