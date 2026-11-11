# Compute and intervention findings

All 18 scientific cases and all 16 attack trajectory analyses are complete.
The [measurement report](report.md) links the figures and validation checks.

More defense compute changes the sampled neighborhoods, but all three defenders
still admit descending low-LR updates and reach similar fixed-probe CE after
1,000 attack updates. Along the shared reference directions, initial sampled
CE spans are 1.088 nats for the baseline, 1.107 for the longer defense, and
0.377 for the deeper defense. Final center CE is 1.949, 1.927, and 1.990,
respectively. The deeper defense looks more level in this initial reference
window while retaining descent directions elsewhere. These two selected
directions do not determine global sharpness or resistance. See the
[compute neighborhoods](comparisons/compute-local-delta.png).
In the [broad random slices](comparisons/compute-broad-random.png), all three
final centers are the sampled grid minimum, with maxima 9.11–9.27 nats above
them. Their initial shape differences do not preserve an elevated central
region after recovery in this sampled view.

Increasing the nominal defense budget from 2,000 to 8,000 simulated attacker
updates did not consistently reduce recovered WMDP under the clean attacks. Each
entry below maximizes over the three declared attack learning rates and saved
checkpoints up to the stated budget.
The pairs mean defense updates × simulated attacker updates per defense update.
The [budget table](strongest-attacks-by-budget.csv) preserves the measured maxima.

| Defense | Maximum WMDP through 128 attack updates | Through 1,000 | MMLU before attack |
|---|---:|---:|---:|
| 125 × 16 baseline | 56.45% | 56.45% | 54.02% |
| 500 × 16 | 55.88% | 56.91% | 54.92% |
| 125 × 64 | 56.57% | 56.57% | 53.06% |

These are descriptive comparisons from one defender seed and one optimizer seed.
The 125 × 64 defender falls below the declared 53.59% utility floor. No saved
state of either new defender met the exploratory utility-matching band. The
conditions also differ in outer data exposure, optimizer reset cadence, and
schedules. Their nominal inner budgets match, but measured defense costs differ:
4.50 allocated GPU hours for 500 × 16 and 3.32 for 125 × 64, including loading,
observations, and checkpoint writes.

Attack learning rate changes the time scale of recovery substantially. On the
baseline, the first measured fixed-probe CE at or below 5 nats occurs at saved
update 3 for LR `1e-5`, 4 for `5e-6`, and 48 for `1e-6`. The largest LR does not
give the highest recovered WMDP at every budget. The deeper defender delays
low-LR recovery further, but recovery still occurs within 1,000 updates.

Noise sigma `1e-4` preserves initial utility and behaves much like the clean
control. Larger perturbations reduce recovered WMDP while severely damaging
MMLU before the attack; this does not demonstrate useful robustness. Full-size
data bootstraps alter the early trajectory, but all three recover WMDP to about
54% by update 1,000. These interventions were tested independently.
The sigma `1e-3` model eventually also has a low-CE central slice, while its
capability and utility remain substantially below the clean model. The forget
loss landscape therefore needs the capability and utility measurements beside it.

The plane fitted to the complete baseline attack trajectory explains 91–96% of
summed squared raw displacement, yet captures only 13–19% of the first saved
raw displacement. This fit is an uncentered singular-value decomposition of
displacements from the attack start, with equal weight for each selected
checkpoint. Large late displacements can dominate it. Direct-update profiles
and first-update-aligned slices therefore complement the fixed reference views.
For example, projecting baseline LR `1e-5` update 4 onto its fitted plane changes
evaluated fixed-probe CE by 5.96 nats. Shared-reference projections can also
remain inaccurate at the endpoint. The translated local slices compare actual
model neighborhoods; a projected path on a fixed slice cannot replace the
optimizer's route.
Negligible capture by an arbitrary random plane is expected in this parameter
dimension and, by itself, does not identify an escape mechanism.

Along the first eight saved update segments, the fixed-probe samples show no
interior CE above the higher endpoint, apart from the severely damaged
sigma `1e-1` control. The actual-batch curves broadly agree, with a small
0.0071-nat sampled barrier at bootstrap 101 update 3. Some segments vary
nonmonotonically below their higher endpoint, and fixed-probe loss can rise
between saved updates. A zero sampled barrier does not imply monotonic descent.
For the three low-LR compute conditions, both fixed-probe and actual-batch
profiles descend over the first eight segments within `1e-4` nats. Their first
fixed-probe reductions are 0.088 nats for the baseline, 0.189 for the longer
defense, and 0.039 for the deeper defense. These local measurements do not prove
a globally barrier-free route or resolve every point between saved states.

Individual batches matter. Bootstrap 102 update 3 reduces its own evaluated
batch CE by 5.78 nats while increasing fixed-probe CE by 1.74 nats. A fixed
probe alone would miss the improvement on the batch driving that update.

Learning rate and bootstrap data leave the defended starting weights and the
fixed-probe objective unchanged. Their first-update planes differ because the
recorded update directions differ. A different-looking initial slice in these
families therefore shows a different direction through the same loss field.
Compute and noise interventions also change the model at the attack start.
See the [early compute profiles](comparisons/compute-early-update-profiles.png)
and [first-update planes](comparisons/compute-first-update-landscape.png).

The completed LR and bootstrap families show a common late change: after 1,000
attack updates, the broad shared-random slice has a low central region, where
the defended model had an elevated central region. Center CE falls from about
11.24 to 1.84–1.96 nats, while sampled outer points remain higher. The span
within the small shared-reference window shrinks from 1.088 nats initially to
0.0027–0.255 across LRs and 0.0046–0.0155 across bootstraps. These are slices
translated through the moving model, with fixed directions and common scales;
the fixed-probe objective itself has not changed.
Compare the [broad LR slices](comparisons/learning-rate-broad-random.png) with
the [small bootstrap neighborhoods](comparisons/data-local-delta.png).

Geometry uses the mixed-precision saved parameters; the loss evaluations use
BF16 weights. For example, the baseline LR `1e-6` first update has raw L2
distance 0.09767 and evaluation-cast distance 0.03583. The direct profiles keep
these quantities separate and verify the losses at the saved endpoints.
Interpolated weights are diagnostic samples, not states visited by the optimizer.
For the sigma `1e-1` control, recorded training batch CE and evaluated batch CE
can differ by about 3.7 nats. Its rough evaluated profiles therefore cannot be
read as equally rough native mixed-precision training losses.
