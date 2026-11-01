# Locking toy: penalized SGD vs. projected ascent

This note specifies the toy experiment from *Locking Pretrained Weights via Deep
Low-Rank Residual Distillation* (DLR-Lock, Sakamoto et al., 2026; arXiv:2605.10777),
Section 2.2 and Figure 2, and the constrained method we added on top of it.
Everything below is implemented in `scripts/analysis/toy-experiments/get_locking_toy.py` and
plotted by `scripts/analysis/toy-experiments/plot_locking_toy.py`. It runs on CPU.

## 1. Setup

**Model.** A two-layer ReLU MLP with parameters $\theta = (W_1, b_1, W_2, b_2)$,
$W_1 \in \mathbb{R}^{128 \times 784}$, $W_2 \in \mathbb{R}^{10 \times 128}$:

$$
f_\theta(x) = W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2 \in \mathbb{R}^{10}.
$$

**Pretraining.** $\theta_0$ is trained with cross-entropy on the 60k MNIST
training images (pixels scaled to $[0,1]$, then normalized with mean 0.1307 and
std 0.3081) using Adam (lr $10^{-3}$, batch 128, 3 epochs). It reaches 97.3% test
accuracy. $\theta_0$ is then frozen; every later run starts from it.

**Data splits.**

| Name | Images | Used for |
| --- | --- | --- |
| Minibatches | random batches of 128 from the training set | penalized SGD |
| Probe set $X_P$ | first $N = 8192$ training images | constraint, $\Omega$ and $G$ in projected ascent |
| Evaluation set $X_E$ | first 1000 test images | every number on the plots |

**Starting point.** Every run starts at $\theta_0 + \xi$ with
$\xi \sim \mathcal{N}(0, (10^{-4})^2 I)$. This is needed because, for the
distance penalty, both the MSE and $\Omega$ have exactly zero gradient at
$\theta_0$, so a run started there would never move.

## 2. The quantities

### 2.1 Function drift

For a set of inputs $X$ the *pooled relative MSE* is

$$
c_X(\theta) = \frac{\sum_{x \in X} \lVert f_\theta(x) - f_{\theta_0}(x) \rVert^2}
                   {\sum_{x \in X} \lVert f_{\theta_0}(x) \rVert^2}.
$$

The constraint in projected ascent is $c_{X_P}(\theta) \le \varepsilon$.

The x-axis of every plot is the paper's *geometric* relative MSE on $X_E$:

$$
\mathrm{relMSE}(\theta) = \exp\!\Big( \tfrac{1}{|X_E|} \sum_{x \in X_E}
  \log \frac{\lVert f_\theta(x) - f_{\theta_0}(x) \rVert^2}{\lVert f_{\theta_0}(x) \rVert^2} \Big).
$$

The JSON also records $c_{X_E}$ as `pooled_relative_mse` and test accuracy as `accuracy`.

### 2.2 Per-sample Hessian trace

For input $x$ let $p = \mathrm{softmax}(f_\theta(x))$, $S = \mathrm{diag}(p) - p p^\top$,
$h = \mathrm{ReLU}(W_1 x + b_1)$ and $m = \mathbb{1}[W_1 x + b_1 > 0]$. Inside a
ReLU region the logits are linear in each layer's parameters, so the diagonal
of the cross-entropy Hessian equals its Gauss-Newton diagonal, and the trace has
the closed form

$$
T_\theta(x) = \mathrm{Tr}\,\nabla^2_\theta \mathrm{CE}
  = (\lVert h \rVert^2 + 1)\,\mathrm{Tr}(S)
  + (\lVert x \rVert^2 + 1) \sum_j m_j\, w_{2,j}^\top S\, w_{2,j},
$$

where $w_{2,j}$ is column $j$ of $W_2$. It does not depend on the label. The
code (`hessian_trace`) is checked against the autograd Hessian in the tests.
The paper uses a Hutchinson estimate instead; ours is exact.

### 2.3 Canonicalization

ReLU is positively homogeneous, so for any $a \in \mathbb{R}^{128}_{>0}$ the map

$$
(w_{1,j},\, b_{1,j},\, w_{2,j}) \mapsto (a_j w_{1,j},\; a_j b_{1,j},\; w_{2,j} / a_j)
\quad \text{for every hidden unit } j
$$

leaves $f_\theta$ unchanged. Here $w_{1,j}$ is row $j$ of $W_1$. The paper's
red-star solutions $(a^{-1} W_1, a W_2)$ are the special case of a single
shared $a$. `canonicalize` picks the one representative of this orbit in which
every unit is balanced:

$$
a_j = \sqrt{\frac{\lVert w_{2,j} \rVert}{\lVert (w_{1,j}, b_{1,j}) \rVert}}
\;\;\Rightarrow\;\;
\lVert (a_j w_{1,j}, a_j b_{1,j}) \rVert = \lVert w_{2,j} / a_j \rVert
= \sqrt{\lVert (w_{1,j}, b_{1,j}) \rVert\, \lVert w_{2,j} \rVert}.
$$

Call the result $\bar\theta = \mathrm{canon}(\theta)$. It computes the same
function as $\theta$, and every rescaled copy of $\theta$ has the same
$\bar\theta$. Rebalancing is the cheap attack the paper gives against the
rescaling defense (its Example 1), so $\bar\theta$ is what an attacker would
fine-tune after undoing any rescaling.

### 2.4 The penalties $\Omega$

| Penalty | Raw $\Omega(\theta)$ | Canonical $\Omega(\theta)$ |
| --- | --- | --- |
| distance | $\lVert \theta - \theta_0 \rVert^2$ | $\lVert \bar\theta - \bar\theta_0 \rVert^2$ |
| hessian | $\frac{1}{|B|}\sum_{x \in B} T_\theta(x)$ | $\frac{1}{|B|}\sum_{x \in B} T_{\bar\theta}(x)$ |

$B$ is the current minibatch for penalized SGD and the probe set $X_P$ for
projected ascent. `--canonical` selects the right-hand column. Canonical
$\Omega$ is constant along each rescaling orbit, so its gradient has no
rescaling component: no run can raise it by rescaling. Gradients flow through
`canonicalize` by autograd.

The plotted y-values are relative and measured on $X_E$:

- distance: $\lVert \theta - \theta_0 \rVert / \lVert \theta_0 \rVert$
  (canonical: $\lVert \bar\theta - \bar\theta_0 \rVert / \lVert \bar\theta_0 \rVert$);
- hessian: $\exp\big(\frac{1}{|X_E|}\sum_x \log \frac{T_\theta(x)}{T_{\theta_0}(x)}\big)$
  (canonical: the same with $\bar\theta$ and $\bar\theta_0$).

## 3. Method A: penalized SGD (the paper's Figure 2; blue lines)

Minimize one joint objective with plain SGD:

$$
L_\lambda(\theta) = \frac{1}{10\,|B|} \sum_{x \in B} \lVert f_\theta(x) - f_{\theta_0}(x) \rVert^2
  \;-\; \lambda\, \Omega(\theta).
$$

Each step draws a fresh minibatch $B$ of 128 training images, computes
$\nabla L_\lambda$, clips its norm to 1, and applies
$\theta \leftarrow \theta - \eta \nabla L_\lambda$. Each run lasts 2000 steps.
The sweep covers 29 values of $\lambda$ log-spaced in $[10^{-4}, 10^3]$,
$\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$, and both penalties: 174 runs. There
is no hard constraint. $\lambda$ only sets the exchange rate between MSE and
$\Omega$, and each run ends wherever that trade-off leads.

Because of clipping, a run moves at most $2000\,\eta$ in parameter norm. That
caps the distance panel at about 0.15, 1.5 and 15 times $\lVert\theta_0\rVert$,
one plateau per learning rate.

## 4. Method B: projected ascent (orange lines)

Solve the constrained problem directly:

$$
\max_\theta\; \Omega(\theta) \quad \text{s.t.} \quad c_{X_P}(\theta) \le \varepsilon,
$$

with canonical $\Omega$ evaluated on $X_P$ and $\varepsilon \in \{10^{-4}, 10^{-3}, 10^{-2}\}$.

### 4.1 The Gauss-Newton matrix

Let $J(\theta) \in \mathbb{R}^{10N \times P}$ be the Jacobian of the stacked
probe logits $f_\theta(X_P)$ with respect to $\theta$. Here $P = 101{,}770$ is
the number of parameters. Define

$$
G(\theta) = \tfrac{1}{N} J^\top J, \qquad G_\mu = G + \mu I, \quad \mu = 10^{-3}.
$$

For a small step $\delta$, the change in summed squared probe error is
$\approx N\, \delta^\top G \delta$ (plus a term linear in the current residual).
So directions with small $\delta^\top G \delta$ barely move the probe outputs,
and directions in $\ker J$ do not move them at all, to first order.

$J$ is never formed. `jacobian_products` computes $Jv$ and $J^\top u$ in closed
form, linearized at the current ReLU mask:

$$
Jv = \big(m \odot (V_1 x + v_{b_1})\big) W_2^\top + h V_2^\top + v_{b_2},
\qquad
J^\top u:\;
\begin{cases}
\delta = (u W_2) \odot m \\
\partial W_1 = \delta^\top x,\ \partial b_1 = \textstyle\sum \delta \\
\partial W_2 = u^\top h,\ \partial b_2 = \textstyle\sum u
\end{cases}
$$

batched over $X_P$. Systems $G_\mu z = r$ are solved by conjugate gradient
(`gauss_newton_solve`) from $z = 0$, for at most 30 iterations. The result is
an approximate solve, not an exact one.

### 4.2 One iteration

State: current $\theta_k$ (feasible) and step length $\eta_k$ ($\eta_0 = 0.1$).

1. **Ascent direction.** Compute $g = \nabla_\theta \Omega(\theta_k)$ with autograd,
   then $u = G_\mu(\theta_k)^{-1} g$ by CG.
   $G_\mu^{-1}$ leaves components of $g$ in near-null directions of $G$ almost
   untouched (scaled by about $1/\mu$) and shrinks components that would move the
   outputs. So $u$ is approximately $g$ projected onto the function-preserving
   directions. It solves the trust-region problem
   $\max_\delta g^\top \delta$ s.t. $\delta^\top G_\mu \delta \le r^2$, up to scale.
2. **Trial step.** $\tilde\theta = \theta_k + \eta_k\, u / \lVert u \rVert$.
   The step has length exactly $\eta_k$ in parameter space.
3. **Retraction.** A finite step still adds some error through curvature and the
   change of ReLU masks. Repeat twice, re-linearizing at the current $\tilde\theta$:

   $$
   r = f_{\tilde\theta}(X_P) - f_{\theta_0}(X_P), \qquad
   b = \tfrac{1}{N} J(\tilde\theta)^\top r, \qquad
   \tilde\theta \leftarrow \tilde\theta - G_\mu(\tilde\theta)^{-1} b.
   $$

   Each pass is a damped Gauss-Newton step on the probe MSE alone. It pulls the
   function back toward $f_{\theta_0}$ and ignores $\Omega$.
4. **Accept or reject.** Accept $\tilde\theta$ if and only if both hold:

   $$
   c_{X_P}(\tilde\theta) \le \varepsilon
   \quad\text{and}\quad
   \Omega(\tilde\theta) > \Omega(\theta_k).
   $$

   - **Accept:** $\theta_{k+1} = \tilde\theta$ and $\eta_{k+1} = 1.5\,\eta_k$.
   - **Reject:** $\eta_k \leftarrow \eta_k / 2$ and go back to step 2 with the
     same $u$. After 12 consecutive rejections the run stops with status
     `stalled`: no step down to $\eta_k / 2048$ keeps the constraint while
     raising $\Omega$, so the run is at the constraint boundary.

The run stops after 100 accepted iterations or a stall. Every 5 iterations the
metrics of Section 2 are logged on $X_E$.

**Guarantee.** Every accepted iterate satisfies $c_{X_P} \le \varepsilon$
exactly, and $\Omega$ increases monotonically. Nothing is guaranteed off the
probe set; see Section 6.

### 4.3 Why it differs from penalized SGD

Near $\theta_0$ the MSE is quadratic in $\delta$, so in Method A its restoring
gradient starts at zero and grows only after the function has already moved.
Meanwhile $\nabla\Omega$ points mostly in directions that do move the function.
Method B uses the curvature $G$ of the MSE before moving: it steers
$\nabla\Omega$ into directions that the probe outputs do not see, repairs the
remaining drift, and enforces $\varepsilon$ as a hard limit.

## 5. Results (seed 0)

Best canonical $\Omega$ reached at about the same held-out drift (x-axis
metric). For Method A this is the best over all 87 runs of that penalty, taken
over every logged point with relMSE at or below the cap.

| $\Omega$ | relMSE cap | Method A | Method B, $\varepsilon$ = cap | Test acc. (B) |
| --- | --- | ---: | ---: | ---: |
| hessian | $10^{-4}$ | 1.1× | 4.7× (relMSE 1.5e-4) | 97.4% |
| hessian | $10^{-3}$ | 1.4× | 7.3× (relMSE 1.1e-3) | 97.5% |
| hessian | $10^{-2}$ | 3.3× | 26× (relMSE 8.3e-3) | 95.8% |
| distance | $10^{-4}$ | 0.10 | 2.0 (relMSE 2.6e-4) | 97.2% |
| distance | $10^{-3}$ | 0.32 | 3.1 (relMSE 3.0e-3) | 96.9% |
| distance | $10^{-2}$ | 0.75 | 5.8 (relMSE 3.8e-2) | 94.1% |

$\theta_0$ test accuracy is 97.3%. All Hessian runs of Method B stalled (after
67, 34 and 28 iterations); all distance runs used the full 100.

## 6. Caveats

- **The constraint holds only on the probe set.** $\ker J(X_P)$ contains
  directions that change $f$ on other inputs. Held-out pooled drift
  $c_{X_E}$ is about 1.5–2× $\varepsilon$ for the Hessian runs, but about 10×
  $\varepsilon$ for the distance runs. With a 2048-image probe the leak was far
  larger, which is why we use 8192 (16384 was tighter but more than twice the
  cost). A likely source of the distance gain is MNIST-specific null space,
  such as weights on always-background border pixels compensated by the bias.
  We have not verified this.
- **Canonicalization removes only per-unit rescaling.** Other cheap
  reparameterizations, such as pruning dead units or zeroing weights on
  constant pixels, are not quotiented out.
- **$\Omega$ is a proxy.** A larger Hessian trace or distance after rebalancing
  does not show that fine-tuning is harder. That needs a relearning attack on
  the Method B models, compared with $\theta_0$.
- **One seed, one architecture.**

## 7. Reproduce

From `examples/tamper-resistance`:

```bash
export HF_HOME=$SCRATCH/huggingface
# Figure 2 reproduction (raw Omega), about 20 min.
.venv-eval/bin/python scripts/analysis/toy-experiments/get_locking_toy.py
.venv-eval/bin/python scripts/analysis/toy-experiments/plot_locking_toy.py
# Method A with canonical Omega, about 20 min.
.venv-eval/bin/python scripts/analysis/toy-experiments/get_locking_toy.py --canonical \
  --output artifacts/analysis/locking-toy-canonical.json
# Method B with canonical Omega, about 40 min.
.venv-eval/bin/python scripts/analysis/toy-experiments/get_locking_toy.py --optimizer projected --canonical \
  --output artifacts/analysis/locking-toy-projected.json
.venv-eval/bin/python scripts/analysis/toy-experiments/plot_locking_toy.py --canonical \
  --input artifacts/analysis/locking-toy-canonical.json \
  --projected artifacts/analysis/locking-toy-projected.json \
  --output artifacts/figures/locking-toy-canonical.pdf
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--probe-examples` | 8192 | $N$, size of $X_P$ |
| `--tolerances` | 1e-4 1e-3 1e-2 | values of $\varepsilon$ |
| `--projected-steps` | 100 | maximum accepted iterations |
| `--projected-step` | 0.1 | $\eta_0$ |
| `--max-backtracks` | 12 | halvings before a stall |
| `--retraction-steps` | 2 | Gauss-Newton passes in step 3 |
| `--gn-damping` | 1e-3 | $\mu$ |
| `--cg-iterations` | 30 | CG iterations per solve |
| `--init-noise` | 1e-4 | std of $\xi$ |
