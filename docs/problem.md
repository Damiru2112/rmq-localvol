# Research Problem

## 1. Background: Recursive Marginal Quantization

Consider a one-dimensional diffusion

```
dS_t = b(S_t) dt + σ(S_t) dW_t
```

Optimal quantization replaces the continuous law of $S_t$ at a fixed time with the best $N$-point discrete approximation: a grid $\{x_1, \dots, x_N\}$ and a set of companion (Voronoi) weights, chosen to minimize the $L^2$ distortion between the true distribution and the discrete one. This is a classical idea from signal processing and information theory, repurposed for numerical probability.

**Recursive marginal quantization (RMQ)**, introduced by Pagès and Sagna (2015) for the Euler scheme, extends this idea across time. Rather than quantizing the full path of $S_t$, RMQ quantizes each time-marginal *recursively*: the optimal grid at step $k+1$ is built directly from the optimal grid and transition law at step $k$, using the known conditional structure of the discretization scheme (e.g. Euler: conditionally Gaussian increments). At each step, the grid and weights are found by Newton's method applied to the gradient of the distortion functional, which has a closed form whenever the one-step conditional law is tractable.

The payoff is a fast, deterministic alternative to Monte Carlo: once the recursive grids and transition probabilities are built, European option prices follow from a single backward pass with no simulation variance, and the same transition structure supports early-exercise products (Bermudan/American) via standard backward induction, and barrier products via absorption/reflection at the boundary.

## 2. The Gap

Higher-order discretization schemes price more accurately for a given number of time steps than Euler. McWalter, Rudd, Kienitz, and Platen (2018) generalize RMQ to the Milstein scheme, whose one-step update is

```
S_{k+1} = S_k + b(S_k) Δ + σ(S_k) ΔW_k + ½ σ(S_k) σ'(S_k) (ΔW_k² − Δ)
```

This is a *quadratic function of a Gaussian increment* — and that quadratic structure is exactly what keeps the conditional law closed-form (a non-central generalization of the Gaussian case), which in turn is what makes the distortion gradient tractable for Newton's method. But the construction requires $\sigma'(S) = \partial \sigma / \partial S$ explicitly, evaluated at each quantization point, at every time step.

McWalter et al. demonstrate Milstein RMQ on the constant elasticity of variance (CEV) model, $\sigma(S) = \sigma S^{\beta}$, where $\sigma'$ is available in closed form. Callegaro, Fiorin, and Grasselli (2015) apply RMQ in a local-volatility setting, but to a parametric surface (quadratic normal volatility) that is likewise analytically differentiable. In both cases, the method's reach is limited by the requirement of an analytic derivative of the volatility function.

A general, market-calibrated local volatility surface — extracted from option prices via Breeden–Litzenberger and not constrained to any convenient parametric form — has no closed-form $\sigma'$. Extending Milstein RMQ to this setting was flagged explicitly as future work by Callegaro, Fiorin, and Grasselli, and remains open.

## 3. Contribution

This project implements **Milstein RMQ driven by a numerically differentiated, market-calibrated local volatility surface**, removing the dependence on an analytic $\partial \sigma / \partial S$.

The pipeline:

1. **Calibration.** Fit an SSVI parametrization to market implied volatilities across strikes and maturities.
2. **Density extraction.** Apply Breeden–Litzenberger to the calibrated implied-vol surface to recover risk-neutral marginal densities, using higher-order finite-difference kernels for numerical stability of the second-derivative extraction.
3. **Local volatility surface.** Convert the extracted densities (via Dupire) into a local volatility surface $\sigma(S, t)$ defined on a grid, with no analytic form.
4. **Numerical differentiation.** Compute $\partial \sigma / \partial S$ directly from this gridded surface, replacing the analytic term in the Milstein RMQ recursion.
5. **Consistency check.** Validate the resulting quantized marginals against an optimal-transport criterion, in addition to the Monte Carlo benchmark used throughout.

The contribution is narrow but load-bearing: it lifts higher-order RMQ off special parametric models (CEV, quadratic normal vol) onto arbitrary calibrated market surfaces, which is the setting practitioners actually need.

## 4. Scope of This Repository

This repository packages a tractable, demonstrable slice of the broader research. Monte Carlo simulation and convergence/error plots are the **primary correctness evidence** throughout; real market data (SPX/SPY) is **applicability evidence**, not correctness evidence, and is introduced only at the final calibration milestone.

Pricing targets: European options (closed-form/Monte Carlo baseline for correctness), and Bermudan options via backward induction on the quantization grid — comparing continuation value against exercise value at each node, exactly as in a binomial tree but on an optimally-placed grid rather than a fixed lattice. American options are treated as the dense-exercise-date limit of the Bermudan case.

**Milestones:**

- **M1 — Quantization primitive.** Optimal 1-D quantizer via Newton's method on the distortion functional; validated against known Gaussian quantizer tables.
- **M2 — Euler RMQ.** European option pricing on the Euler scheme, validated against Monte Carlo.
- **M3 — Milstein RMQ on CEV.** Analytic ground truth, reproducing McWalter et al. (2018); European and Bermudan pricing via backward induction, validated against Monte Carlo.
- **M4 — Numerical local volatility extension.** SSVI calibration → Breeden–Litzenberger extraction → numerical $\partial \sigma / \partial S$ → Milstein RMQ on the resulting surface; European and Bermudan pricing, validated against Monte Carlo and an optimal-transport consistency check.

## References

- Pagès, G. & Sagna, A. (2015). *Recursive marginal quantization of the Euler scheme of a diffusion process.* Applied Mathematical Finance, 22(5), 463–498.
- McWalter, T. A., Rudd, R., Kienitz, J. & Platen, E. (2018). *Recursive marginal quantization of higher-order schemes.* Quantitative Finance, 18(4), 693–706. (arXiv:1701.02681)
- Callegaro, G., Fiorin, L. & Grasselli, M. (2015). *Quantized calibration in local volatility.* Risk Magazine, 28(4), 62–67.
- Callegaro, G., Fiorin, L. & Grasselli, M. (2017). *Pricing via recursive quantization in stochastic volatility models.* Quantitative Finance, 17(6), 855–872.