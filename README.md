# Diffusive SEIRS: spatial heterogeneity and R0

NSRI Summer Research Hackathon 2026

**Track:** AI, Data Science & Computing

**Status:** Solo. Correctness checks passed, first sweep experiment run and diagnosed, next step scoped for future work.

## Repository structure

```
seirs-spatial-heterogeneity-research/
├── .git/
├── .gitignore
├── .vscode/
├── notebooks/
│   ├── nb1_global_correctness.ipynb
│   └── nb2_on_beta.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── README.md
├── src/
│   └── core/
│       ├── SEIRSParameters.py
│       ├── solvers.py
│       └── utils.py
├── sshr/
└── tests/
    └── test_sanity.py
```

## Research question

In a 2D diffusive SEIRS model with mass action incidence and uniform baseline β0, γ0, a localized Gaussian perturbation is added to β. How does R0 (the principal eigenvalue of the coupled next generation operator) respond to that perturbation's amplitude and width, and does the response cross from peak dominated (localized, slow diffusion limit) to mass dominated (well mixed, fast diffusion limit) as diffusion increases?

These two limiting regimes are established results (Allen, Bolker, Lou and Nevai 2008 for SIS; Song, Lou and Xiao 2019 for this exact SEIRS eigenvalue problem). The crossover between them, for a specific amplitude/width decomposition, is not, and is the actual object of study here.

## Model

Mass action incidence, Neumann boundary conditions, following Castellano, Salako and Xue (2026):

```
S_t = d_S ΔS + αR − β(x,y) S I
E_t = d_E ΔE + β(x,y) S I − σE
I_t = d_I ΔI + σE − γ(x,y) I
R_t = d_R ΔR + γ(x,y) I − αR
```

R0 is the principal eigenvalue μ\* of the coupled elliptic problem:

```
0 = d_E ΔΨ − σΨ + μ β Φ
0 = d_I ΔΦ + σΨ − γΦ
```

with R0 = N / (|Ω| μ\*).

**Perturbation setup:** β = β0 + A*β · Gaussian(x0, σ*β), added to a nonzero uniform background. γ stays uniform for this experiment.

## Correctness checks (completed)

Before any sweep result is trusted, the eigensolver was validated against three exact analytical limits from the source papers:

1. Global β rescaling linearity: R0(nβ) = n · R0(β) (Castellano, Salako and Xue 2026, Prop. 2.10(i))
2. Well mixed limit (d_E, d_I large): R0 → (N/|Ω|)(∫β/∫γ) (Prop. 2.9(v))
3. Localized limit (d_E, d_I small): R0 → (N/|Ω|) max(β/γ) (Prop. 2.9(iii))

All three passed. Additional checks (the off-diagonal mixed regime and flatness under d_S, d_R perturbation) are planned but not yet run; see Future work.

## β patch sweep: current state

**Setup:** a dimensionless diffusion parameter ε = sqrt(d/γ0) / σ*β was used to compare diffusion length against patch width, with d = d_E = d_I swept along the diagonal. Five target values, ε = 0.1, 0.3, 1, 3, 10, each converted to a corresponding d. At each ε, A*β and σ*β were swept over a grid, and R0 was fit against two candidate laws: linear in A*β (predicted for the localized/peak dominated regime) and linear in A*β·σ*β² (predicted for the well mixed/mass dominated regime).

**Result:** the peak law fit R0 well at every ε tested (R² approximately 0.99 to 1.0, from ε = 0.1 through ε = 10). The mass law fit poorly throughout (R² approximately 0.35 to 0.39) and did not improve with increasing ε. No crossover was observed within the tested range.

**Diagnosis:** ε as defined compares diffusion length to patch width (σ*β), which governs when the patch itself gets smeared out. It does not directly track when the eigenfunction becomes flat across the whole domain, which is what the well mixed limit actually requires. That second comparison is ε_domain = sqrt(d/γ0) / L. Because σ*β was deliberately kept much smaller than L (to avoid boundary contamination, per the numerical validation plan), ε_domain stayed well below 1 across the entire tested range, even at ε = 10. In other words, the sweep likely smeared the patch without ever approaching genuine domain scale mixing.

## Conclusion of this step

The hypothesized crossover was not observed, but the diagnosis above means this is inconclusive rather than a negative result: absence of evidence within an apparently undertested diffusion range, not evidence of absence. This has not yet been confirmed directly. The next required step, before any claim about the crossover's existence or shape, is a direct diagnostic: fix one (A*β, σ*β) pair and sweep d over a much wider range, explicitly pushing ε_domain past 1, and check whether R0 moves toward the well mixed formula. That work is scoped for future work rather than continued here.

## Limitations, known going in

- The eigenvalue solve is a numerical approximation of a continuous spectral problem; a grid convergence check is still needed before trusting any scaling result.
- Restricted to mass action incidence, as in the source paper; findings may not carry over to frequency dependent (standard) incidence.
- Solo project; no external code review beyond self checks against the paper's proven limiting cases.
- The ε used in the sweep tracks patch scale smearing, not domain scale mixing; this distinction was identified only after the first sweep and is the main open issue going into future work.

## Future work

- Direct wide range diagnostic: sweep d alone at fixed (A*β, σ*β), reaching ε_domain well past 1, to determine whether the crossover appears once the range is corrected.
- Remaining correctness checks: the off-diagonal mixed regime and R0 flatness under d_S, d_R perturbation.
- Grid convergence check on a representative sweep point.
- γ patch sweep (same battery as the β sweep, on γ instead), and comparison of crossover shape between the two, since γ enters the eigenvalue problem as a subtracted diagonal term rather than a multiplicative coupling.
- Placement corollary: fix a β hotspot and a γ ("immunization") patch, sweep their relative offset, and locate the R0 minimizing placement.

## Sources

- Castellano, K. G., Salako, R. B., Xue, S. (2026). "On the dynamics of a diffusive SEIRS epidemic model." _European Journal of Applied Mathematics_, published online 13 April 2026. Primary source: mass action model, R0 definition, amplitude rescaling result used as a validation target.
- Song, P., Lou, Y., Xiao, Y. (2019). "A spatial SEIRS reaction-diffusion model in heterogeneous environment." _Journal of Differential Equations_ 267, 5084 to 5114. Direct precursor (standard incidence): same eigenvalue apparatus, general bound min(β/γ) ≤ R0 ≤ max(β/γ), localized and well mixed limits, a third mixed regime, R0 independence from d_S and d_R.
- Allen, L. J. S., Bolker, B. M., Lou, Y., Nevai, A. L. (2008). "Asymptotic profiles of the steady states for an SIS epidemic reaction-diffusion model." _Discrete and Continuous Dynamical Systems, Series A_ 21(1), 1 to 20. Origin of the principal eigenvalue R0 formulation and the two limiting regimes.
- NSRI Summer Research Hackathon 2026, Final Participant Guide.

## AI use log, running

- Claude used for: locating and reading the Castellano, Salako and Xue (2026) paper, planning the finite difference/eigenvalue numerical approach, drafting project documentation.
- Claims traced to fetched papers are attributed above. All code and numerical results are my own implementation and execution.
- Day 2: Claude stress tested the two candidate directions into one unified question, checked novelty against Allen, Bolker, Lou and Nevai (2008) and Song, Lou and Xiao (2019), and narrowed the claimed contribution accordingly.
- Day 3: Claude helped derive the dimensionless diffusion parameter ε used to design the sweep, built the sweep and plotting harness code, and, after the first sweep produced a flat result, helped diagnose that the tested ε range likely never reached domain scale mixing (the ε versus ε_domain distinction), reframing the result as inconclusive rather than negative pending a direct wide range check.
