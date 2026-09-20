# dax
Dynamical systems, filtering, smoothing, and identification in Jax

```
pip install git+https://github.com/PredictiveScienceLab/dax.git
```

## Probability-model conventions

`StochasticDifferentialEquation.diffusion` returns the vector of diagonal
diffusion amplitudes. For a time step `dt`, `EulerMaruyama` therefore uses a
Gaussian transition with mean `x + drift * dt` and diagonal variance
`diffusion**2 * dt`. The Gaussian initial-state and observation models include
their normalizing constants, so reported log densities and marginal-likelihood
estimates are on the correct absolute scale.

`ExpectationMaximization` implements a particle Monte Carlo approximation. It
draws smoothing trajectories once at the beginning of each M-step and holds
them fixed during the inner optimization. Finite particle approximations and
numerical M-steps do not guarantee monotonic ascent of the observed-data
likelihood.

## Development

```
pip install -e ".[test]"
pytest -q
```
