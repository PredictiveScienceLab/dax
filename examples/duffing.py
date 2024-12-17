"""Stochastic Duffing oscillator example.

Author:
    Ilias Bilionis

Date:
    11/26/2024

"""

import dax
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import optax


class DuffingControl(dax.ControlFunction):

    omega: jax.Array

    def __init__(self, omega):
        self.omega = jnp.array(omega)
    
    def _eval(self, t):
        # This is to avoid training the parameter
        omega = jax.lax.stop_gradient(self.omega)
        return jnp.cos(omega * t)

    
class Duffing(dax.StochasticDifferentialEquation):

    alpha: jax.Array
    beta: jax.Array
    delta: jax.Array

    gamma: jax.Array
    log_sigma_x: jax.Array
    log_sigma_v: jax.Array

    @property
    def sigma_x(self):
        return jnp.exp(self.log_sigma_x)

    @property
    def sigma_v(self):
        return jnp.exp(self.log_sigma_v)

    def __init__(self, alpha, beta, gamma, delta, sigma_x, sigma_v, u):
        super().__init__(control_function=u)
        self.alpha = jnp.array(alpha)
        self.beta = jnp.array(beta)
        self.delta = jnp.array(delta)

        self.gamma = jnp.array(gamma)
        self.log_sigma_x = jnp.log(sigma_x)
        self.log_sigma_v = jnp.log(sigma_v)
    
    def drift(self, x, u):
        # We won't train gamma
        gamma = jax.lax.stop_gradient(self.gamma)
        return jnp.array([x[1], -self.delta * x[1] - self.alpha * x[0] - self.beta * x[0] ** 3 + gamma * u])
    
    def diffusion(self, x, u):
        # We won't train sigma_x and sigma_v
        sigma_x = jax.lax.stop_gradient(self.sigma_x)
        sigma_v = jax.lax.stop_gradient(self.sigma_v)
        return jnp.array([sigma_x, sigma_v])
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("paper")
    sns.set_style("ticks")

    omega = 1.2
    alpha = -1.0
    beta = 1.0
    delta = 0.3
    gamma = 0.5
    sigma_x = 0.01
    sigma_v = 0.05
    u = DuffingControl(omega)
    true_sde = Duffing(alpha, beta, gamma, delta, sigma_x, sigma_v, u)
    print(true_sde)

    # True initial conditions
    true_x0 = jnp.array([1.0, 0.0])

    # Observation model
    observation_function = dax.SingleStateSelector(0)
    # Observation variance
    s = 0.1
    true_likelihood = dax.GaussianLikelihood(s, observation_function)
    print(true_likelihood)

    # Generate synthetic data
    t0 = 0.0
    t1 = 40.0
    dt = 0.1
    key = jr.PRNGKey(0)
    sol = true_sde.sample_path(key, t0, t1, true_x0, dt=dt, dt0=0.05)
    key, subkey = jr.split(key)
    xs = sol.ys
    ts = sol.ts
    us = u(ts)
    keys = jr.split(key, xs.shape[0])
    ys = true_likelihood.sample(xs, us, keys)
    
    # Training data
    t_train = 200
    ts_train = ts[:t_train]
    ts_train_w_init = jnp.concatenate([jnp.array([ts[0] - dt]), ts_train])
    ys_train = ys[:t_train]
    us_train = us[:t_train]

    # Test data
    ts_test = ts[t_train-1:-1]
    us_test = us[t_train:]

    fig, ax = plt.subplots()
    ax.plot(ts, xs[:, 0], label='r$X_t$')
    ax.plot(ts, xs[:, 1], label='r$\\dot{X}_t$')
    ax.plot(ts, ys, 'k.', label='r$Y_t$', alpha=0.5)
    ax.legend(frameon=False)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    sns.despine(trim=True)
    plt.show()

    # Make the statespace model
    x0 = dax.DiagonalGaussian(
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, 1.0])
    )
    ssm = dax.StateSpaceModel(
        dax.DiagonalGaussian(
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 1.0])),
        dax.EulerMaruyama(true_sde, dt=dt),
        true_likelihood
    )

    # Ready to do the filtering
    print('Filtering...')
    filter = dax.BootstrapFilter(num_particles=1_000)
    key, subkey = jr.split(key)
    pas, log_L = filter.filter(ssm, us_train, ys_train, subkey)

    key, subkey = jr.split(key)
    lower, median, upper = pas.get_credible_interval(subkey)

    # Predict the future
    print('Predicting...')
    key, subkey = jr.split(key)
    pas_test = ssm.predict(pas[-1], us_test, subkey)
    print(pas_test)

    key, subkey = jr.split(key)
    lower_test, median_test, upper_test = pas_test.get_credible_interval(subkey)

    # # Smooth
    print('Smoothing...')
    smoother = dax.BootstrapSmoother(100)
    key, subkey = jr.split(key)
    trajectories = smoother.smooth(ssm, pas, us_train, ys_train, key)
    smoothed_lower, smoothed_median, smoothed_upper = trajectories.get_credible_interval()

    # Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(ts, xs[:, 0], label='True')
    ax[0].plot(ts_train, ys_train, 'k.')

    ax[0].plot(ts_train_w_init, median[:, 0], 'r-', label='Filter')
    ax[0].fill_between(ts_train_w_init, lower[:, 0], upper[:, 0], color='r', alpha=0.2)
    
    ax[0].plot(ts_test, median_test[:, 0], 'g-', label='Prediction')
    ax[0].fill_between(ts_test, lower_test[:, 0], upper_test[:, 0], color='g', alpha=0.2)
    ax[0].plot(ts_test, pas_test.particles[:, 0::1000, 0], '-.', color='green', lw=0.5)

    ax[0].set_ylim(-2, 2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Value')
    ax[0].legend(frameon=False, loc='best')

    for a in ax:
        a.axvline(ts[200], linestyle='--', color='black', alpha=0.5)

    ax[1].plot(ts, xs[:, 1], label='True')
    ax[1].plot(ts_train_w_init, median[:, 1], 'r-', label='Filter')
    ax[1].fill_between(ts_train_w_init, lower[:, 1], upper[:, 1], color='r', alpha=0.2)
    
    ax[1].plot(ts_test, median_test[:, 1], 'g-', label='Prediction')
    ax[1].fill_between(ts_test, lower_test[:, 1], upper_test[:, 1], color='g', alpha=0.2)
    ax[1].plot(ts_test, pas_test.particles[:, 0::1000, 1], '-.', color='green', lw=0.5)

    ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Value')
    ax[1].legend(frameon=False, loc='best')

    sns.despine(trim=True)

    # Plot smoothed results
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(ts_train, xs[:t_train, 0], label='True')
    ax[0].plot(ts_train, ys_train, 'k.')

    ax[0].plot(ts_train_w_init, median[:, 0], 'r-', label='Filter')
    ax[0].fill_between(ts_train_w_init, lower[:, 0], upper[:, 0], color='r', alpha=0.2)

    ax[0].plot(ts_train_w_init, smoothed_median[:, 0], 'g-', label='Smoother')
    ax[0].fill_between(ts_train_w_init, smoothed_lower[:, 0], smoothed_upper[:, 0], color='g', alpha=0.2)

    ax[0].set_ylim(-2, 2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Value')
    ax[0].legend(frameon=False, loc='best')

    ax[1].plot(ts_train, xs[:t_train, 1], label='True')
    ax[1].plot(ts_train_w_init, median[:, 1], 'r-', label='Filter')
    ax[1].fill_between(ts_train_w_init, lower[:, 1], upper[:, 1], color='r', alpha=0.2)

    ax[1].plot(ts_train_w_init, smoothed_median[:, 1], 'g-', label='Smoother')
    ax[1].fill_between(ts_train_w_init, smoothed_lower[:, 1], smoothed_upper[:, 1], color='g', alpha=0.2)

    ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Value')
    ax[1].legend(frameon=False, loc='best')

    sns.despine(trim=True)

    plt.show()

    # Test particle MCMC

    # Start a model from the wrong parameters
    alpha_wrong = -0.9
    beta_wrong = 0.8
    delta_wrong = 0.2
    s_wrong = 1.0


    u = DuffingControl(omega)
    sde = Duffing(alpha_wrong, beta_wrong, gamma, delta_wrong, sigma_x, sigma_v, u)

    class GaussianLikelihood(dax.Likelihood):

        log_s: jax.Array

        @property
        def s(self):
            return jnp.exp(self.log_s)

        def __init__(self, s):
            self.log_s = jnp.log(s)

        def _log_prob(self, y, x, u):
            return -0.5 * jnp.sum( ((y - x[0]) / self.s) ** 2) - self.log_s
        
        def _sample(self, x, u, key):
            return x[0] + self.s * jr.normal(key, shape=x.shape)

    ssm = dax.StateSpaceModel(
        dax.DiagonalGaussian(
            jnp.array([0.0, 1.0]),
            jnp.array([2.0, 2.0])),
        dax.EulerMaruyama(sde, dt=dt),
        GaussianLikelihood(s_wrong)
    )

    filter = dax.BootstrapFilter(num_particles=100)

    class SSMPrior(dax.Prior):
            
            def log_prob(self, ssm):
                # Flat prior over all parameters
                return 0.0
    
            def sample(self, key):
                return None
    
    theta = {
        'alpha': -0.9,
        'beta': 0.92,
        'delta': 0.39,
        'log_s': 0.15,
        'x0': {
            'mean': jnp.array([0.0, 0.0]),
            'log_sigma': jnp.array([1.0, 1.0])
        }
    }

    def ssm_from_theta(theta):
        return dax.StateSpaceModel(
            dax.DiagonalGaussian(
                theta['x0']['mean'],
                jnp.exp(theta['x0']['log_sigma'])),
            dax.EulerMaruyama(Duffing(theta['alpha'], theta['beta'], gamma, theta['delta'], sigma_x, sigma_v, u), dt=dt),
            GaussianLikelihood(jnp.exp(theta['log_s']))
        )
    

    prior = SSMPrior()
    mcmc = dax.ParticleMCMC(prior, filter, 0.01, ssm_from_theta)

    state = (theta, filter.filter(ssm_from_theta(theta), us_train, ys_train, key)[1], 0.0, key, us_train, ys_train)
    theta, (accept, log_Ls, thetas) = mcmc.run(theta, us_train, ys_train, 10_000, key)
    
    fig, ax = plt.subplots()
    ax.plot(log_Ls)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Likelihood')
    sns.despine(trim=True)

    plt.savefig('duffing_mcmc.png', dpi=300)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(thetas['alpha'])
    ax[0].axhline(alpha, color='black', linestyle='--', alpha=0.5)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel(r'$\alpha$')

    ax[1].plot(thetas['beta'])
    ax[1].axhline(beta, color='black', linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel(r'$\beta$')

    ax[2].plot(thetas['delta'])
    ax[2].axhline(delta, color='black', linestyle='--', alpha=0.5)
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel(r'$\delta$')

    sns.despine(trim=True)

    plt.savefig('duffing_mcmc_params.png', dpi=300)

    # Do the histograms of the parameters
    skip = 1_000
    alpha_keep = thetas['alpha'][skip:]
    beta_keep = thetas['beta'][skip:]
    delta_keep = thetas['delta'][skip:]

    fig, ax = plt.subplots(3, 1)
    ax[0].hist(alpha_keep, bins=20, color='blue', alpha=0.5)
    ax[0].axvline(alpha, color='black', linestyle='--', alpha=0.5)
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel('Frequency')

    ax[1].hist(beta_keep, bins=20, color='blue', alpha=0.5)
    ax[1].axvline(beta, color='black', linestyle='--', alpha=0.5)
    ax[1].set_xlabel(r'$\beta$')
    ax[1].set_ylabel('Frequency')

    ax[2].hist(delta_keep, bins=20, color='blue', alpha=0.5)
    ax[2].axvline(delta, color='black', linestyle='--', alpha=0.5)
    ax[2].set_xlabel(r'$\delta$')
    ax[2].set_ylabel('Frequency')

    sns.despine(trim=True)

    plt.savefig('duffing_mcmc_hist.png', dpi=300)

    plt.show()


    quit()

    # Test the EM algorithm

    # Start a model from the wrong parameters
    alpha_wrong = -0.0
    beta_wrong = 0.2
    delta_wrong = 0.1
    s_wrong = 2.0

    # alpha = -1.0
    # beta = 1.0
    # delta = 0.3

    u = DuffingControl(omega)
    sde = Duffing(alpha_wrong, beta_wrong, gamma, delta_wrong, sigma_x, sigma_v, u)

    class GaussianLikelihood(dax.Likelihood):

        s: jax.Array

        def __init__(self, s):
            self.s = jnp.array(s)

        def _log_prob(self, y, x, u):
            return -0.5 * jnp.sum( ((y - x[0]) / self.s) ** 2) - jnp.log(self.s)
        
        def _sample(self, x, u, key):
            return x[0] + self.s * jr.normal(key, shape=x.shape)

    ssm = dax.StateSpaceModel(
        dax.DiagonalGaussian(
            jnp.array([0.0, 1.0]),
            jnp.array([2.0, 2.0])),
        dax.EulerMaruyama(sde, dt=dt),
        GaussianLikelihood(s_wrong)
    )

    filter = dax.BootstrapFilter(num_particles=1_000)
    smoother = dax.BootstrapSmoother(1)
    optimizer = optax.adam(1e-3)

    em = dax.ExpectationMaximization(filter, smoother, optimizer, max_m_step_iters=100)
    
    ssm, log_L = em.run(ssm, us_train, ys_train, 100, key)

    print(f'alpha: {ssm.transition.sde.alpha:.2f}')
    print(f'beta: {ssm.transition.sde.beta:.2f}')
    print(f'delta: {ssm.transition.sde.delta:.2f}')
    print(f'sigma: {ssm.likelihood.s:.2f}')

    fig, ax = plt.subplots()
    ax.plot(log_L)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Likelihood')
    sns.despine(trim=True)
    plt.savefig('duffing_em.png', dpi=300)

    # Do filtering and smoothing
    key, subkey = jr.split(key)
    pas, log_L = filter.filter(ssm, us_train, ys_train, subkey)
    key, subkey = jr.split(key)
    lower, median, upper = pas.get_credible_interval(subkey)

    key, subkey = jr.split(key)
    pas_test = ssm.predict(pas[-1], us_test, subkey)

    key, subkey = jr.split(key)
    lower_test, median_test, upper_test = pas_test.get_credible_interval(subkey)

    # Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(ts, xs[:, 0], label='True')
    ax[0].plot(ts_train, ys_train, 'k.')

    ax[0].plot(ts_train_w_init, median[:, 0], 'r-', label='Filter')
    ax[0].fill_between(ts_train_w_init, lower[:, 0], upper[:, 0], color='r', alpha=0.2)

    ax[0].plot(ts_test, median_test[:, 0], 'g-', label='Prediction')
    ax[0].fill_between(ts_test, lower_test[:, 0], upper_test[:, 0], color='g', alpha=0.2)
    ax[0].plot(ts_test, pas_test.particles[:, 0::1000, 0], '-.', color='green', lw=0.5)

    ax[0].set_ylim(-2, 2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Value')
    ax[0].legend(frameon=False, loc='best')

    for a in ax:
        a.axvline(ts[200], linestyle='--', color='black', alpha=0.5)
    
    ax[1].plot(ts, xs[:, 1], label='True')
    ax[1].plot(ts_train_w_init, median[:, 1], 'r-', label='Filter')
    ax[1].fill_between(ts_train_w_init, lower[:, 1], upper[:, 1], color='r', alpha=0.2)

    ax[1].plot(ts_test, median_test[:, 1], 'g-', label='Prediction')
    ax[1].fill_between(ts_test, lower_test[:, 1], upper_test[:, 1], color='g', alpha=0.2)
    ax[1].plot(ts_test, pas_test.particles[:, 0::1000, 1], '-.', color='green', lw=0.5)

    ax[1].set_ylim(-2, 2)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Value')
    ax[1].legend(frameon=False, loc='best')

    ax[0].set_title('Duffing Oscillator - True: $\\alpha=-1.0$, $\\beta=1.0$, $\\delta=0.3$, $\\sigma=0.1$ / Estimated: $\\alpha={:.2f}$, $\\beta={:.2f}$, $\\delta={:.2f}$, $\\sigma={:.2f}$'.format(ssm.transition.sde.alpha, ssm.transition.sde.beta, ssm.transition.sde.delta, ssm.likelihood.s))

    sns.despine(trim=True)

    plt.savefig('duffing_fitted.png', dpi=300)

    plt.show()
