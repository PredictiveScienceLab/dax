
import jax
import jax.numpy as jnp 
import jax.random as jr
import equinox as eqx


__all__ = ['ParticleApproximation', 'TrajectoryParticleApproximation', 'TrajectorySamples']


class ParticleApproximation(eqx.Module):
    particles: jax.Array
    log_weights: jax.Array
    resampled: bool

    def __init__(self, particles, log_weights=None, resampled=False):
        self.particles = particles
        if log_weights is None:
            log_weights = -jnp.log(particles.shape[0]) * jnp.ones(particles.shape[0])
            resampled = True
        self.log_weights = log_weights
        self.resampled = resampled
    
    @property
    def weights(self):
        return jnp.exp(self.log_weights)
    
    @property
    def num_particles(self):
        return self.particles.shape[0]
    
    @eqx.filter_jit
    def normalize(self):
        new_log_weights = self.log_weights - jax.scipy.special.logsumexp(self.log_weights)
        return ParticleApproximation(self.particles, new_log_weights)
    
    @eqx.filter_jit
    def resample(self, key):
        key, subkey = jr.split(key)
        indices = jr.choice(subkey, self.num_particles, shape=(self.num_particles,), p=self.weights)
        particles = self.particles[indices]
        log_weights = -jnp.log(self.num_particles) * jnp.ones(self.num_particles)
        return ParticleApproximation(particles, log_weights, resampled=True)
    
    @eqx.filter_jit
    def sample(self, key):
        return jr.choice(key, self.particles, p=self.weights)
    

class TrajectoryParticleApproximation(eqx.Module):
    particles: jax.Array
    log_weights: jax.Array
    resampled: bool

    @property
    def weights(self):
        return jnp.exp(self.log_weights)

    def __init__(self, particles, log_weights=None, resampled=False):
        self.particles = particles
        if log_weights is None:
            log_weights = -jnp.log(particles.shape[1]) * jnp.ones(particles.shape[:2])
            resampled = True
        self.log_weights = log_weights
        self.resampled = resampled

    @staticmethod
    @eqx.filter_jit
    def make_from_init_and_rest(pa0, pas, resampled=False):
        particles = jnp.concatenate([pa0.particles[None, :, :], pas.particles], axis=0)
        log_weights = jnp.concatenate([pa0.log_weights[None, :], pas.log_weights], axis=0)
        return TrajectoryParticleApproximation(particles, log_weights, resampled=resampled)
    
    @staticmethod
    @eqx.filter_jit
    def make_from_particle_approximation(pas, resampled=False):
        return TrajectoryParticleApproximation(pas.particles, pas.log_weights, resampled=resampled)

    @eqx.filter_jit
    def __getitem__(self, index):
        if isinstance(index, slice):
            particles = self.particles[index.start:index.stop:index.step]
            log_weights = self.log_weights[index.start:index.stop:index.step]
            return TrajectoryParticleApproximation(particles, log_weights)
        else:
            return ParticleApproximation(self.particles[index], self.log_weights[index])
    
    @property
    @eqx.filter_jit
    def mean(self):
        return jnp.sum(self.particles * self.weights[:, :, None], axis=1)
    
    @property
    @eqx.filter_jit
    def var(self):
        return jnp.sum(self.particles ** 2 * self.weights[:, :, None], axis=1) - self.mean ** 2
    
    @property
    @eqx.filter_jit
    def std(self):
        return jnp.sqrt(self.var)
    
    @property
    def num_steps(self):
        return self.particles.shape[0]
    
    @eqx.filter_jit
    def resample(self, key):
        def f(x, log_w, key):
            return ParticleApproximation(x, log_w).resample(key)
        return TrajectoryParticleApproximation.make_from_particle_approximation(
            jax.vmap(f, in_axes=(0, 0, 0))(self.particles, self.log_weights, jr.split(key, self.num_steps)),
            resampled=True
        )
    
    @eqx.filter_jit
    def percentile(self, a, key):
        if self.resampled:
            particles = self.particles
        else:
            particles = self.resample(key).particles
        return jnp.percentile(particles, a, axis=1)
    
    @eqx.filter_jit
    def get_credible_interval(self, key):
        return self.percentile(jnp.array([2.5, 50., 97.5]), key)
    

class TrajectorySamples(eqx.Module):
    samples: jax.Array

    def __init__(self, samples):
        self.samples = samples
    
    @property
    def num_steps(self):
        return self.samples.shape[1]
    
    @property
    def num_samples(self):
        return self.samples.shape[0]
    
    @eqx.filter_jit
    def __getitem__(self, index):
        if isinstance(index, slice):
            return TrajectorySamples(self.samples[index])
        else:
            return self.samples[index]
    
    @eqx.filter_jit
    def mean(self):
        return jnp.mean(self.samples, axis=0)
    
    @eqx.filter_jit
    def var(self):
        return jnp.var(self.samples, axis=0)
    
    @eqx.filter_jit
    def std(self):
        return jnp.std(self.samples, axis=0)
    
    @eqx.filter_jit
    def percentile(self, a):
        return jnp.percentile(self.samples, a, axis=0)
    
    @eqx.filter_jit
    def get_credible_interval(self):
        return self.percentile(jnp.array([2.5, 50., 97.5]))
    
    @eqx.filter_jit
    def expect_vectorized(self, f_vectorized, axis=0):
        return jnp.mean(f_vectorized(self.samples), axis=axis)

    @eqx.filter_jit
    def expect(self, f, axis=0):
        return self.expect_vectorized(jax.vmap(f), axis=axis)
    