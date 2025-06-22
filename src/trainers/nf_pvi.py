import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map


class ConditionalNormalizingFlow(eqx.Module):
    """
    Conditional Normalizing Flow that transforms base samples u to target samples z
    conditioned on particle parameters ρ.
    """
    layers: list
    base_dim: int
    target_dim: int
    
    def __init__(self, base_dim: int, target_dim: int, n_layers: int = 4, hidden_dim: int = 64, key: jax.random.PRNGKey = None):
        self.base_dim = base_dim
        self.target_dim = target_dim
        keys = jax.random.split(key, n_layers) if key is not None else None
        
        # Simple affine coupling layers for the flow
        self.layers = []
        for i in range(n_layers):
            layer_key = keys[i] if keys is not None else None
            self.layers.append(AffineCouplingLayer(target_dim, hidden_dim, layer_key))
    
    def forward(self, u: jax.Array, rho: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Forward transformation: u -> z
        Returns (z, log_det_jacobian)
        """
        z = u
        log_det_jac = 0.0
        
        for layer in self.layers:
            z, ldj = layer.forward(z, rho)
            log_det_jac += ldj
            
        return z, log_det_jac
    
    def inverse(self, z: jax.Array, rho: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Inverse transformation: z -> u
        Returns (u, log_det_jacobian)
        """
        u = z
        log_det_jac = 0.0
        
        for layer in reversed(self.layers):
            u, ldj = layer.inverse(u, rho)
            log_det_jac += ldj
            
        return u, log_det_jac


class AffineCouplingLayer(eqx.Module):
    """
    Affine coupling layer conditioned on particle parameters ρ.
    """
    net_s: eqx.nn.MLP
    net_t: eqx.nn.MLP
    split_dim: int
    
    def __init__(self, dim: int, hidden_dim: int, key: jax.random.PRNGKey = None):
        self.split_dim = dim // 2
        keys = jax.random.split(key, 2) if key is not None else [None, None]
        
        # Networks that take both x and rho as input
        input_dim = self.split_dim + dim  # split_dim for x, dim for rho
        
        self.net_s = eqx.nn.MLP(
            in_size=input_dim,
            out_size=dim - self.split_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[0]
        )
        
        self.net_t = eqx.nn.MLP(
            in_size=input_dim,
            out_size=dim - self.split_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[1]
        )
    
    def forward(self, x: jax.Array, rho: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x1, x2 = x[:self.split_dim], x[self.split_dim:]
        
        # Condition on both x1 and rho
        net_input = np.concatenate([x1, rho])
        s = self.net_s(net_input)
        t = self.net_t(net_input)
        
        y1 = x1
        y2 = x2 * np.exp(s) + t
        
        y = np.concatenate([y1, y2])
        log_det_jac = np.sum(s)
        
        return y, log_det_jac
    
    def inverse(self, y: jax.Array, rho: jax.Array) -> Tuple[jax.Array, jax.Array]:
        y1, y2 = y[:self.split_dim], y[self.split_dim:]
        
        # Condition on both y1 and rho
        net_input = np.concatenate([y1, rho])
        s = self.net_s(net_input)
        t = self.net_t(net_input)
        
        x1 = y1
        x2 = (y2 - t) * np.exp(-s)
        
        x = np.concatenate([x1, x2])
        log_det_jac = -np.sum(s)
        
        return x, log_det_jac


class NFPID(eqx.Module):
    """
    Normalizing Flow PID with conditional flows
    """
    particles: jax.Array  # ρ particles that parameterize flows
    flow: ConditionalNormalizingFlow
    base_dim: int
    target_dim: int
    
    def __init__(self, n_particles: int, particle_dim: int, base_dim: int, target_dim: int, 
                 n_flow_layers: int = 4, hidden_dim: int = 64, key: jax.random.PRNGKey = None):
        keys = jax.random.split(key, 2) if key is not None else [None, None]
        
        # Initialize particles ρ from standard Gaussian
        self.particles = jax.random.normal(keys[0], (n_particles, particle_dim))
        
        # Initialize conditional normalizing flow
        self.flow = ConditionalNormalizingFlow(base_dim, target_dim, n_flow_layers, hidden_dim, keys[1])
        
        self.base_dim = base_dim
        self.target_dim = target_dim
    
    def sample_base(self, key: jax.random.PRNGKey, n_samples: int) -> jax.Array:
        """Sample from base distribution q0(u) ~ N(0, I)"""
        return jax.random.normal(key, (n_samples, self.base_dim))
    
    def conditional_sample(self, key: jax.random.PRNGKey, rho: jax.Array, n_samples: int) -> jax.Array:
        """Sample from conditional distribution q(z|ρ) using flow"""
        u = self.sample_base(key, n_samples)
        z, _ = vmap(self.flow.forward, (0, None))(u, rho)
        return z
    
    def conditional_log_prob(self, z: jax.Array, rho: jax.Array) -> jax.Array:
        """Compute log q(z|ρ) using flow"""
        u, log_det_jac = self.flow.inverse(z, rho)
        # Base log prob: log q0(u) = -0.5 * ||u||^2 - const
        base_log_prob = -0.5 * np.sum(u**2)
        return base_log_prob + log_det_jac
    
    def log_prob(self, z: jax.Array, y: jax.Array = None) -> jax.Array:
        """Compute log q(z) as mixture over particles"""
        # For each particle, compute log q(z|ρ_i)
        log_probs = vmap(self.conditional_log_prob, (None, 0))(z, self.particles)
        # log q(z) = log(1/N * sum_i exp(log q(z|ρ_i)))
        return jax.scipy.special.logsumexp(log_probs) - np.log(len(self.particles))
    
    def sample(self, key: jax.random.PRNGKey, n_samples: int, y: jax.Array = None) -> jax.Array:
        """Sample from approximate posterior q(z)"""
        keys = jax.random.split(key, n_samples)
        
        def sample_one(key):
            # Randomly select a particle
            particle_key, sample_key = jax.random.split(key)
            particle_idx = jax.random.randint(particle_key, (), 0, len(self.particles))
            rho = self.particles[particle_idx]
            return self.conditional_sample(sample_key, rho, 1)[0]
        
        return vmap(sample_one)(keys)


def nf_particle_grad(key: jax.random.PRNGKey,
                     nfpid: NFPID,
                     target: Target,
                     particles: jax.Array,
                     y: jax.Array,
                     mc_n_samples: int):
    """
    Compute gradients for NF-PVI particle optimization
    """
    def elbo_per_particle(particle):
        """Compute ELBO for a single particle ρ"""
        # Sample u from base distribution
        u = nfpid.sample_base(key, mc_n_samples)
        
        # Transform through flow: z = f_ρ(u)
        z, log_det_jac = vmap(nfpid.flow.forward, (0, None))(u, particle)
        
        # Compute log probabilities
        base_log_prob = -0.5 * np.sum(u**2, axis=1)  # log q0(u)
        flow_log_prob = base_log_prob + log_det_jac   # log q(z|ρ)
        target_log_prob = vmap(target.log_prob, (0, None))(z, y)  # log p(x,z)
        
        # ELBO = E[log p(x,z) - log q(z|ρ)]
        elbo = np.mean(target_log_prob - flow_log_prob)
        return -elbo  # Negative because we minimize
    
    # Compute gradients for all particles
    grad = vmap(jax.grad(elbo_per_particle))(particles)
    return grad


def nf_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            hyperparams: PIDParameters):
    """
    NF-PVI loss function (negative ELBO)
    """
    nfpid = eqx.combine(params, static)
    
    # Sample from all particles and compute mixture log prob
    samples = nfpid.sample(key, hyperparams.mc_n_samples, y)
    
    # Compute log q(z) and log p(x,z)
    logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(samples, None)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    
    # Return negative ELBO
    return np.mean(logq - logp, axis=0)


def nf_particle_step(key: jax.random.PRNGKey,
                     nfpid: NFPID,
                     target: Target,
                     y: jax.Array,
                     optim: PIDOpt,
                     carry: PIDCarry,
                     hyperparams: PIDParameters):
    """
    Particle optimization step for NF-PVI
    """
    grad_fn = lambda particles: nf_particle_grad(
        key,
        nfpid,
        target,
        particles,
        y,
        hyperparams.mc_n_samples)
        
    g_grad, r_precon_state = optim.r_precon.update(
        nfpid.particles,
        grad_fn,
        carry.r_precon_state,)
        
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=nfpid.particles,
        index=y)
        
    nfpid = eqx.tree_at(lambda tree: tree.particles,
                        nfpid,
                        nfpid.particles + update)
    
    carry = PIDCarry(
        id=nfpid,
        theta_opt_state=carry.theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state)
        
    return nfpid, carry


def nf_step(key: jax.random.PRNGKey,
            carry: PIDCarry,
            target: Target,
            y: jax.Array,
            optim: PIDOpt,
            hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    """
    Complete NF-PVI optimization step
    """
    theta_key, r_key = jax.random.split(key, 2)
    
    def loss(key, params, static):
        return nf_loss(key,
                       params,
                       static,
                       target,
                       y,
                       hyperparams)
    
    # Optimize flow parameters θ
    lval, nfpid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )
    
    # Optimize particles ρ
    nfpid, carry = nf_particle_step(
        r_key,
        nfpid,
        target,
        y,
        optim,
        carry,
        hyperparams)
    
    carry = PIDCarry(
        id=nfpid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state)
        
    return lval, carry