import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.conditional import FixedDiagNormCondWSkip, Conditional
from src.trainers.util import loss_step
from typing import Tuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree


class SimpleNFConditional(Conditional):
    """
    Simple normalizing flow conditional that extends the existing conditional interface.
    """
    base_conditional: eqx.Module
    flow_net: eqx.nn.MLP
    
    def __init__(self, key: jax.random.PRNGKey, d_x: int, d_z: int, d_y: int, 
                 n_hidden: int, flow_hidden: int = 64):
        key1, key2 = jax.random.split(key, 2)
        
        # Set the dimensions
        self.d_x = d_x
        self.d_z = d_z
        self.d_y = d_y
        
        # Base conditional (reuse existing implementation)
        self.base_conditional = FixedDiagNormCondWSkip(key1, d_x, d_z, d_y, n_hidden)
        
        # Simple flow network that modifies the base output
        self.flow_net = eqx.nn.MLP(
            in_size=d_z + d_x,  # Takes both z and base output
            out_size=d_x,       # Outputs correction
            width_size=flow_hidden,
            depth=2,
            activation=jax.nn.tanh,
            key=key2
        )
    
    def log_prob(self, x: jax.Array, z: jax.Array, y: jax.Array):
        """Use base conditional log prob with flow correction."""
        base_logp = self.base_conditional.log_prob(x, z, y)
        
        # Simple correction based on flow
        flow_input = np.concatenate([z, x])
        flow_correction = self.flow_net(flow_input)
        correction_term = -0.5 * np.sum(flow_correction**2)  # Regularization term
        
        return base_logp + 0.1 * correction_term  # Small correction
    
    def f(self, z: jax.Array, y: jax.Array, eps: jax.Array):
        """Generate samples with flow enhancement."""
        # Get base sample
        base_sample = self.base_conditional.f(z, y, eps)
        
        # Flatten all arrays to 1D for concatenation, then reshape flow output appropriately
        z_flat = z.flatten() if z.ndim > 0 else np.array([z])
        base_flat = base_sample.flatten() if base_sample.ndim > 0 else np.array([base_sample])
        
        # Concatenate flattened versions
        flow_input = np.concatenate([z_flat, base_flat])
        
        # Get flow correction
        flow_correction = self.flow_net(flow_input)
        
        # Reshape flow correction to match base_sample shape
        if base_sample.ndim > 0:
            flow_correction = flow_correction.reshape(base_sample.shape)
        
        # Apply small correction
        enhanced_sample = base_sample + 0.1 * flow_correction
        return enhanced_sample
    
    def base_sample(self, key: jax.random.PRNGKey, n_samples: int):
        """Delegate to base conditional."""
        return self.base_conditional.base_sample(key, n_samples)
    
    def sample(self, key: jax.random.PRNGKey, n_samples: int, z: jax.Array, y: jax.Array):
        """Sample from the enhanced conditional distribution."""
        # For n_samples > 1, we need to handle vectorization properly
        if n_samples == 1:
            eps = self.base_sample(key, 1)
            return self.f(z, y, eps).reshape(1, -1)  # Ensure (1, d_x) shape
        else:
            # For multiple samples, vectorize over the f function
            keys = jax.random.split(key, n_samples)
            
            def sample_one(sample_key):
                eps = self.base_sample(sample_key, 1)
                return self.f(z, y, eps[0])  # f expects single eps, not batch
            
            # Apply to all samples
            enhanced_samples = jax.vmap(sample_one)(keys)
            return enhanced_samples
    
    def get_filter_spec(self):
        """Return filter specification for optimization."""
        import jax.tree_util as jtu
        
        filter_spec = jtu.tree_map(lambda _: False, self)
        
        # Mark base conditional as trainable
        filter_spec = eqx.tree_at(lambda tree: tree.base_conditional,
                                  filter_spec,
                                  self.base_conditional.get_filter_spec())
        
        # Mark flow network as trainable
        filter_spec = eqx.tree_at(lambda tree: tree.flow_net,
                                  filter_spec,
                                  jtu.tree_map(eqx.is_array, self.flow_net))
        
        return filter_spec


def create_nf_pid(key: jax.random.PRNGKey, d_x: int, d_z: int, d_y: int,
                  n_particles: int, n_hidden: int, flow_hidden: int = 64):
    """
    Create a PID with normalizing flow conditional.
    """
    conditional_key, particle_key = jax.random.split(key, 2)
    
    # Create flow-enhanced conditional
    conditional = SimpleNFConditional(
        conditional_key, d_x, d_z, d_y, n_hidden, flow_hidden
    )
    
    # Create PID with flow conditional
    init_particles = jax.random.normal(particle_key, (n_particles, d_z))
    
    return PID(particle_key, conditional, n_particles, init=init_particles)


# Use the same step functions as regular PVI
def nf_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            hyperparams: PIDParameters):
    """
    NF-PVI loss function (same as regular PVI for now).
    """
    pid = eqx.combine(params, static)
    _samples = pid.sample(key, hyperparams.mc_n_samples, None)
    logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
    logp = vmap(target.log_prob, (0, None))(_samples, y)
    return np.mean(logq - logp, axis=0)


def nf_particle_grad(key: jax.random.PRNGKey,
                     pid: PID,
                     target: Target,
                     particles: jax.Array,
                     y: jax.Array,
                     mc_n_samples: int):
    """
    Compute particle gradients for NF-PVI (same as regular PVI for now).
    """
    def ediff_score(particle, eps):
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        assert samples.shape == (mc_n_samples, target.dim)
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        assert logp.shape == (mc_n_samples,)
        assert logq.shape == (mc_n_samples,)
        logp = np.mean(logp, 0)
        logq = np.mean(logq, 0)
        return logq - logp
    
    eps = pid.conditional.base_sample(key, mc_n_samples)
    grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
    return grad


def nf_particle_step(key: jax.random.PRNGKey,
                     pid: PID,
                     target: Target,
                     y: jax.Array,
                     optim: PIDOpt,
                     carry: PIDCarry,
                     hyperparams: PIDParameters):
    """
    Particle optimization step for NF-PVI.
    """
    grad_fn = lambda particles: nf_particle_grad(
        key, pid, target, particles, y, hyperparams.mc_n_samples)
        
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        grad_fn,
        carry.r_precon_state,)
        
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y)
        
    pid = eqx.tree_at(lambda tree: tree.particles,
                      pid,
                      pid.particles + update)
    
    carry = PIDCarry(
        id=pid,
        theta_opt_state=carry.theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state)
        
    return pid, carry


def nf_step(key: jax.random.PRNGKey,
            carry: PIDCarry,
            target: Target,
            y: jax.Array,
            optim: PIDOpt,
            hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    """
    NF-PVI optimization step.
    """
    theta_key, r_key = jax.random.split(key, 2)
    
    def loss(key, params, static):
        return nf_loss(key, params, static, target, y, hyperparams)
    
    # Optimize conditional parameters (including flow)
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )
    
    # Optimize particles
    pid, carry = nf_particle_step(
        r_key,
        pid,
        target,
        y,
        optim,
        carry,
        hyperparams)
    
    carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state)
        
    return lval, carry