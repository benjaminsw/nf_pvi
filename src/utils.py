import jax
import optax
from src.base import *
from src.base import RPreconParameters
from src.nn import Net
from src.id import PID, SID
from src.conditional import KERNELS
from src.preconditioner import (clip_grad_norm,
                                identity,
                                rms)
from src.ropt import (regularized_wasserstein_descent,
                      stochastic_gradient_to_update,
                      scale_by_schedule,
                      kl_descent,
                      lr_to_schedule)
from src.trainers.pvi import de_step as pvi_de_step
from src.trainers.svi import de_step as svi_de_step
from src.trainers.uvi import de_step as uvi_de_step
from src.trainers.sm import de_step as sm_de_step

# Import NF-PVI implementation
try:
    from src.trainers.simple_nf_pvi import nf_step as nf_pvi_de_step, create_nf_pid
    NF_PVI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NF-PVI implementation: {e}")
    NF_PVI_AVAILABLE = False

# Import fixed WGF-GMM implementation
try:
    from src.trainers.wgf_gmm import wgf_gmm_pvi_step
    WGF_GMM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import WGF-GMM implementation: {e}")
    WGF_GMM_AVAILABLE = False

import equinox as eqx
import yaml
import re


def wgf_gmm_de_step(key, carry, target, y, optim, hyperparams):
    """
    Wrapper for WGF-GMM PVI step to match the standard de_step interface.
    No fallback - will raise errors if WGF-GMM fails.
    """
    if not WGF_GMM_AVAILABLE:
        raise ImportError("WGF-GMM implementation is not available")
    
    # Handle the gmm_state attribute that WGF-GMM expects
    if not hasattr(carry, 'gmm_state'):
        # Create a temporary extended carry with gmm_state
        class ExtendedCarry:
            def __init__(self, original_carry):
                self.id = original_carry.id
                self.theta_opt_state = original_carry.theta_opt_state
                self.r_opt_state = original_carry.r_opt_state
                self.r_precon_state = original_carry.r_precon_state
                self.gmm_state = None  # Initialize as None
        
        extended_carry = ExtendedCarry(carry)
    else:
        extended_carry = carry
    
    # Call the WGF-GMM implementation - no try/catch, let errors propagate
    lval, updated_extended_carry = wgf_gmm_pvi_step(
        key=key,
        carry=extended_carry,
        target=target,
        y=y,
        optim=optim,
        hyperparams=hyperparams,
        lambda_reg=0.1,    # Wasserstein regularization strength
        lr_mean=0.01,      # Learning rate for means
        lr_cov=0.001,      # Learning rate for covariances
        lr_weight=0.01     # Learning rate for weights
    )
    
    # Convert back to standard PIDCarry format
    updated_carry = type(carry)(
        id=updated_extended_carry.id,
        theta_opt_state=updated_extended_carry.theta_opt_state,
        r_opt_state=updated_extended_carry.r_opt_state,
        r_precon_state=updated_extended_carry.r_precon_state
    )
    
    return lval, updated_carry


def gmm_pvi_de_step(key, carry, target, y, optim, hyperparams):
    """
    Wrapper for GMM-PVI step (simplified version).
    No fallback - will raise errors if GMM-PVI fails.
    """
    if not WGF_GMM_AVAILABLE:
        raise ImportError("GMM-PVI implementation is not available")
    
    # For now, just call the WGF-GMM implementation
    return wgf_gmm_de_step(key, carry, target, y, optim, hyperparams)


# Update DE_STEPS to include NF-PVI, WGF-GMM and GMM-PVI
DE_STEPS = {
    'pvi': pvi_de_step,
    'nf_pvi': nf_pvi_de_step if NF_PVI_AVAILABLE else pvi_de_step,  # Fallback to regular PVI if not available
    'wgf_gmm': wgf_gmm_de_step,
    'svi': svi_de_step,
    'uvi': uvi_de_step,
    'sm': sm_de_step
}


def make_nf_model(key: jax.random.PRNGKey,
                  model_parameters: ModelParameters,
                  d_x: int):
    """
    Make an NF-PVI model based on the model hyperparameters.
    """
    try:
        from src.trainers.simple_nf_pvi import create_nf_pid
        
        return create_nf_pid(
            key=key,
            d_x=d_x,
            d_z=model_parameters.d_z,
            d_y=model_parameters.d_y,
            n_particles=model_parameters.n_particles,
            n_hidden=model_parameters.n_hidden,
            flow_hidden=getattr(model_parameters, 'flow_hidden_dim', 64)
        )
    except ImportError:
        # Fallback to regular PID if NF-PVI is not available
        print("Warning: NF-PVI not available, falling back to regular PVI")
        return make_model(key, model_parameters, d_x)


def make_model(key: jax.random.PRNGKey,
               model_parameters: ModelParameters,
               d_x: int):
    """
    Make a model based on the model hyperparameters.
    """
    key1, key2 = jax.random.split(key, 2)
    assert model_parameters.kernel in KERNELS
    likelihood = KERNELS[model_parameters.kernel]

    conditional = likelihood(
        key1,
        d_x,
        model_parameters.d_z,
        model_parameters.d_y,
        n_hidden=model_parameters.n_hidden)
    n_particles = model_parameters.n_particles

    if model_parameters.use_particles:
        init = jax.random.normal(key2, (model_parameters.n_particles,
                                        model_parameters.d_z))
        model = PID(
            key2,
            conditional,
            n_particles,
            init=init 
        )
    else:
        model = SID(conditional)
    return model


def make_theta_opt(topt_param: ThetaOptParameters):
    """
    Make an optimizer for kernel parameters based on the optimizer hyperparameters.
    """
    theta_transform = []

    if topt_param.clip:
        clip = optax.clip_by_global_norm(topt_param.max_clip)
        theta_transform.append(clip)

    if topt_param.lr_decay:
        lr = optax.linear_schedule(topt_param.lr,
                                   topt_param.min_lr,
                                   topt_param.interval)
    else:
        lr = topt_param.lr

    if topt_param.optimizer == 'adam':
        optimizer = optax.adam(lr, b1=0.9, b2=0.99)
    elif topt_param.optimizer == 'rmsprop':
        optimizer = optax.rmsprop(lr)
    else:
        optimizer = optax.sgd(lr)

    theta_transform.append(optimizer)
    return optax.chain(*theta_transform)


def make_r_opt(key: jax.random.PRNGKey,
               ropt_param: ROptParameters,
               sgld: bool=False):
    """
    Make an optimizer for distribution space based on the optimizer hyperparameters.
    """
    transform = []

    if ropt_param.lr_decay:
        lr = optax.linear_schedule(ropt_param.lr,
                                   ropt_param.min_lr,
                                   ropt_param.interval)
    else:
        lr = ropt_param.lr
    if sgld:
        transform.append(
            kl_descent(key)
        )
    else:
        transform.append(
            regularized_wasserstein_descent(key,
                                            ropt_param.regularization))
    transform.append(scale_by_schedule(lr_to_schedule(lr)))
    transform.append(stochastic_gradient_to_update())
    return optax.chain(*transform)


def make_r_precon(r_precon_param: RPreconParameters):
    """
    Make a preconditioner for distribution space based on the preconditioner hyperparameters.
    """
    if r_precon_param:
        if r_precon_param.type == 'clip':
            return clip_grad_norm(r_precon_param.max_norm,
                                r_precon_param.agg)
        elif r_precon_param.type == 'rms':
            return rms(r_precon_param.agg, False)
    return identity()


def make_step_and_carry(
        key: jax.random.PRNGKey,
        parameters: Parameters,
        target: Target):
    """
    Make a step function and carry for a given algorithm.
    """
    model_key, key = jax.random.split(key, 2)
    
    # Use special NF model for NF-PVI algorithm
    if parameters.algorithm == 'nf_pvi':
        id = make_nf_model(model_key,
                          parameters.model_parameters,
                          target.dim)
    else:
        id = make_model(model_key,
                       parameters.model_parameters,
                       target.dim)

    if parameters.theta_opt_parameters is not None:
        theta_optim = make_theta_opt(parameters.theta_opt_parameters)

    if target.de:
        step = DE_STEPS[parameters.algorithm]
    else:
        raise NotImplementedError("Only DE is supported")
    
    id_state = eqx.filter(id, id.get_filter_spec())
    
    # Handle PVI-based algorithms (pvi, nf_pvi, wgf_gmm) the same way
    if parameters.algorithm in ['pvi', 'nf_pvi', 'wgf_gmm']:
        ropt_key, key = jax.random.split(key, 2)
        r_optim = make_r_opt(ropt_key,
                            parameters.r_opt_parameters)
        r_precon = make_r_precon(parameters.r_precon_parameters)
        optim = PIDOpt(theta_optim, r_optim, r_precon)
        carry = PIDCarry(id,
                        theta_optim.init(id_state),
                        r_optim.init(id_state),
                        r_precon.init(id),
                        gmm_state=None)  # Add gmm_state parameter
    elif parameters.algorithm == 'uvi':
        optim = SVIOpt(theta_optim)
        carry = SVICarry(id, theta_optim.init(id_state))
    elif parameters.algorithm == 'svi':
        optim = SVIOpt(theta_optim)
        carry = SVICarry(id, theta_optim.init(id_state))
    elif parameters.algorithm == 'sm':
        dual_optim = make_theta_opt(parameters.dual_opt_parameters)
        dual = Net(key,
                   target.dim,
                   target.dim,
                   parameters.dual_parameters.n_hidden,
                   act=jax.nn.relu)
        dual_state = eqx.filter(dual, dual.get_filter_spec())
        carry = SMCarry(id,
                        theta_optim.init(id_state),
                        dual,
                        dual_optim.init(dual_state))
        optim = SMOpt(theta_optim, dual_optim)
    else:
        raise ValueError(f"Unknown algorithm {parameters.algorithm}")

    def partial_step(key, carry, target, y):
        return step(
            key,
            carry,
            target,
            y,
            optim,
            parameters.extra_alg_parameters,
        )
    return partial_step, carry


def config_to_parameters(config: dict, algorithm: str):
    """
    Make a parameters from a config dictionary and an algorithm name.
    """
    parameters = {'algorithm': algorithm,}
    
    # Get model parameters and add NF-specific ones if needed
    model_params = dict(config[algorithm]['model'])
    if algorithm == 'nf_pvi':
        # Add NF-specific parameters with defaults
        model_params['particle_dim'] = model_params.get('particle_dim', 2)
        model_params['n_flow_layers'] = model_params.get('n_flow_layers', 4) 
        model_params['flow_hidden_dim'] = model_params.get('flow_hidden_dim', 64)
    
    parameters['model_parameters'] = ModelParameters(**model_params)
    
    if 'theta_opt' in config[algorithm]:
        parameters['theta_opt_parameters'] = ThetaOptParameters(
            **config[algorithm]['theta_opt']
        )
    
    # Handle PVI-based algorithms (pvi, nf_pvi, wgf_gmm) the same way
    if algorithm in ['pvi', 'nf_pvi', 'wgf_gmm']:
        parameters['r_opt_parameters'] = ROptParameters(
            **config[algorithm]['r_opt']
        )
        if 'r_precon' in config[algorithm]:
            parameters['r_precon_parameters'] = RPreconParameters(
                **config[algorithm]['r_precon']
            )
        parameters['extra_alg_parameters'] = PIDParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'svi':
        parameters['extra_alg_parameters'] = SVIParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'uvi':
        parameters['extra_alg_parameters'] = UVIParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'sm':
        parameters['dual_opt_parameters'] = ThetaOptParameters(
            **config[algorithm]['dual_opt']
        )
        parameters['dual_parameters'] = DualParameters(
            **config[algorithm]['dual']
        )
        parameters['extra_alg_parameters'] = SMParameters(
            **config[algorithm]['extra_alg']
        )
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported")
    return Parameters(**parameters)


def parse_config(config_path: str):
    """
    Parse a config file into a dictionary.
    """
    def none_to_dict(loader, node):
        # Construct the mapping using the default method
        mapping = loader.construct_mapping(node)
        # Replace all None values with {}
        return {k: ({} if v is None else v) for k, v in mapping.items()}
    
    # Load Empty Entries as an Empty Dictionary
    yaml.SafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        none_to_dict)
    
    # Simple scientific notation support
    yaml.SafeLoader.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'),
        list('-+0123456789.'))
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config