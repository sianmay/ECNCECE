import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt

import networkx as nx
from sbx import PPO

from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from typing import (
  Any,
)
from collections.abc import Iterable, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from jax import eval_shape, lax
from jax.core import ShapedArray
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)

import gymnasium as gym
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy, Flatten

from utils import Graph2ffnn


def build_model(env, graph, lr=0.0009, gamma = 0.9, v=0, seed=None, sort_graph=False, mlp=False, batch_size=64, 
                n_steps=256, ent_coef=0.02498524250387804, clip_range=0.4, n_epochs=20, gae_lambda=0.8,
                max_grad_norm=0.6, vf_coef=0.5393659832430022, activation_fn=nn.tanh):

    if mlp:
        Policy = "MlpPolicy"
    else:
        Policy = CustomPPOPolicy
        ffnn = Graph2ffnn(graph)
        Policy.ffnn = ffnn
    
    policy_kwargs = dict(activation_fn=activation_fn)
    model = PPO(Policy, env, verbose=v, batch_size=batch_size, n_steps=n_steps, gamma=gamma, learning_rate=lr, 
            ent_coef=ent_coef, clip_range=clip_range, n_epochs=n_epochs, gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, 
            vf_coef=vf_coef, seed=seed, policy_kwargs=policy_kwargs)#, activation_fn=activation_fn)

    
    return model

def evaluate(model, n_episodes=100, deterministic=False, render=False, print_out=False):
    env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=deterministic, render=render)
    if print_out:
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward,  std_reward

def g2nn(G):
    #remove utgoing connections frm output nodes
    for n in range(5):
        l = list(G.out_edges(n))
        G.remove_edges_from(l)

    Gs = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute="old_label")
    return Gs


class ForWard(nn.Module):
    critic: bool = False
    ffnn: Any = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    def setup(self):

        ffnn = self.ffnn
        self.flatten = Flatten()
        self.in_features = ffnn.in_features
        self.sources = ffnn.sources
        self.layer_preds = ffnn.layer_preds
        self.L = ffnn.L
        self.out_features = ffnn.out_features
        self.sinks = ffnn.sinks
        self.num_nodes = ffnn.dag.number_of_nodes()
   
        weights = []
        biases = []
        for i, (layer, preds, w) in enumerate(zip(self.L[1:], ffnn.layer_preds, ffnn.weights)):
            W = w
            b = []
            for j, v in enumerate(layer):
                b.append(ffnn.dag.nodes[v]["bias"])
                for pred in ffnn.dag.predecessors(v):
                    weight = ffnn.dag[pred][v]["weight"]
                    W[preds.index(pred),j] = weight
            W = jnp.array(W)
            b = jnp.array(b)
            weights.append(W)
            biases.append(b)

        masks = []
        for i, (w, m) in enumerate(zip(weights, ffnn.masks)):
            M = jnp.array(m)#m.detach().numpy())
            self.variable('mask', f'M{i+1}', lambda : M)
            masks.append(M)

        self.masks = masks
        self.weights = weights
        self.biases = biases
        denseLayers = []

        for layer in ffnn.L[1:-1]:
            dense = DenseMasked(len(layer))
            denseLayers.append(dense)
        if self.critic:
            dense = nn.Dense(1)
            denseLayers.append(dense)
        else:
            dense = DenseMasked(len(ffnn.L[-1]))
            denseLayers.append(dense)

        self.denseLayers = denseLayers

    def __call__(self, x, deterministic=True):
        if x.shape[1] != self.in_features:
            raise RuntimeError(f'The model expects {self.in_features} input features, layers: {self.L}')
        
        activations = jnp.zeros((x.shape[0], self.num_nodes))
        activations = activations.at[:, self.sources].set(x)

        
        for i, (layer, preds, dense, M, W, b) in enumerate(zip(self.L[1:], self.layer_preds, self.denseLayers, self.masks, self.weights, self.biases)):
            if self.critic and i == len(self.L)-2:
                out = dense(activations[:, preds])
                return out
            out = dense(activations[:, preds], M, W, b)
            if i + 2 < len(self.L):
                out = self.activation_fn(out)

            activations = activations.at[:, layer].set(out)
        
        out = activations[:, self.sinks]
        return out

default_kernel_init = initializers.lecun_normal()

class DenseMasked(nn.Module):

    features: int
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    # Deprecated. Will be removed.
    dot_general: DotGeneralT | None = None
    dot_general_cls: Any = None

    @compact
    def __call__(self, inputs: Array, mask, W, b) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        
        '''
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )'''
        

        kernel = self.param('kernel', lambda _: W)

        if self.use_bias:
            bias = self.param('bias', lambda _: b)
            '''
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
            '''
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general
        y = dot_general(
            inputs,
            kernel*mask,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
            )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

tfd = tfp.distributions

class Critic(nn.Module):
    ffnn: Any
    n_units: int = 64
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    def setup(self):
        self.flatten = Flatten()
        self.fwd = ForWard(critic=True, ffnn=self.ffnn, activation_fn=self.activation_fn) #TODO uncomment
        self.dense = nn.Dense(1)

    #@nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.flatten(x)
        x = self.fwd(x)
        return x


class Actor(nn.Module):
    ffnn: Any
    action_dim: int
    n_units: int = 256
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    def setup(self):
        self.flatten = Flatten()
        self.fwd = ForWard(ffnn=self.ffnn, activation_fn=self.activation_fn)

    #@nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution: 
        x = self.flatten(x)
        action_logits = self.fwd(x)

        
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfp.distributions.Independent(
                tfp.distributions.Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return dist


class CustomPPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        ffnn: Any = None
    ):
        if optimizer_kwargs is None:
            # Small values to avoid NaN in Adam optimizer
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.n_units = net_arch[0]
            else:
                assert isinstance(net_arch, dict)
                self.n_units = net_arch["pi"][0]
        else:
            self.n_units = 5#64
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": int(np.prod(self.action_space.shape)),
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                f"Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.actor = Actor(
            ffnn=self.ffnn,
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.vf = Critic(ffnn=self.ffnn,n_units=self.n_units, activation_fn=self.activation_fn)

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    def predict_all(self, observation: np.ndarray, key: jax.Array) -> np.ndarray:
        return self._predict_all(self.actor_state, self.vf_state, observation, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, vf_state, obervations, key):
        dist = actor_state.apply_fn(actor_state.params, obervations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = vf_state.apply_fn(vf_state.params, obervations).flatten()
        return actions, log_probs, values
