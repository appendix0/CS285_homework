"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims :
            layers.append(nn.Linear(input_dim,hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim,action_dim*chunk_size))
        self.net = nn.Sequential(*layers)    
    

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.sample_actions(state)
        return ((prediction-action_chunk)**2).mean()

        
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        prediction = self.net(state)
        return prediction.reshape(state.shape[0], self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim + action_dim*chunk_size +1

        for hidden_dim in hidden_dims : 
            layers.append(nn.Linear(input_dim , hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim,action_dim*chunk_size))

        self.net = nn.Sequential(*layers)

         
            
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        batch_size = action_chunk.shape[0]

        tau = torch.rand(
            batch_size, 1, 1,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )

        x_tau = (1 - tau) * noise + tau * action_chunk
        target_v = action_chunk - noise

        network_input = torch.cat(
            [
                state,
                tau.reshape(batch_size, 1),
                x_tau.reshape(batch_size, -1),
            ],
            dim=1,
        )

        predicted_v = self.net(network_input)
        predicted_v = predicted_v.reshape(
            batch_size, self.chunk_size, self.action_dim
        )

        return ((target_v - predicted_v) ** 2).mean()


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        dt = 1.0/num_steps

        action_chunk = torch.randn(batch_size , self.chunk_size, self.action_dim,
                                   device = state.device,
                                   dtype = state.dtype
                                   )

        for step in range(num_steps) : 
            current_time = step * dt
            tau = torch.full((batch_size,1),current_time,
                             device=state.device,
                             dtype=state.dtype
                             )

            network_input = torch.cat([
                state, tau, action_chunk.reshape(batch_size,-1)
            ], dim = 1
            )

            predicted_v = self.net(network_input)
            predicted_v = predicted_v.reshape(batch_size, self.chunk_size, self.action_dim)

            action_chunk = action_chunk + dt * predicted_v

        return action_chunk 







PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
