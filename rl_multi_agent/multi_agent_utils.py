import warnings
from typing import Dict

import torch
from torch import nn, optim as optim

from rl_multi_agent import MultiAgent
from utils.misc_util import ensure_shared_grads


def compute_losses_no_backprop(agent: MultiAgent):
    full_loss = None
    last_losses = {}
    for k, loss in agent.loss().items():
        loss = loss.squeeze()
        last_losses["loss/" + k] = loss.item()
        if full_loss is None:
            full_loss = loss
        elif (full_loss.is_cuda == loss.is_cuda) and (
            not full_loss.is_cuda or full_loss.get_device() == loss.get_device()
        ):
            full_loss += loss
        else:
            warnings.warn("Loss {} is on a different device!".format(k))
    assert full_loss is not None
    return last_losses


class TrainingCompleteException(Exception):
    pass


class EndProcessException(Exception):
    pass


def compute_losses_and_backprop(
    agent: MultiAgent,
    shared_model: nn.Module,
    optimizer: optim.Optimizer,
    update_lock,
    gpu: bool,
    retain_graph: bool = False,
    skip_backprop: bool = False,
) -> Dict[str, float]:
    agent.model.zero_grad()
    full_loss = None
    last_losses = {}
    for k, loss in agent.loss().items():
        loss = loss.squeeze()
        last_losses["loss/" + k] = loss.item()
        if full_loss is None:
            full_loss = loss
        elif (full_loss.is_cuda == loss.is_cuda) and (
            not full_loss.is_cuda or full_loss.get_device() == loss.get_device()
        ):
            full_loss += loss
        else:
            warnings.warn("Loss {} is on a different device!".format(k))
    if full_loss is not None:
        if not skip_backprop:
            full_loss.backward(retain_graph=retain_graph)
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 3, "inf")

            if update_lock is not None:
                update_lock.acquire()
            ensure_shared_grads(agent.model, shared_model, gpu=gpu)
            optimizer.step()
            if update_lock is not None:
                update_lock.release()
    else:
        warnings.warn(
            (
                "No loss avaliable for agent.\n"
                "Episode length: {}\n"
                "Rewards: {}\n"
                "Actions: {}\n"
                "Expert actions: {}\n"
            ).format(
                agent.episode.num_steps_taken_in_episode(),
                agent.rewards_per_agent,
                agent.actions,
                agent.expert_actions,
            )
        )

    return last_losses
