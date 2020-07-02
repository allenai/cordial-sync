from __future__ import division

import copy
from abc import ABC
from typing import Dict, List, Union, Optional, Any, TypeVar, Generic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import net_util
from utils.misc_util import huber_loss
from .episode import Episode

EnvType = TypeVar("EnvType")


class RLAgent(Generic[EnvType]):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        episode: Optional[Episode[EnvType]] = None,
        gpu_id: int = -1,
        **kwargs,
    ) -> None:
        # Important that gpu_id is set first as the model setter requires it
        self.gpu_id = gpu_id
        self._model: Optional[nn.Module] = None
        if model is not None:
            self.model = model
        self._episode = episode

        self.hidden: Optional[Any] = None
        self.global_episode_count = 0

    def sync_with_shared(self, shared_model: nn.Module) -> None:
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, state, hidden, **kwargs):
        raise NotImplementedError()

    def eval_at_current_state(self):
        return self.eval_at_state(state=self.state, hidden=self.hidden)

    @property
    def episode(self) -> Optional[Episode[EnvType]]:
        return self._episode

    @episode.setter
    def episode(self, value: Episode[EnvType]) -> None:
        self.reset_hidden()
        self.clear_history()
        self._episode = value

    @property
    def environment(self) -> Optional[EnvType]:
        if self.episode is None:
            return None
        return self.episode.environment

    @property
    def state(self):
        raise NotImplementedError()

    @property
    def model(self) -> nn.Module:
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, model_to_set: nn.Module):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self._model = model_to_set.cuda()
        else:
            self._model = model_to_set

    def gpuify(self, tensor: torch.Tensor) -> Union[torch.Tensor]:
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                tensor = tensor.cuda()
        return tensor

    def act(self, **kwargs) -> Dict:
        raise NotImplementedError()

    def reset_hidden(self) -> None:
        self.hidden = None

    def repackage_hidden(self) -> None:
        self.hidden = net_util.recursively_detach(self.hidden)

    def preprocess_action(
        self, last_action: Optional[int]
    ) -> Optional[Union[torch.LongTensor, torch.cuda.LongTensor]]:
        if last_action is not None:
            # noinspection PyTypeChecker
            return self.gpuify(torch.LongTensor([[last_action]]))
        return None

    def preprocess_frame(
        self, frame: np.ndarray, new_frame_size: int
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        frame = net_util.resnet_input_transform(frame, new_frame_size)
        # noinspection PyArgumentList
        return self.gpuify(torch.FloatTensor(frame))

    def preprocess_long_tensor_frame(
        self, frame: np.ndarray
    ) -> Union[torch.LongTensor, torch.cuda.LongTensor]:
        agent_local_tensor = torch.LongTensor(frame)
        if len(agent_local_tensor.shape) == 2:
            agent_local_tensor = agent_local_tensor.unsqueeze(0)
        return self.gpuify(agent_local_tensor)

    def preprocess_depth_frame(
        self, depth_frame: np.ndarray, new_frame_size: int
    ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        depth_frame = net_util.resize_image(
            torch.FloatTensor(np.expand_dims(depth_frame / 5000.0, axis=0)),
            new_frame_size,
        )
        return self.gpuify(depth_frame)

    def clear_history(self) -> None:
        raise NotImplementedError()

    def clear_graph_data(self):
        raise NotImplementedError()

    def loss(self, **kwargs) -> Dict[str, torch.FloatTensor]:
        raise NotImplementedError()


class A3CAgent(RLAgent[EnvType], ABC):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        episode: Optional[Episode[EnvType]] = None,
        gamma: float = 0.99,
        tau: float = 1.0,
        beta: float = 1e-2,
        gpu_id: int = -1,
        include_test_eval_results: bool = False,
        huber_delta: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, episode=episode, gpu_id=gpu_id, **kwargs)
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.include_test_eval_results = include_test_eval_results
        self.huber_delta = huber_delta

        self.last_reward: Optional[float] = None
        self.values: List[torch.FloatTensor] = []
        self.log_prob_of_actions: List[torch.FloatTensor] = []
        self.rewards: List[float] = []
        self.entropies: List[torch.FloatTensor] = []
        self.actions: List[int] = []
        self.step_results: List[Dict[str, Any]] = []
        self.eval_results: List[Dict[str, Any]] = []

        self._a3c_loss_disabled_for_episode: bool = False

    def eval_at_state(self, state, hidden, **kwargs):
        raise NotImplementedError()

    def disable_a3c_loss_for_episode(self, mode: bool = True):
        self._a3c_loss_disabled_for_episode = mode

    @property
    def episode(self):
        return super(A3CAgent, self).episode

    @episode.setter
    def episode(self, value: Episode[EnvType]) -> None:
        super(A3CAgent, self).episode = value
        self.disable_a3c_loss_for_episode(False)

    def act(
        self,
        train: bool = True,
        action: Optional[int] = None,
        sample_action: bool = True,
        **kwargs,
    ) -> Dict:
        if not self.model.training and train:
            self.model.train()
        if self.model.training and not train:
            self.model.eval()

        assert self.episode is not None
        assert not self.episode.is_complete()

        if not train and self.hidden is not None:
            self.repackage_hidden()

        eval = self.eval_at_current_state()
        value = eval["value"]
        self.hidden = None if "hidden" not in eval else eval["hidden"]

        logit = eval["logit"]
        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)
        if hasattr(self.model, "action_groups"):
            entropy_list = []
            k = 0
            marginal_probs = []
            for action_group in self.model.action_groups:
                marginal_probs.append(
                    probs[:, k : (k + len(action_group))].sum(1).unsqueeze(1)
                )
                group_probs = F.softmax(logit[:, k : (k + len(action_group))], dim=1)
                group_log_probs = F.log_softmax(
                    logit[:, k : (k + len(action_group))], dim=1
                )
                entropy_list.append(-(group_log_probs * group_probs).sum(1))
                k += len(action_group)
            entropy = sum(entropy_list) / len(entropy_list)
            marginal_probs = torch.cat(tuple(marginal_probs), dim=1)
            entropy += -(marginal_probs * torch.log(marginal_probs)).sum(1)
        else:
            entropy = -(log_probs * probs).sum().unsqueeze(0)

        if action is None:
            if sample_action:
                action = probs.multinomial(num_samples=1).item()
            else:
                action = probs.argmax(dim=-1, keepdim=True).item()

        assert log_probs.shape[0] == 1
        log_prob_of_action = log_probs.view(-1)[action]

        before_location = self.environment.get_agent_location()
        step_result = self.episode.step(action)
        after_location = self.environment.get_agent_location()
        self.last_reward = (
            None if step_result["reward"] is None else float(step_result["reward"])
        )

        step_result["before_location"] = before_location
        step_result["after_location"] = after_location
        eval["probs"] = probs
        eval["action"] = action
        eval["step_result"] = step_result
        eval["training"] = train
        self.step_results.append(step_result)
        self.actions.append(action)
        self.rewards.append(self.last_reward)
        if train or self.include_test_eval_results:
            self.eval_results.append(eval)
        if train:
            self.entropies.append(entropy)
            self.values.append(value)
            self.log_prob_of_actions.append(log_prob_of_action)
        return eval

    def clear_history(self) -> None:
        self.clear_graph_data()
        self.rewards = []
        self.actions = []
        self.step_results = []

    def clear_graph_data(self):
        self.values = []
        self.log_prob_of_actions = []
        self.entropies = []
        self.eval_results = []

    def loss(self, **kwargs) -> Dict[str, torch.FloatTensor]:
        assert self.episode is not None
        if self._a3c_loss_disabled_for_episode:
            return {}

        if self.episode.is_complete():
            future_reward_est = 0.0
        else:
            eval = self.eval_at_current_state()
            future_reward_est = eval["value"].item()
        return self.a3c_loss(
            values=self.values,
            rewards=self.rewards,
            log_prob_of_actions=self.log_prob_of_actions,
            entropies=self.entropies,
            future_reward_est=future_reward_est,
            gamma=self.gamma,
            tau=self.tau,
            beta=self.beta,
            gpu_id=self.gpu_id,
        )

    @staticmethod
    def a3c_loss(
        values: List[torch.FloatTensor],
        rewards: List[float],
        log_prob_of_actions: List[torch.FloatTensor],
        entropies: List[torch.FloatTensor],
        future_reward_est: float,
        gamma: float,
        tau: float,
        beta: float,
        gpu_id: int,
        huber_delta: Optional[float] = None,
    ):
        R = torch.FloatTensor([[future_reward_est]])
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()
        values = copy.copy(values)
        values.append(R)
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + (
                0.5 * advantage.pow(2)
                if huber_delta is None
                else 0.5 * huber_loss(advantage, huber_delta)
            )

            # Generalized Advantage Estimation
            delta_t = rewards[i] + gamma * values[i + 1].detach() - values[i].detach()

            gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - log_prob_of_actions[i] * gae
            entropy_loss -= beta * entropies[i]

        return {
            "policy": policy_loss,
            "value": 0.5 * value_loss,
            "entropy": entropy_loss,
        }
