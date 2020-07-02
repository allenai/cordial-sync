import abc
from typing import Dict, Callable, Any, List, Optional

import torch
from torch import nn

from rl_base.agent import RLAgent
from rl_multi_agent import MultiAgent


class ExperimentConfig(abc.ABC):
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError()

    @classmethod
    def create_agent(cls, **kwargs) -> MultiAgent:
        raise NotImplementedError()

    @property
    def saved_model_path(self) -> Dict[str, str]:
        raise NotImplementedError()

    @property
    def init_train_agent(self) -> Callable:
        raise NotImplementedError()

    @property
    def init_test_agent(self) -> Callable:
        raise NotImplementedError()

    def create_episode_summary(
        self,
        agent: RLAgent,
        additional_metrics: Dict[str, float],
        step_results: List[Dict],
    ) -> Any:
        """Summarize an episode.

        On a worker thread, generate an episode summary that is
        passed to the :func:`save_episode_summary <ExperimentConfig.save_episode_summary>`
        method of of experiment object on the Train/Test Manager main thread. This
        can be used to summarize episodes and then save them for analysis.

        :param agent: Agent from the worker thread.
        :param additional_metrics: Mean of metrics, across the episode, saved in
                                   agent.eval_results[i]["additional_metrics"] which is of type
                                   Dict[str, float].
        :param step_results: The step results from the episode.
        :return: Data to be saved.
        """
        return None

    def save_episode_summary(self, data_to_save: Any):
        raise RuntimeError("Attempting to save episode summary but do not know how.")

    def create_episode_init_queue(
        self, mp_module
    ) -> Optional[torch.multiprocessing.Queue]:
        raise RuntimeError(
            "Attempting to create an episode init queue but do not know how."
        )

    def stopping_criteria_reached(self):
        return False
