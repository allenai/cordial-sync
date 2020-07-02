from typing import Optional

from rl_multi_agent.experiments.furnmove_grid_mixture_base_config import (
    FurnMoveExperimentConfig,
)


class FurnMoveGridMixtureExperimentConfig(FurnMoveExperimentConfig):
    @classmethod
    def get_init_train_params(cls):
        init_train_params = FurnMoveExperimentConfig.get_init_train_params()
        init_train_params["environment_args"] = {"min_steps_between_agents": 2}
        return init_train_params

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveGridMixtureExperimentConfig()
