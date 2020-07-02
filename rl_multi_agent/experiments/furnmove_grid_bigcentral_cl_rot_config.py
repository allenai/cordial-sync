from typing import Optional

from torch import nn

from rl_multi_agent.experiments.furnmove_grid_bigcentral_base_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.models import A3CLSTMBigCentralEgoGridsEmbedCNN


class FurnMoveBigCentralExperimentConfig(FurnMoveExperimentConfig):
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return A3CLSTMBigCentralEgoGridsEmbedCNN(
            num_inputs_per_agent=9,
            action_groups=cls.episode_class.class_available_action_groups(
                include_move_obj_actions=cls.include_move_obj_actions
            ),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
            occupancy_embed_length=8,
        )

    @classmethod
    def get_init_train_params(cls):
        init_train_params = FurnMoveExperimentConfig.get_init_train_params()
        init_train_params["environment_args"] = {"min_steps_between_agents": 2}
        return init_train_params

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveBigCentralExperimentConfig()
