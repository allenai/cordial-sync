from typing import Optional

from torch import nn

from rl_multi_agent.experiments.furnmove_vision_bigcentral_base_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.models import A3CLSTMBigCentralEgoVision


class FurnMoveBigCentralVisionExperimentConfig(FurnMoveExperimentConfig):
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return A3CLSTMBigCentralEgoVision(
            num_inputs_per_agent=3 + 1 * cls.include_depth_frame,
            action_groups=cls.episode_class.class_available_action_groups(
                include_move_obj_actions=cls.include_move_obj_actions
            ),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
        )

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveBigCentralVisionExperimentConfig()
