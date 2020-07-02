from typing import Optional

from torch import nn

from rl_multi_agent import MultiAgent
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

    @classmethod
    def create_agent(cls, **kwargs) -> MultiAgent:
        return cls.agent_class(
            model=kwargs["model"],
            gpu_id=kwargs["gpu_id"],
            include_test_eval_results=cls.include_test_eval_results,
            use_a3c_loss_when_not_expert_forcing=cls.use_a3c_loss_when_not_expert_forcing,
            record_all_in_test=cls.record_all_in_test,
            include_depth_frame=cls.include_depth_frame,
            resize_image_as=cls.screen_size,
            # discourage_failed_coordination=5000,
        )

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveBigCentralVisionExperimentConfig()
