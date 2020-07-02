from typing import Optional

from rl_multi_agent.agents import MultiAgent
from rl_multi_agent.experiments.furnmove_vision_marginal_nocl_rot_config import (
    FurnMoveExperimentConfig,
)


class FurnMoveVisionMarginalWithCLExperimentConfig(FurnMoveExperimentConfig):
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
            discourage_failed_coordination=5000,
        )

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveVisionMarginalWithCLExperimentConfig()
