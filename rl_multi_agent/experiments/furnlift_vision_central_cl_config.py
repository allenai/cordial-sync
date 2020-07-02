from torch import nn

from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.furnlift_base_config import FurnLiftBaseConfig
from rl_multi_agent.models import OldTaskA3CLSTMBigCentralEgoVision


class FurnLiftExperimentConfig(FurnLiftBaseConfig):
    # Env/episode config
    min_dist_between_agents_to_pickup = 8
    visible_agents = True
    # Agent config
    agent_class = MultiAgent

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OldTaskA3CLSTMBigCentralEgoVision(
            num_inputs_per_agent=3 + 1 * (cls.include_depth_frame),
            action_groups=cls.episode_class.class_available_action_groups(),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
        )


def get_experiment():
    return FurnLiftExperimentConfig()
