import torch.nn as nn

from rl_multi_agent.experiments.furnmove_grid_mixture_3agents_config import (
    FurnMove3AgentMixtureExperimentConfig as FurnMoveCoordinatedMultiAgentConfig,
)
from rl_multi_agent.models import A3CLSTMCentralEgoGridsEmbedCNN


class FurnMove3AgentCentralExperimentConfig(FurnMoveCoordinatedMultiAgentConfig):
    num_agents = 3

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return A3CLSTMCentralEgoGridsEmbedCNN(
            num_inputs_per_agent=13,
            action_groups=cls.episode_class.class_available_action_groups(
                include_move_obj_actions=cls.include_move_obj_actions
            ),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
            occupancy_embed_length=8,
        )


def get_experiment():
    return FurnMove3AgentCentralExperimentConfig()
