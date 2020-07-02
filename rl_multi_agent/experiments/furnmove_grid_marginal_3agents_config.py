from rl_multi_agent.experiments.furnmove_grid_mixture_3agents_config import (
    FurnMove3AgentMixtureExperimentConfig as FurnMoveCoordinatedMultiAgentConfig,
)


class FurnMove3AgentUncoordinatedExperimentConfig(FurnMoveCoordinatedMultiAgentConfig):
    coordinate_actions = False
    discourage_failed_coordination = False


def get_experiment():
    return FurnMove3AgentUncoordinatedExperimentConfig()
