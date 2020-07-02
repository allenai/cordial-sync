from rl_multi_agent.experiments.furnlift_vision_central_cl_config import (
    FurnLiftExperimentConfig,
)


class FurnLiftNoimplicitExperimentConfig(FurnLiftExperimentConfig):
    # Env/episode config
    visible_agents = False


def get_experiment():
    return FurnLiftNoimplicitExperimentConfig()
