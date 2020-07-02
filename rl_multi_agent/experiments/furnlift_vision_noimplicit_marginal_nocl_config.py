from rl_multi_agent.experiments.furnlift_vision_marginal_nocl_config import (
    FurnLiftMinDistUncoordinatedExperimentConfig,
)


class FurnLiftMinDistUncoordinatedNoimplicitExperimentConfig(
    FurnLiftMinDistUncoordinatedExperimentConfig
):
    # Env/episode config
    visible_agents = False


def get_experiment():
    return FurnLiftMinDistUncoordinatedNoimplicitExperimentConfig()
