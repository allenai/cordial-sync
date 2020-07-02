from rl_multi_agent.experiments.furnlift_vision_marginalnocomm_nocl_config import (
    FurnLiftMinDistUncoordinatedNocommExperimentConfig,
)


class FurnLiftMinDistUncoordinatedNocommNoimplicitExperimentConfig(
    FurnLiftMinDistUncoordinatedNocommExperimentConfig
):
    # Env/episode config
    visible_agents = False


def get_experiment():
    return FurnLiftMinDistUncoordinatedNocommNoimplicitExperimentConfig()
