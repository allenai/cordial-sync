import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_multi_agent.experiments.furnmove_grid_marginal_3agents_config import (
    FurnMove3AgentUncoordinatedExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


class EvalConfig(SaveFurnMoveMixin, FurnMove3AgentUncoordinatedExperimentConfig):
    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_grid_marginal_3agents_1000000_2020-02-28_00-24-58.dat",
        )

    def simple_name(self):
        return "grid_marginal_3agents"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__3agents__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
