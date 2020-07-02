import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_multi_agent.experiments.furnmove_grid_central_3agents_config import (
    FurnMove3AgentCentralExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


class EvalConfig(SaveFurnMoveMixin, FurnMove3AgentCentralExperimentConfig):
    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_grid_central_3agents_1000000_2020-03-02_19-28-34.dat",
        )

    def simple_name(self):
        return "grid_central_3agents"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__3agents__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
