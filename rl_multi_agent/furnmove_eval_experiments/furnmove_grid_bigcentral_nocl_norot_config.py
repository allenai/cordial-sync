import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_multi_agent.experiments.furnmove_grid_bigcentral_nocl_norot_config import (
    FurnMoveNoRotationsExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


class EvalConfig(SaveFurnMoveMixin, FurnMoveNoRotationsExperimentConfig):
    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_grid_bigcentral_nocl_norot_1000000_2019-11-19_07-42-15.dat",
        )

    def simple_name(self):
        return "grid_bigcentral_nocl_norot"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
