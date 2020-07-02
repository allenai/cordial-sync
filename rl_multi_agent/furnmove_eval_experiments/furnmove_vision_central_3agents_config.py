import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_multi_agent.experiments.furnmove_vision_central_3agents_config import (
    FurnMoveCentralVision3AgentsExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


class EvalConfig(SaveFurnMoveMixin, FurnMoveCentralVision3AgentsExperimentConfig):
    @property
    def saved_model_path(cls) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_vision_central_3agents_300000_2020-03-08_21-38-23.dat",
        )

    def simple_name(self):
        return "vision_central_3agents"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__3agents__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
