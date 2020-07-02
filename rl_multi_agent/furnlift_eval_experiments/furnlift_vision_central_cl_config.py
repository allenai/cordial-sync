import os
from typing import Optional

from constants import ABS_PATH_TO_FINAL_FURNLIFT_CKPTS
from constants import SPLIT_TO_USE_FOR_EVALUATION
from rl_multi_agent.experiments.furnlift_vision_central_cl_config import (
    FurnLiftExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnLiftMixin


class EvalConfigCentralVision(SaveFurnLiftMixin, FurnLiftExperimentConfig):
    max_ep_using_expert_actions = 0

    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNLIFT_CKPTS,
            "furnlift_vision_central_cl_100000_2020-02-26_00-00-23.dat",
        )

    def simple_name(self):
        return "furnlift__vision_central_cl"


def get_experiment():
    ec = EvalConfigCentralVision()
    ec.episode_init_queue_file_name = "furnlift_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
