import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNLIFT_CKPTS
from rl_multi_agent.experiments.furnlift_vision_marginal_nocl_config import (
    FurnLiftMinDistUncoordinatedExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnLiftMixin


class EvalConfigMarginalVision(
    SaveFurnLiftMixin, FurnLiftMinDistUncoordinatedExperimentConfig
):
    max_ep_using_expert_actions = 0

    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNLIFT_CKPTS,
            "furnlift_vision_marginal_nocl_100000_2020-02-28_09-16-44.dat",
        )

    def simple_name(self):
        return "furnlift__vision_marginal_nocl"


def get_experiment():
    ec = EvalConfigMarginalVision()
    ec.episode_init_queue_file_name = "furnlift_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
