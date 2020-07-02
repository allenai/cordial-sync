import os
from typing import Optional

from constants import ABS_PATH_TO_FINAL_FURNLIFT_CKPTS
from constants import SPLIT_TO_USE_FOR_EVALUATION
from rl_multi_agent.experiments.furnlift_vision_noimplicit_marginal_nocl_config import (
    FurnLiftMinDistUncoordinatedNoimplicitExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnLiftMixin


class EvalConfigMarginalVision(
    SaveFurnLiftMixin, FurnLiftMinDistUncoordinatedNoimplicitExperimentConfig,
):
    max_ep_using_expert_actions = 0

    @property
    def saved_model_path(cls) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNLIFT_CKPTS,
            "furnlift_vision_noimplicit_marginal_nocl_100000_2020-03-03_01-43-25.dat",
        )

    def simple_name(self):
        return "furnlift__vision_noimplicit_marginal_nocl"


def get_experiment():
    ec = EvalConfigMarginalVision()
    ec.episode_init_queue_file_name = "furnlift_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
