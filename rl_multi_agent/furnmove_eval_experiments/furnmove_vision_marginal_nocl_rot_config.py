import os
from typing import Optional

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_multi_agent.experiments.furnmove_vision_marginal_nocl_rot_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.furnmove_eval_experiments.furnmove_vision_mixture_cl_rot_config import (
    before_step,
    after_step,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


class EvalConfig(SaveFurnMoveMixin, FurnMoveExperimentConfig):
    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_vision_marginal_nocl_rot_500000_2020-02-21_11-11-10.dat",
        )

    @classmethod
    def get_init_train_params(cls):
        init_train_params = super(EvalConfig, cls).get_init_train_params()
        init_train_params["episode_args"]["before_step_function"] = before_step
        init_train_params["episode_args"]["after_step_function"] = after_step
        return init_train_params

    def simple_name(self):
        return "vision_marginal_nocl_rot"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
