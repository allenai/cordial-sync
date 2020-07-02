import os
from typing import Optional, List, Dict

from constants import SPLIT_TO_USE_FOR_EVALUATION, ABS_PATH_TO_FINAL_FURNMOVE_CKPTS
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_base import Episode
from rl_multi_agent.experiments.furnmove_vision_mixture_cl_rot_config import (
    FurnMoveMixtureVisionExperimentConfig,
)
from rl_multi_agent.furnmove_utils import SaveFurnMoveMixin


def add_tv_and_dresser_info_to_info(info, env, agent_id):
    dresser = [
        o
        for o in env.all_objects_with_properties(
            {"objectType": "Dresser"}, agent_id=agent_id
        )
        if len(o["objectId"].split("|")) == 2
    ][0]
    info["dresser_visible"] = dresser["visible"]

    if agent_id == 0:
        info["dresser_location"] = {
            "position": dresser["position"],
            "rotation": dresser["rotation"],
        }

    television = [
        o
        for o in env.all_objects_with_properties(
            {"objectType": "Television"}, agent_id=agent_id
        )
        if len(o["objectId"].split("|")) == 2
    ][0]
    info["tv_visible"] = television["visible"]
    if agent_id == 0:
        info["tv_location"] = {
            "position": television["position"],
            "rotation": television["rotation"],
        }


def before_step(episode: Episode):
    env: AI2ThorEnvironment = episode.environment

    extra_infos = []
    for agent_id in range(env.num_agents):
        info = {}
        add_tv_and_dresser_info_to_info(info=info, env=env, agent_id=agent_id)
        extra_infos.append(info)

    return extra_infos


def after_step(
    step_results: List[Dict], before_info: Optional[List[Dict]], episode: Episode
):
    after_info = before_step(episode=episode)
    for sr, bi, ai in zip(step_results, before_info, after_info):
        sr["extra_before_info"] = bi
        sr["extra_after_info"] = ai


class EvalConfig(SaveFurnMoveMixin, FurnMoveMixtureVisionExperimentConfig):
    @property
    def saved_model_path(self) -> Optional[str]:
        return os.path.join(
            ABS_PATH_TO_FINAL_FURNMOVE_CKPTS,
            "furnmove_vision_mixture_cl_rot_500000_2019-11-09_19-24-52.dat",
        )

    @classmethod
    def get_init_train_params(cls):
        init_train_params = super(EvalConfig, cls).get_init_train_params()
        init_train_params["episode_args"]["before_step_function"] = before_step
        init_train_params["episode_args"]["after_step_function"] = after_step
        return init_train_params

    def simple_name(self):
        return "vision_mixture_cl_rot"


def get_experiment():
    ec = EvalConfig()
    ec.episode_init_queue_file_name = "furnmove_episode_start_positions_for_eval__{}.json".format(
        SPLIT_TO_USE_FOR_EVALUATION
    )
    return ec
