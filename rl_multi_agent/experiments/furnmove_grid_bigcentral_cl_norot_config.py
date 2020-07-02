from typing import Optional

from torch import nn

import constants
from rl_multi_agent.experiments.furnmove_grid_bigcentral_base_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.furnmove_episodes import (
    FurnMoveEgocentricNoRotationsFastGridEpisode,
)
from rl_multi_agent.models import A3CLSTMBigCentralEgoGridsEmbedCNN


class FurnMoveNoRotationsExperimentConfig(FurnMoveExperimentConfig):
    episode_class = FurnMoveEgocentricNoRotationsFastGridEpisode

    @classmethod
    def get_init_train_params(cls):
        init_train_params = {
            "scenes": cls.train_scenes,
            "num_agents": cls.num_agents,
            "object_type": "Television",
            "to_object_type": "Dresser",
            "to_object_silhouette": constants.DRESSER_SILHOUETTE_STRING,
            "episode_class": cls.episode_class,
            "player_screen_height": cls.screen_size,
            "player_screen_width": cls.screen_size,
            "max_ep_using_expert_actions": cls.max_ep_using_expert_actions,
            "visible_agents": cls.visible_agents,
            "include_depth_frame": cls.include_depth_frame,
            "object_initial_height": 1.3,
            "headless": cls.headless,
            "max_distance_from_object": 0.76,
            "max_episode_length": 500,
            "environment_args": {"min_steps_between_agents": 2},
            "episode_args": {
                "include_move_obj_actions": cls.include_move_obj_actions,
                "first_correct_coord_reward": 0.0,
                "exploration_bonus": 0.0,
                "failed_action_penalty": -0.02,
                "step_penalty": -0.01,
                # "increasing_rotate_penalty": True,
                "joint_pass_penalty": -0.09,
                "moved_closer_reward": 1.0,
                # "moved_closer_reward": 0.50,
                "min_dist_to_to_object": 0.26,
                "frame_type": cls.frame_type,
                "reached_target_reward": 1.0,
                "return_likely_successfuly_move_actions": cls.return_likely_successfuly_move_actions,
                "pass_conditioned_coordination": True,
            },
        }
        return init_train_params

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return A3CLSTMBigCentralEgoGridsEmbedCNN(
            num_inputs_per_agent=9,
            action_groups=cls.episode_class.class_available_action_groups(
                include_move_obj_actions=cls.include_move_obj_actions
            ),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
            occupancy_embed_length=8,
        )

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveNoRotationsExperimentConfig()
