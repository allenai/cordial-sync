from typing import Optional

import constants
from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.furnmove_grid_mixture_base_config import (
    FurnMoveExperimentConfig,
)


class FurnMoveGridExperimentConfig(FurnMoveExperimentConfig):
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
                "joint_pass_penalty": -0.09,
                "moved_closer_reward": 1.0,
                "min_dist_to_to_object": 0.26,
                "reached_target_reward": 1.0,
                "return_likely_successfuly_move_actions": cls.return_likely_successfuly_move_actions,
                "frame_type": "fast-egocentric-relative-tensor",
                "pass_conditioned_coordination": True,
            },
        }
        return init_train_params

    @classmethod
    def create_agent(cls, **kwargs) -> MultiAgent:
        return cls.agent_class(
            model=kwargs["model"],
            gpu_id=kwargs["gpu_id"],
            include_test_eval_results=cls.include_test_eval_results,
            use_a3c_loss_when_not_expert_forcing=cls.use_a3c_loss_when_not_expert_forcing,
            record_all_in_test=cls.record_all_in_test,
            include_depth_frame=cls.include_depth_frame,
        )

    @property
    def saved_model_path(self) -> Optional[str]:
        return None


def get_experiment():
    return FurnMoveGridExperimentConfig()
