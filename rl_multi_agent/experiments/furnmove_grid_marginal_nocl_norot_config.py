from typing import Optional

from torch import nn

import constants
from rl_multi_agent.experiments.furnmove_grid_marginal_nocl_base_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.furnmove_episodes import (
    FurnMoveEgocentricNoRotationsFastGridEpisode,
)
from rl_multi_agent.models import A3CLSTMNStepComCoordinatedActionsEgoGridsEmbedCNN


class FurnMoveMarginalNoRotationsExperimentConfig(FurnMoveExperimentConfig):
    episode_class = FurnMoveEgocentricNoRotationsFastGridEpisode
    # Increasing the params of marginal to match mixture
    final_cnn_channels = 288

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

    @property
    def saved_model_path(self) -> Optional[str]:
        return None

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        def _create_model(**kwargs):
            return A3CLSTMNStepComCoordinatedActionsEgoGridsEmbedCNN(
                **{
                    **dict(
                        num_inputs=9,
                        action_groups=cls.episode_class.class_available_action_groups(
                            include_move_obj_actions=cls.include_move_obj_actions
                        ),
                        num_agents=cls.num_agents,
                        state_repr_length=cls.state_repr_length,
                        occupancy_embed_length=8,
                        talk_embed_length=cls.talk_embed_length,
                        agent_num_embed_length=cls.agent_num_embed_length,
                        reply_embed_length=cls.reply_embed_length,
                        turn_off_communication=cls.turn_off_communication,
                        coordinate_actions=cls.coordinate_actions,
                        coordinate_actions_dim=13 if cls.coordinate_actions else None,
                        separate_actor_weights=False,
                        num_talk_symbols=cls.num_talk_symbols,
                        num_reply_symbols=cls.num_reply_symbols,
                        final_cnn_channels=cls.final_cnn_channels,
                    ),
                    **kwargs,
                }
            )

        return _create_model(**kwargs)


def get_experiment():
    return FurnMoveMarginalNoRotationsExperimentConfig()
