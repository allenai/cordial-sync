from typing import Optional

from torch import nn

from rl_multi_agent.experiments.furnmove_grid_marginalnocomm_base_config import (
    FurnMoveExperimentConfig,
)
from rl_multi_agent.models import A3CLSTMNStepComCoordinatedActionsEgoGridsEmbedCNN


class FurnMoveGridExperimentConfig(FurnMoveExperimentConfig):
    # Increasing the params of marginal to match mixture
    final_cnn_channels = 288

    @classmethod
    def get_init_train_params(cls):
        init_train_params = FurnMoveExperimentConfig.get_init_train_params()
        init_train_params["environment_args"] = {"min_steps_between_agents": 2}
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
    return FurnMoveGridExperimentConfig()
