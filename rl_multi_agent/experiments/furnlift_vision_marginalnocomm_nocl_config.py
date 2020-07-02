from torch import nn

from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.furnlift_base_config import FurnLiftBaseConfig
from rl_multi_agent.experiments.furnlift_vision_mixture_cl_config import (
    FurnLiftMinDistMixtureConfig,
)
from rl_multi_agent.models import BigA3CLSTMNStepComCoordinatedActionsEgoVision
from utils.net_util import (
    binary_search_for_model_with_least_upper_bound_parameters,
    count_parameters,
)


class FurnLiftMinDistUncoordinatedNocommExperimentConfig(FurnLiftBaseConfig):
    # Env/episode config
    min_dist_between_agents_to_pickup = 8
    visible_agents = True
    # Model config
    turn_off_communication = True
    # Agent config
    agent_class = MultiAgent
    # Coordination
    coordinate_actions = False  # Marginal

    # Balancing
    final_cnn_channels = 64
    increased_params_for_balancing_marginal = True
    balance_against_model_function = FurnLiftMinDistMixtureConfig.create_model

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        def _create_model(**kwargs):
            return BigA3CLSTMNStepComCoordinatedActionsEgoVision(
                **{
                    **dict(
                        num_inputs_per_agent=3 + 1 * (cls.include_depth_frame),
                        action_groups=cls.episode_class.class_available_action_groups(),
                        num_agents=cls.num_agents,
                        state_repr_length=cls.state_repr_length,
                        talk_embed_length=cls.talk_embed_length,
                        agent_num_embed_length=cls.agent_num_embed_length,
                        reply_embed_length=cls.reply_embed_length,
                        num_talk_symbols=cls.num_talk_symbols,
                        num_reply_symbols=cls.num_reply_symbols,
                        turn_off_communication=cls.turn_off_communication,
                        coordinate_actions=cls.coordinate_actions,
                        separate_actor_weights=False,
                        final_cnn_channels=cls.final_cnn_channels,
                    ),
                    **kwargs,
                }
            )

        if cls.increased_params_for_balancing_marginal:
            final_cnn_channels = binary_search_for_model_with_least_upper_bound_parameters(
                target_parameter_count=count_parameters(
                    cls.balance_against_model_function()
                ),
                create_model_func=lambda x: _create_model(final_cnn_channels=x),
                lower=cls.final_cnn_channels,
                upper=2 * cls.final_cnn_channels,
            )
            kwargs["final_cnn_channels"] = final_cnn_channels
            return _create_model(**kwargs)
        else:
            return _create_model(**kwargs)


def get_experiment():
    return FurnLiftMinDistUncoordinatedNocommExperimentConfig()
