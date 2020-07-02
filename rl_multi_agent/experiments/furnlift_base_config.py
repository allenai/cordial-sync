from abc import ABC
from typing import Callable, Optional

import torch

import constants
from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.experiment import ExperimentConfig
from rl_multi_agent.furnlift_episode_samplers import FurnLiftEpisodeSamplers
from rl_multi_agent.furnlift_episodes import FurnLiftNApartStateEpisode


class FurnLiftBaseConfig(ExperimentConfig, ABC):
    # Env/episode config
    num_agents = 2
    screen_size = 84
    episode_class = FurnLiftNApartStateEpisode
    episode_sampler_class = FurnLiftEpisodeSamplers
    min_dist_between_agents_to_pickup = None
    visible_agents = None
    include_depth_frame = False
    # CVPR 2019 baselines allowed intersection of agent.
    # With new capabilities in AI2Thor, we can switch on/off this option
    allow_agents_to_intersect = True

    # Model config
    state_repr_length = 512
    talk_embed_length = 8
    reply_embed_length = 8
    agent_num_embed_length = 8
    num_talk_symbols = 2
    num_reply_symbols = 2

    # Agent config
    agent_class = None
    turn_off_communication = None

    # Training config
    max_ep_using_expert_actions = 10000
    dagger_mode = True
    train_scenes = constants.TRAIN_SCENE_NAMES[20:40]
    valid_scenes = constants.VALID_SCENE_NAMES[5:10]
    use_a3c_loss_when_not_expert_forcing = True

    # Misc (e.g. visualization)
    record_all_in_test = False
    save_talk_reply_probs_path = None
    include_test_eval_results = not torch.cuda.is_available()

    # Balancing params
    increased_params_for_balancing_marginal = False

    @classmethod
    def get_init_train_params(cls):
        init_train_params = {
            "scenes": cls.train_scenes,
            "num_agents": cls.num_agents,
            "object_type": "Television",
            "episode_class": cls.episode_class,
            "player_screen_height": cls.screen_size,
            "player_screen_width": cls.screen_size,
            "min_dist_between_agents_to_pickup": cls.min_dist_between_agents_to_pickup,
            "max_ep_using_expert_actions": cls.max_ep_using_expert_actions,
            "visible_agents": cls.visible_agents,
            "include_depth_frame": cls.include_depth_frame,
            "allow_agents_to_intersect": cls.allow_agents_to_intersect,
        }
        return init_train_params

    @classmethod
    def get_init_valid_params(cls):
        init_valid_params = {**cls.get_init_train_params(), "scenes": cls.valid_scenes}
        if cls.save_talk_reply_probs_path is not None:
            init_valid_params[
                "save_talk_reply_probs_path"
            ] = cls.save_talk_reply_probs_path
        return init_valid_params

    def __init__(self):
        self._init_train_agent = self.episode_sampler_class(
            **self.get_init_train_params()
        )
        self._init_test_agent = self.episode_sampler_class(
            **self.get_init_valid_params()
        )

    @classmethod
    def create_agent(cls, **kwargs) -> MultiAgent:
        return cls.agent_class(
            model=kwargs["model"],
            gpu_id=kwargs["gpu_id"],
            include_test_eval_results=cls.include_test_eval_results,
            use_a3c_loss_when_not_expert_forcing=cls.use_a3c_loss_when_not_expert_forcing,
            record_all_in_test=cls.record_all_in_test,
            include_depth_frame=cls.include_depth_frame,
            resize_image_as=cls.screen_size,
            dagger_mode=cls.dagger_mode,
        )

    @property
    def init_train_agent(self) -> Callable:
        return self._init_train_agent

    @property
    def init_test_agent(self) -> Callable:
        return self._init_test_agent

    @property
    def saved_model_path(self) -> Optional[str]:
        return None
