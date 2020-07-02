from typing import Callable, Optional

import torch

import constants
from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.experiment import ExperimentConfig
from rl_multi_agent.furnmove_episode_samplers import FurnMoveGridEpisodeSampler
from rl_multi_agent.furnmove_episodes import FurnMoveEgocentricFastGridEpisode


class FurnMoveExperimentConfig(ExperimentConfig):
    # Env/episode config
    num_agents = 2
    screen_size = 84
    episode_class = FurnMoveEgocentricFastGridEpisode
    frame_type = "fast-egocentric-relative-tensor"
    episode_sampler_class = FurnMoveGridEpisodeSampler
    visible_agents = True
    include_depth_frame = False
    include_move_obj_actions = True
    headless = True if torch.cuda.is_available() else False

    # Model config
    state_repr_length = 512

    # Agent config
    agent_class = MultiAgent

    # Training config
    max_ep_using_expert_actions = 0
    train_scenes = constants.TRAIN_SCENE_NAMES[20:40]
    valid_scenes = constants.VALID_SCENE_NAMES[5:10]
    use_a3c_loss_when_not_expert_forcing = True

    # Misc (e.g. visualization)
    record_all_in_test = False
    save_talk_reply_probs_path = None
    include_test_eval_results = not torch.cuda.is_available()
    return_likely_successfuly_move_actions = False

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
    def get_init_valid_params(cls):
        init_valid_params = {
            **cls.get_init_train_params(),
            "scenes": cls.valid_scenes,
            "player_screen_height": 224,
            "player_screen_width": 224,
            "headless": False,
        }
        if cls.save_talk_reply_probs_path is not None:
            init_valid_params[
                "save_talk_reply_probs_path"
            ] = cls.save_talk_reply_probs_path
        # init_valid_params["episode_args"]["visualize_test_gridworld"] = True
        # init_valid_params["episode_args"]["visualizing_ms"] = 50
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
            discourage_failed_coordination=5000,
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


def get_experiment():
    return FurnMoveExperimentConfig()
