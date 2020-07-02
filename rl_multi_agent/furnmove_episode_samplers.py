import queue
import random
import warnings
from collections import Counter
from typing import List, Callable, Optional, Dict

import torch
from torch import multiprocessing as mp

from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.ai2thor_gridworld_environment import AI2ThorLiftedObjectGridEnvironment
from rl_base.agent import RLAgent
from rl_multi_agent import MultiAgent
from rl_multi_agent.furnlift_episode_samplers import save_talk_reply_data_frame
from rl_multi_agent.furnmove_episodes import FurnMoveEpisode
from rl_multi_agent.multi_agent_utils import TrainingCompleteException


class FurnMoveEpisodeSampler(object):
    def __init__(
        self,
        scenes: List[str],
        num_agents: int,
        object_type: str,
        to_object_type: str,
        env_args=None,
        max_episode_length: int = 500,
        episode_class: Callable = FurnMoveEpisode,
        player_screen_height=224,
        player_screen_width=224,
        save_talk_reply_probs_path: Optional[str] = None,
        max_ep_using_expert_actions: int = 10000,
        visible_agents: bool = True,
        include_depth_frame: bool = False,
        object_initial_height: float = 1.0,
        max_distance_from_object: float = 0.5,
        headless: bool = False,
        episode_args: Optional[Dict] = None,
        environment_args: Optional[Dict] = None,
    ):
        self.visible_agents = visible_agents
        self.max_ep_using_expert_actions = max_ep_using_expert_actions
        self.save_talk_reply_probs_path = save_talk_reply_probs_path
        self.player_screen_height = player_screen_height
        self.player_screen_width = player_screen_width
        self.episode_class = episode_class
        self.max_episode_length = max_episode_length
        self.env_args = env_args
        self.num_agents = num_agents
        self.scenes = scenes
        self.object_type = object_type
        self.to_object_type = to_object_type
        self.include_depth_frame = include_depth_frame
        self.object_initial_height = object_initial_height
        self.max_distance_from_object = max_distance_from_object
        self.headless = headless
        self.episode_args = episode_args if episode_args is not None else {}
        self.environment_args = environment_args if environment_args is not None else {}

        self.grid_size = 0.25

        self._current_train_episode = 0
        self._internal_episodes = 0

        self.episode_init_data = None

    @staticmethod
    def _create_environment(
        num_agents,
        env_args,
        visible_agents: bool,
        render_depth_image: bool,
        headless: bool,
        **environment_args,
    ) -> AI2ThorEnvironment:
        env = AI2ThorEnvironment(
            docker_enabled=env_args.docker_enabled,
            x_display=env_args.x_display,
            local_thor_build=env_args.local_thor_build,
            num_agents=num_agents,
            restrict_to_initially_reachable_points=True,
            visible_agents=visible_agents,
            render_depth_image=render_depth_image,
            time_scale=1.0,
            headless=headless,
            **environment_args,
        )
        return env

    @property
    def current_train_episode(self):
        return self._current_train_episode

    @current_train_episode.setter
    def current_train_episode(self, value):
        self._current_train_episode = value

    def start_env(self, agent: RLAgent, env: Optional[AI2ThorEnvironment]):
        if env is None:
            if agent.environment is not None:
                env = agent.environment
            else:
                env = self._create_environment(
                    num_agents=self.num_agents,
                    env_args=self.env_args,
                    visible_agents=self.visible_agents,
                    render_depth_image=self.include_depth_frame,
                    headless=self.headless,
                    **self.environment_args,
                )
                env.start(
                    "FloorPlan1_physics",
                    move_mag=self.grid_size,
                    quality="Very Low",
                    player_screen_height=self.player_screen_height,
                    player_screen_width=self.player_screen_width,
                )
        return env

    def __call__(
        self,
        agent: MultiAgent,
        agent_location_seed=None,
        env: Optional[AI2ThorEnvironment] = None,
        episode_init_queue: Optional[mp.Queue] = None,
    ) -> None:
        self._internal_episodes += 1

        if (
            self.max_ep_using_expert_actions != 0
            and self.current_train_episode <= self.max_ep_using_expert_actions
        ):
            agent.take_expert_action_prob = 0.9 * (
                1.0 - self.current_train_episode / self.max_ep_using_expert_actions
            )
        else:
            agent.take_expert_action_prob = 0

        env = self.start_env(agent=agent, env=env)
        self.episode_init_data = None

        if episode_init_queue is not None:
            try:
                self.episode_init_data = episode_init_queue.get(timeout=1)
            except queue.Empty:
                raise TrainingCompleteException("No more data in episode init queue.")

            scene = self.episode_init_data["scene"]
            env.reset(scene)

            agent_locations = self.episode_init_data["agent_locations"]
            for i in range(self.num_agents):
                env.teleport_agent_to(
                    **agent_locations[i], standing=True, force_action=True, agent_id=i
                )

            # Place agents in appropriate locations
            env.step(
                {
                    "action": "DisableAllObjectsOfType",
                    "objectId": self.object_type,
                    "agentId": 0,
                }
            )
            lifted_object_loc = self.episode_init_data["lifted_object_location"]
            sr = env.step(
                {
                    "action": "CreateObjectAtLocation",
                    "objectType": self.object_type,
                    "position": {
                        "x": lifted_object_loc["x"],
                        "y": lifted_object_loc["y"],
                        "z": lifted_object_loc["z"],
                    },
                    "rotation": {"x": 0, "y": lifted_object_loc["rotation"], "z": 0},
                    "forceAction": True,
                    "agentId": 0,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            objects_of_type = env.all_objects_with_properties(
                {"objectType": self.object_type}, agent_id=0
            )
            if len(objects_of_type) != 1:
                print("len(objects_of_type): {}".format(len(objects_of_type)))
                raise (Exception("len(objects_of_type) != 1"))

            object = objects_of_type[0]
            object_id = object["objectId"]

            # Create the object we should navigate to
            to_object_loc = self.episode_init_data["to_object_location"]
            env.step(
                {
                    "action": "CreateObjectAtLocation",
                    "objectType": self.to_object_type,
                    "position": {
                        "x": to_object_loc["x"],
                        "y": to_object_loc["y"],
                        "z": to_object_loc["z"],
                    },
                    "rotation": {"x": 0, "y": to_object_loc["rotation"], "z": 0},
                    "forceAction": True,
                    "agentId": 0,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            objects_of_to_type = [
                o["objectId"]
                for o in env.all_objects_with_properties(
                    {"objectType": self.to_object_type}, agent_id=0
                )
                if len(o["objectId"].split("|")) == 2
            ]
            assert len(objects_of_to_type) == 1
            to_object_id = objects_of_to_type[0]
            assert to_object_id is not None and len(to_object_id) != 0
        else:
            scene = random.choice(self.scenes)
            env.reset(scene)

            for i in range(self.num_agents):
                env.teleport_agent_to(
                    **env.last_event.metadata["agent"]["position"],
                    rotation=env.last_event.metadata["agent"]["rotation"]["y"],
                    horizon=30.0,
                    standing=True,
                    force_action=True,
                    agent_id=i,
                )

            scene_successfully_setup = False
            failure_reasons = []
            for _ in range(10):
                # Randomly generate agents looking at the TV
                env.step(
                    {
                        "action": "DisableAllObjectsOfType",
                        "objectId": self.object_type,
                        "agentId": 0,
                    }
                )
                sr = env.step(
                    {
                        "action": "RandomlyCreateLiftedFurniture",
                        "objectType": self.object_type,
                        "y": self.object_initial_height,
                        "z": self.max_distance_from_object,
                    }
                )
                if (
                    env.last_event.metadata["lastAction"]
                    != "RandomlyCreateLiftedFurniture"
                    or not env.last_event.metadata["lastActionSuccess"]
                ):
                    failure_reasons.append(
                        "Could not randomize location of {} in {}. Error message {}.".format(
                            self.object_type,
                            scene,
                            env.last_event.metadata["errorMessage"],
                        )
                    )
                    continue

                # Refreshing reachable here should not be done as it means the
                # space underneath the TV would be considered unreachable even though,
                # once the TV moves, it is.
                # env.refresh_initially_reachable()

                objects_of_type = env.all_objects_with_properties(
                    {"objectType": self.object_type}, agent_id=0
                )
                if len(objects_of_type) != 1:
                    print("len(objects_of_type): {}".format(len(objects_of_type)))
                    raise (Exception("len(objects_of_type) != 1"))

                object = objects_of_type[0]
                object_id = object["objectId"]
                assert object_id == sr.metadata["actionReturn"]

                # Create the object we should navigate to
                env.step(
                    {
                        "action": "RandomlyCreateAndPlaceObjectOnFloor",
                        "objectType": self.to_object_type,
                        "agentId": 0,
                    }
                )

                if (
                    env.last_event.metadata["lastAction"]
                    != "RandomlyCreateAndPlaceObjectOnFloor"
                    or not env.last_event.metadata["lastActionSuccess"]
                ):
                    failure_reasons.append(
                        "Could not randomize location of {} in {}. Error message: {}.".format(
                            self.to_object_type,
                            scene,
                            env.last_event.metadata["errorMessage"],
                        )
                    )
                    continue

                to_object_id = env.last_event.metadata["actionReturn"]
                assert to_object_id is not None and len(to_object_id) != 0

                scene_successfully_setup = True
                break

            if not scene_successfully_setup:
                # raise Exception(
                warnings.warn(
                    (
                        "Failed to randomly initialize objects and agents in scene {} 10 times"
                        + "for the following reasons:"
                        + "\n\t* ".join(failure_reasons)
                        + "\nTrying a new scene."
                    ).format(env.scene_name)
                )
                yield from self(agent, env=env)
                return

        assert env.scene_name == scene

        # Task data includes the target object id and one agent state/metadata to naviagate to.
        task_data = {"move_obj_id": object_id, "move_to_obj_id": to_object_id}

        agent.episode = self.episode_class(
            env,
            task_data,
            max_steps=self.max_episode_length,
            num_agents=self.num_agents,
            max_distance_from_object=self.max_distance_from_object,
            include_depth_frame=self.include_depth_frame,
            **self.episode_args,
        )

        yield True

        if self.save_talk_reply_probs_path is not None:
            if torch.cuda.is_available():
                save_talk_reply_data_frame(
                    agent,
                    self.save_talk_reply_probs_path,
                    None,
                    use_hash=True,  # self._internal_episodes,
                )
            else:
                save_talk_reply_data_frame(
                    agent, self.save_talk_reply_probs_path, self._internal_episodes
                )


class FurnMoveGridEpisodeSampler(object):
    def __init__(
        self,
        scenes: List[str],
        num_agents: int,
        object_type: str,
        to_object_type: str,
        to_object_silhouette: str,
        env_args=None,
        max_episode_length: int = 500,
        episode_class: Callable = FurnMoveEpisode,
        player_screen_height=224,
        player_screen_width=224,
        save_talk_reply_probs_path: Optional[str] = None,
        max_ep_using_expert_actions: int = 10000,
        visible_agents: bool = True,
        include_depth_frame: bool = False,
        object_initial_height: float = 1.0,
        max_distance_from_object: float = 0.5,
        headless: bool = False,
        episode_args: Optional[Dict] = None,
        environment_args: Optional[Dict] = None,
    ):
        self.visible_agents = visible_agents
        self.max_ep_using_expert_actions = max_ep_using_expert_actions
        self.save_talk_reply_probs_path = save_talk_reply_probs_path
        self.player_screen_height = player_screen_height
        self.player_screen_width = player_screen_width
        self.episode_class = episode_class
        self.max_episode_length = max_episode_length
        self.env_args = env_args
        self.num_agents = num_agents
        self.scenes = scenes
        self.object_type = object_type
        self.to_object_type = to_object_type
        self.include_depth_frame = include_depth_frame
        self.object_initial_height = object_initial_height
        self.max_distance_from_object = max_distance_from_object
        self.headless = headless
        self.episode_args = episode_args if episode_args is not None else {}
        self.environment_args = environment_args if environment_args is not None else {}
        self.to_object_silhouette = to_object_silhouette

        self.grid_size = 0.25

        self._current_train_episode = 0
        self._internal_episodes = 0

        self.episode_init_data = None

    @staticmethod
    def _create_environment(
        num_agents,
        env_args,
        object_initial_height,
        max_distance_from_object,
        object_type,
        visible_agents: bool,
        render_depth_image: bool,
        headless: bool,
        **environment_args,
    ) -> AI2ThorLiftedObjectGridEnvironment:
        # TODO: restrict_to_initially_reachable_points
        # and render_depth_image are excluded from this.

        env = AI2ThorLiftedObjectGridEnvironment(
            docker_enabled=env_args.docker_enabled,
            x_display=env_args.x_display,
            local_thor_build=env_args.local_thor_build,
            num_agents=num_agents,
            visible_agents=visible_agents,
            time_scale=1.0,
            headless=headless,
            lifted_object_height=object_initial_height,
            max_dist_to_lifted_object=max_distance_from_object,
            object_type=object_type,
            **environment_args,
        )
        return env

    @property
    def current_train_episode(self):
        return self._current_train_episode

    @current_train_episode.setter
    def current_train_episode(self, value):
        self._current_train_episode = value

    def start_env(self, agent: RLAgent, env: Optional[AI2ThorEnvironment]):
        if agent.environment is not None:
            env = agent.environment
        else:
            env = self._create_environment(
                num_agents=self.num_agents,
                env_args=self.env_args,
                visible_agents=self.visible_agents,
                render_depth_image=self.include_depth_frame,
                headless=self.headless,
                object_initial_height=self.object_initial_height,
                max_distance_from_object=self.max_distance_from_object,
                object_type=self.object_type,
                **self.environment_args,
            )
            env.start(
                "FloorPlan1_physics",
                move_mag=self.grid_size,
                quality="Very Low",
                player_screen_height=self.player_screen_height,
                player_screen_width=self.player_screen_width,
            )
        return env

    def __call__(
        self,
        agent: MultiAgent,
        agent_location_seed=None,
        env: Optional[AI2ThorEnvironment] = None,
        episode_init_queue: Optional[mp.Queue] = None,
    ) -> None:
        self._internal_episodes += 1

        if (
            self.max_ep_using_expert_actions != 0
            and self.current_train_episode <= self.max_ep_using_expert_actions
        ):
            agent.take_expert_action_prob = 0.9 * (
                1.0 - self.current_train_episode / self.max_ep_using_expert_actions
            )
        else:
            agent.take_expert_action_prob = 0

        env = self.start_env(agent=agent, env=env)

        if episode_init_queue is not None:
            try:
                self.episode_init_data = episode_init_queue.get(timeout=1)
            except queue.Empty:
                raise TrainingCompleteException("No more data in episode init queue.")

            scene = self.episode_init_data["scene"]
            env.reset(scene)

            agent_locations = self.episode_init_data["agent_locations"]
            for i in range(self.num_agents):
                loc = agent_locations[i]
                env.step(
                    {
                        "action": "TeleportFull",
                        "agentId": i,
                        "x": loc["x"],
                        "z": loc["z"],
                        "rotation": {"y": loc["rotation"]},
                        "allowAgentIntersection": True,
                        "makeReachable": True,
                    }
                )
                assert env.last_event.metadata["lastActionSuccess"]

            # Place agents in appropriate locations
            lifted_object_loc = self.episode_init_data["lifted_object_location"]
            env.step(
                {
                    "action": "CreateLiftedFurnitureAtLocation",
                    "objectType": self.object_type,
                    "x": lifted_object_loc["x"],
                    "z": lifted_object_loc["z"],
                    "rotation": {"y": lifted_object_loc["rotation"]},
                    "agentId": 0,
                    "forceAction": True,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]
            object_id = env.last_event.metadata["actionReturn"]

            # Create the object we should navigate to
            to_object_loc = self.episode_init_data["to_object_location"]
            env.step(
                {
                    "action": "CreateAndPlaceObjectOnFloorAtLocation",
                    "objectType": "Dresser",
                    "object_mask": env.controller.parse_template_to_mask(
                        self.to_object_silhouette
                    ),
                    **to_object_loc,
                    "rotation": {"y": to_object_loc["rotation"]},
                    "agentId": 0,
                    "forceAction": True,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]
            to_object_id = env.last_event.metadata["actionReturn"]["objectId"]
            assert to_object_id is not None and len(to_object_id) != 0

        else:
            scene = random.choice(self.scenes)
            env.reset(scene)

            scene_successfully_setup = False
            failure_reasons = []
            for _ in range(10 if env.num_agents <= 3 else 50):
                # Randomly generate agents looking at the TV
                sr = env.step(
                    {
                        "action": "RandomlyCreateLiftedFurniture",
                        "agentId": 0,
                        "objectType": self.object_type,
                        "y": self.object_initial_height,
                        "z": self.max_distance_from_object,
                    }
                )
                if (
                    env.last_event.metadata["lastAction"]
                    != "RandomlyCreateLiftedFurniture"
                    or not env.last_event.metadata["lastActionSuccess"]
                ):
                    failure_reasons.append(
                        "Could not randomize location of {} in {}.".format(
                            self.object_type, scene
                        )
                    )
                    continue

                object_id = sr.metadata["actionReturn"]

                # Create the object we should navigate to
                env.step(
                    {
                        "action": "RandomlyCreateAndPlaceObjectOnFloor",
                        "objectType": self.to_object_type,
                        "object_mask": env.controller.parse_template_to_mask(
                            self.to_object_silhouette
                        ),
                        "agentId": 0,
                    }
                )

                if (
                    env.last_event.metadata["lastAction"]
                    != "RandomlyCreateAndPlaceObjectOnFloor"
                    or not env.last_event.metadata["lastActionSuccess"]
                ):
                    failure_reasons.append(
                        "Could not randomize location of {} in {}.".format(
                            self.to_object_type, scene
                        )
                    )
                    continue

                to_object_id = env.last_event.metadata["actionReturn"]["objectId"]
                assert to_object_id is not None and len(to_object_id) != 0

                scene_successfully_setup = True
                break

            if not scene_successfully_setup:
                # raise Exception(
                warnings.warn(
                    (
                        "Failed to randomly initialize objects and agents in scene {} 10 times".format(
                            scene
                        )
                        + "for the following reasons:"
                        + "\n\t* ".join(
                            [
                                "({} times): {}".format(cnt, reason)
                                for reason, cnt in Counter(failure_reasons).items()
                            ]
                        )
                        + "\nTrying a new scene."
                    ).format(env.scene_name)
                )
                yield from self(agent, env=env)
                return

        assert env.scene_name == scene

        # Task data includes the target object id and one agent state/metadata to naviagate to.
        task_data = {"move_obj_id": object_id, "move_to_obj_id": to_object_id}

        agent.episode = self.episode_class(
            env,
            task_data,
            max_steps=self.max_episode_length,
            num_agents=self.num_agents,
            max_distance_from_object=self.max_distance_from_object,
            include_depth_frame=self.include_depth_frame,
            **self.episode_args,
        )

        yield True

        if self.save_talk_reply_probs_path is not None:
            if torch.cuda.is_available():
                save_talk_reply_data_frame(
                    agent,
                    self.save_talk_reply_probs_path,
                    None,
                    use_hash=True,  # self._internal_episodes,
                )
            else:
                save_talk_reply_data_frame(
                    agent, self.save_talk_reply_probs_path, self._internal_episodes
                )
