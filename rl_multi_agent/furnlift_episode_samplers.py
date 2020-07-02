import hashlib
import itertools
import math
import os
import queue
import random
import warnings
from typing import Dict, List, Optional, Callable, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx import find_cliques
from torch import multiprocessing as mp

from constants import TELEVISION_ROTATION_TO_OCCUPATIONS
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironmentWithGraph
from rl_ai2thor.ai2thor_utils import manhattan_dists_between_positions
from rl_multi_agent import MultiAgent
from rl_multi_agent.furnlift_episodes import JointNavigationEpisode
from rl_multi_agent.furnmove_utils import at_same_location
from rl_multi_agent.multi_agent_utils import TrainingCompleteException


def create_or_append(dict: Dict[str, List], key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def save_talk_reply_data_frame(
    agent, save_path, index: Optional[int], use_hash: bool = False, prefix: str = ""
):
    num_agents = agent.environment.num_agents
    eval_results = agent.eval_results

    object_id = agent.episode.object_id
    o = agent.environment.get_object_by_id(object_id, agent_id=0)
    o_pos = {**o["position"], "rotation": o["rotation"]["y"]}

    data = {}
    for i in range(len(eval_results)):
        er = eval_results[i]
        for k in er:
            if "cpu" in dir(er[k]):
                er[k] = er[k].cpu()
        sr = er["step_result"]

        for agent_id in range(num_agents):
            agent_key = "agent_{}_".format(agent_id)

            for probs_key in ["talk_probs", "reply_probs"]:
                probs = er[probs_key][agent_id, :].detach().numpy()
                for j in range(len(probs)):
                    create_or_append(
                        data, agent_key + probs_key + "_" + str(j), probs[j]
                    )

            sr_i = sr[agent_id]
            for key in ["goal_visible", "pickup_action_taken", "action_success"]:
                create_or_append(data, agent_key + key, sr_i[key])

            create_or_append(
                data,
                agent_key + "action",
                agent.episode.available_actions[sr_i["action"]],
            )

            before_loc = sr_i["before_location"]
            for key in before_loc:
                create_or_append(data, agent_key + key, before_loc[key])

            create_or_append(
                data,
                agent_key + "l2_dist_to_object",
                math.sqrt(
                    (o_pos["x"] - before_loc["x"]) ** 2
                    + (o_pos["z"] - before_loc["z"]) ** 2
                ),
            )
            create_or_append(
                data,
                agent_key + "manhat_dist_to_object",
                abs(o_pos["x"] - before_loc["x"]) + abs(o_pos["z"] - before_loc["z"]),
            )

        for key in o_pos:
            create_or_append(data, "object_" + key, o_pos[key])

        mutual_agent_distance = int(
            4
            * (
                abs(data["agent_0_x"][-1] - data["agent_1_x"][-1])
                + abs(data["agent_0_z"][-1] - data["agent_1_z"][-1])
            )
        )
        create_or_append(data, "mutual_agent_distance", mutual_agent_distance)

    df = pd.DataFrame(
        data={**data, "scene_name": [agent.environment.scene_name] * len(eval_results)}
    )
    for df_k in df.keys():
        if df[df_k].dtype in [float, np.float32, np.float64]:
            df[df_k] = np.round(df[df_k], 4)
    if use_hash:
        file_name = (
            agent.environment.scene_name
            + "_"
            + hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
            + ".tsv"
        )
    else:
        file_name = "{}_{}_talk_reply_data.tsv".format(index, prefix)

    df.to_csv(
        os.path.join(save_path, file_name),
        sep="\t",
        # float_format="%.4f",
    )


def _create_environment(
    num_agents,
    env_args,
    visible_agents: bool,
    render_depth_image: bool,
    headless: bool = False,
    **environment_args,
) -> AI2ThorEnvironmentWithGraph:
    env = AI2ThorEnvironmentWithGraph(
        docker_enabled=env_args.docker_enabled,
        x_display=env_args.x_display,
        local_thor_build=env_args.local_thor_build,
        num_agents=num_agents,
        restrict_to_initially_reachable_points=True,
        visible_agents=visible_agents,
        render_depth_image=render_depth_image,
        headless=headless,
        **environment_args,
    )
    return env


class FurnLiftEpisodeSamplers(object):
    def __init__(
        self,
        scenes: List[str],
        num_agents: int,
        object_type: str,
        env_args=None,
        max_episode_length: int = 500,
        episode_class: Callable = JointNavigationEpisode,
        player_screen_height=224,
        player_screen_width=224,
        save_talk_reply_probs_path: Optional[str] = None,
        min_dist_between_agents_to_pickup: int = 0,
        max_ep_using_expert_actions: int = 10000,
        visible_agents: bool = True,
        max_visible_positions_push: int = 8,
        min_max_visible_positions_to_consider: Tuple[int, int] = (8, 16),
        include_depth_frame: bool = False,
        allow_agents_to_intersect: bool = False,
    ):
        self.visible_agents = visible_agents
        self.max_ep_using_expert_actions = max_ep_using_expert_actions
        self.min_dist_between_agents_to_pickup = min_dist_between_agents_to_pickup
        self.save_talk_reply_probs_path = save_talk_reply_probs_path
        self.player_screen_height = player_screen_height
        self.player_screen_width = player_screen_width
        self.episode_class = episode_class
        self.max_episode_length = max_episode_length
        self.env_args = env_args
        self.num_agents = num_agents
        self.scenes = scenes
        self.object_type = object_type
        self.max_visible_positions_push = max_visible_positions_push
        self.min_max_visible_positions_to_consider = (
            min_max_visible_positions_to_consider
        )
        self.include_depth_frame = include_depth_frame
        self.allow_agents_to_intersect = allow_agents_to_intersect

        self.grid_size = 0.25

        self._current_train_episode = 0
        self._internal_episodes = 0

    @property
    def current_train_episode(self):
        return self._current_train_episode

    @current_train_episode.setter
    def current_train_episode(self, value):
        self._current_train_episode = value

    def _contain_n_positions_at_least_dist_apart(self, positions, n, min_dist):
        for sub_positions in itertools.combinations(positions, n):
            if all(
                x >= min_dist
                for x in sum(
                    manhattan_dists_between_positions(sub_positions, self.grid_size),
                    [],
                )
            ):
                return True

        return False

    def __call__(
        self,
        agent: MultiAgent,
        agent_location_seed=None,
        env: Optional[AI2ThorEnvironmentWithGraph] = None,
        episode_init_queue: Optional[mp.Queue] = None,
    ) -> None:
        self._internal_episodes += 1

        if env is None:
            if agent.environment is not None:
                env = agent.environment
            else:
                env = _create_environment(
                    num_agents=self.num_agents,
                    env_args=self.env_args,
                    visible_agents=self.visible_agents,
                    render_depth_image=self.include_depth_frame,
                    allow_agents_to_intersect=self.allow_agents_to_intersect,
                )
                env.start(
                    "FloorPlan1_physics",
                    move_mag=self.grid_size,
                    quality="Very Low",
                    player_screen_height=self.player_screen_height,
                    player_screen_width=self.player_screen_width,
                )

        if (
            self.max_ep_using_expert_actions != 0
            and self.current_train_episode <= self.max_ep_using_expert_actions
        ):
            agent.take_expert_action_prob = 0.9 * (
                1.0 - self.current_train_episode / self.max_ep_using_expert_actions
            )
        else:
            agent.take_expert_action_prob = 0

        if episode_init_queue is not None:
            try:
                self.episode_init_data = episode_init_queue.get(timeout=1)
            except queue.Empty:
                raise TrainingCompleteException("No more data in episode init queue.")

            scene = self.episode_init_data["scene"]
            env.reset(scene)
            assert self.object_type == "Television"

            env.step(
                {
                    "action": "DisableAllObjectsOfType",
                    "objectId": self.object_type,
                    "agentId": 0,
                }
            )

            for agent_id, agent_location in enumerate(
                self.episode_init_data["agent_locations"]
            ):
                env.teleport_agent_to(
                    **agent_location,
                    agent_id=agent_id,
                    only_initially_reachable=False,
                    force_action=True,
                )
                assert env.last_event.metadata["lastActionSuccess"]

            object_location = self.episode_init_data["object_location"]
            env.step(
                {
                    "action": "CreateObjectAtLocation",
                    "objectType": self.object_type,
                    **object_location,
                    "position": {
                        "x": object_location["x"],
                        "y": object_location["y"],
                        "z": object_location["z"],
                    },
                    "rotation": {"x": 0, "y": object_location["rotation"], "z": 0},
                    "agentId": 0,
                    "forceAction": True,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            env.refresh_initially_reachable()

            objects_of_type = env.all_objects_with_properties(
                {"objectType": self.object_type}, agent_id=0
            )
            if len(objects_of_type) != 1:
                print("len(objects_of_type): {}".format(len(objects_of_type)))
                raise (Exception("len(objects_of_type) != 1"))

            object = objects_of_type[0]
            object_id = object["objectId"]
            obj_rot = int(object["rotation"]["y"])
            object_points_set = set(
                (
                    round(object["position"]["x"] + t[0], 2),
                    round(object["position"]["z"] + t[1], 2),
                )
                for t in TELEVISION_ROTATION_TO_OCCUPATIONS[obj_rot]
            )

            # Find agent metadata from where the target is visible
            env.step(
                {
                    "action": "GetPositionsObjectVisibleFrom",
                    "objectId": object_id,
                    "agentId": 0,
                }
            )
            possible_targets = env.last_event.metadata["actionVector3sReturn"]
            distances = []
            for i, r in enumerate(env.last_event.metadata["actionFloatsReturn"]):
                possible_targets[i]["rotation"] = int(r)
                possible_targets[i]["horizon"] = 30
                distances.append(
                    env.position_dist(object["position"], possible_targets[i])
                )

            possible_targets = [
                env.get_key(x[1])
                for x in sorted(
                    list(zip(distances, possible_targets)), key=lambda x: x[0]
                )
            ]

            if self.min_dist_between_agents_to_pickup != 0:
                possible_targets_array = np.array([t[:2] for t in possible_targets])
                manhat_dist_mat = np.abs(
                    (
                        (
                            possible_targets_array.reshape((-1, 1, 2))
                            - possible_targets_array.reshape((1, -1, 2))
                        )
                        / self.grid_size
                    ).round()
                ).sum(2)
                sufficiently_distant = (
                    manhat_dist_mat >= self.min_dist_between_agents_to_pickup
                )

                g = nx.Graph(sufficiently_distant)
                good_cliques = []
                for i, clique in enumerate(find_cliques(g)):
                    if i > 1000 or len(good_cliques) > 40:
                        break
                    if len(clique) == env.num_agents:
                        good_cliques.append(clique)
                    elif len(clique) > env.num_agents:
                        good_cliques.extend(
                            itertools.combinations(clique, env.num_agents)
                        )

                good_cliques = [
                    [possible_targets[i] for i in gc] for gc in good_cliques
                ]
            else:
                assert False, "Old code."

            if len(possible_targets) < env.num_agents:
                raise Exception(
                    (
                        "Using data from episode queue (scene {} and replication {}) "
                        "but there seem to be no good final positions for the agents?"
                    ).format(scene, self.episode_init_data["replication"])
                )
            scene_successfully_setup = True
        else:
            scene = random.choice(self.scenes)

            env.reset(scene)

            scene_successfully_setup = False
            failure_reasons = []
            for _ in range(10):
                env.step(
                    {
                        "action": "DisableAllObjectsOfType",
                        "objectId": self.object_type,
                        "agentId": 0,
                    }
                )

                for agent_id in range(self.num_agents):
                    env.randomize_agent_location(
                        agent_id=agent_id,
                        seed=agent_location_seed[agent_id]
                        if agent_location_seed
                        else None,
                        partial_position={"horizon": 30},
                        only_initially_reachable=True,
                    )

                env.step(
                    {
                        "action": "RandomlyCreateAndPlaceObjectOnFloor",
                        "objectType": self.object_type,
                        "agentId": 0,
                    }
                )
                if (
                    env.last_event.metadata["lastAction"]
                    != "RandomlyCreateAndPlaceObjectOnFloor"
                    or not env.last_event.metadata["lastActionSuccess"]
                ):
                    failure_reasons.append(
                        "Could not randomize location of {}.".format(
                            self.object_type, scene
                        )
                    )
                    continue

                env.refresh_initially_reachable()

                objects_of_type = env.all_objects_with_properties(
                    {"objectType": self.object_type}, agent_id=0
                )
                if len(objects_of_type) != 1:
                    print("len(objects_of_type): {}".format(len(objects_of_type)))
                    raise (Exception("len(objects_of_type) != 1"))

                object = objects_of_type[0]
                object_id = object["objectId"]

                obj_rot = int(object["rotation"]["y"])
                object_points_set = set(
                    (
                        round(object["position"]["x"] + t[0], 2),
                        round(object["position"]["z"] + t[1], 2),
                    )
                    for t in TELEVISION_ROTATION_TO_OCCUPATIONS[obj_rot]
                )

                for agent_id in range(self.num_agents):
                    env.randomize_agent_location(
                        agent_id=agent_id,
                        seed=agent_location_seed[agent_id]
                        if agent_location_seed
                        else None,
                        partial_position={"horizon": 30},
                        only_initially_reachable=True,
                    )

                # Find agent metadata from where the target is visible
                env.step(
                    {
                        "action": "GetPositionsObjectVisibleFrom",
                        "objectId": object_id,
                        "agentId": 0,
                    }
                )
                possible_targets = env.last_event.metadata["actionVector3sReturn"]
                distances = []
                for i, r in enumerate(env.last_event.metadata["actionFloatsReturn"]):
                    possible_targets[i]["rotation"] = int(r)
                    possible_targets[i]["horizon"] = 30
                    distances.append(
                        env.position_dist(object["position"], possible_targets[i])
                    )

                possible_targets = [
                    env.get_key(x[1])
                    for x in sorted(
                        list(zip(distances, possible_targets)), key=lambda x: x[0]
                    )
                ]

                if self.min_dist_between_agents_to_pickup != 0:
                    possible_targets_array = np.array([t[:2] for t in possible_targets])
                    manhat_dist_mat = np.abs(
                        (
                            (
                                possible_targets_array.reshape((-1, 1, 2))
                                - possible_targets_array.reshape((1, -1, 2))
                            )
                            / self.grid_size
                        ).round()
                    ).sum(2)
                    sufficiently_distant = (
                        manhat_dist_mat >= self.min_dist_between_agents_to_pickup
                    )

                    g = nx.Graph(sufficiently_distant)
                    good_cliques = []
                    for i, clique in enumerate(find_cliques(g)):
                        if i > 1000 or len(good_cliques) > 40:
                            break
                        if len(clique) == env.num_agents:
                            good_cliques.append(clique)
                        elif len(clique) > env.num_agents:
                            good_cliques.extend(
                                itertools.combinations(clique, env.num_agents)
                            )

                    if len(good_cliques) == 0:
                        failure_reasons.append(
                            "Failed to find a tuple of {} targets all {} steps apart.".format(
                                env.num_agents, self.min_dist_between_agents_to_pickup
                            )
                        )
                        continue

                    good_cliques = [
                        [possible_targets[i] for i in gc] for gc in good_cliques
                    ]
                else:
                    assert False, "Old code."
                    (
                        min_considered,
                        max_considered,
                    ) = self.min_max_visible_positions_to_consider
                    offset0 = min(
                        max(len(possible_targets) - min_considered, 0),
                        self.max_visible_positions_push,
                    )
                    # offset1 = min(
                    #     max(len(possible_targets) - max_considered, 0),
                    #     self.max_visible_positions_push,
                    # )
                    min_slice = slice(offset0, offset0 + min_considered)
                    # max_slice = slice(offset1, offset1 + max_considered)

                    possible_targets = possible_targets[min_slice]

                    good_cliques = list(
                        itertools.combinations(possible_targets, self.num_agents)
                    )

                if len(possible_targets) < env.num_agents:
                    failure_reasons.append(
                        "The number of positions from which the object was visible was less than the number of agents."
                    )
                    continue
                scene_successfully_setup = True
                break

        if not scene_successfully_setup:
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

        # Task data includes the target object id and one agent state/metadata to navigate to.
        good_cliques = {tuple(sorted(gc)) for gc in good_cliques}
        task_data = {"goal_obj_id": object_id, "target_key_groups": good_cliques}

        agent.episode = self.episode_class(
            env,
            task_data,
            max_steps=self.max_episode_length,
            num_agents=self.num_agents,
            min_dist_between_agents_to_pickup=self.min_dist_between_agents_to_pickup,
            object_points_set=object_points_set,
            include_depth_frame=self.include_depth_frame,
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


class FurnLiftForViewsEpisodeSampler(object):
    def __init__(
        self,
        scene: str,
        num_agents: int,
        object_type: str,
        agent_start_locations: List,
        object_start_location: Dict,
        env_args=None,
        max_episode_length: int = 500,
        episode_class: Callable = JointNavigationEpisode,
        player_screen_height=224,
        player_screen_width=224,
        save_talk_reply_probs_path: Optional[str] = None,
        min_dist_between_agents_to_pickup: int = 0,
        visible_agents: bool = True,
    ):
        self.scene = scene
        self.agent_start_locations = agent_start_locations
        self.object_start_location = object_start_location
        self.visible_agents = visible_agents
        self.min_dist_between_agents_to_pickup = min_dist_between_agents_to_pickup
        self.save_talk_reply_probs_path = save_talk_reply_probs_path
        self.player_screen_height = player_screen_height
        self.player_screen_width = player_screen_width
        self.episode_class = episode_class
        self.max_episode_length = max_episode_length
        self.env_args = env_args
        self.num_agents = num_agents
        self.object_type = object_type

        self.grid_size = 0.25

        self._current_train_episode = 0
        self._internal_episodes = 0

    @property
    def current_train_episode(self):
        return self._current_train_episode

    @current_train_episode.setter
    def current_train_episode(self, value):
        self._current_train_episode = value

    def __call__(
        self,
        agent: MultiAgent,
        agent_location_seed=None,
        env: Optional[AI2ThorEnvironmentWithGraph] = None,
    ) -> None:
        self._internal_episodes += 1
        agent.take_expert_action_prob = 0

        if env is None:
            if agent.environment is not None:
                env = agent.environment
            else:
                env = _create_environment(
                    num_agents=self.num_agents,
                    env_args=self.env_args,
                    visible_agents=self.visible_agents,
                    render_depth_image=False,
                )
                env.start(
                    self.scene,
                    move_mag=self.grid_size,
                    quality="Very Low",
                    player_screen_height=self.player_screen_height,
                    player_screen_width=self.player_screen_width,
                )
        env.reset(self.scene)

        env.step(
            {
                "action": "DisableAllObjectsOfType",
                "objectId": self.object_type,
                "agentId": 0,
            }
        )

        object_pos = {
            "position": {k: self.object_start_location[k] for k in ["x", "y", "z"]},
            "rotation": {"x": 0, "y": self.object_start_location["rotation"], "z": 0},
        }

        env.teleport_agent_to(
            **{
                **self.agent_start_locations[0],
                "rotation": self.agent_start_locations[0]["rotation"]["y"],
            },
            force_action=True,
            agent_id=0,
        )
        assert at_same_location(
            env.last_event.metadata["agent"], self.agent_start_locations[0]
        )
        env.teleport_agent_to(
            **{
                **self.agent_start_locations[1],
                "rotation": self.agent_start_locations[1]["rotation"]["y"],
            },
            force_action=True,
            agent_id=1,
        )
        assert at_same_location(
            env.last_event.metadata["agent"], self.agent_start_locations[1]
        )

        env.step(
            {
                "action": "CreateObjectAtLocation",
                "objectType": self.object_type,
                "agentId": 0,
                **object_pos,
            }
        )
        assert env.last_event.metadata["lastActionSuccess"]

        env.refresh_initially_reachable()

        objects_of_type = env.all_objects_with_properties(
            {"objectType": self.object_type}, agent_id=0
        )
        if len(objects_of_type) != 1:
            print("len(objects_of_type): {}".format(len(objects_of_type)))
            raise (Exception("len(objects_of_type) != 1"))
        object = objects_of_type[0]
        assert at_same_location(object, object_pos, ignore_camera=True)
        object_id = object["objectId"]

        obj_rot = int(object["rotation"]["y"])
        object_points_set = set(
            (
                round(object["position"]["x"] + t[0], 2),
                round(object["position"]["z"] + t[1], 2),
            )
            for t in TELEVISION_ROTATION_TO_OCCUPATIONS[obj_rot]
        )

        # Find agent metadata from where the target is visible
        env.step(
            {
                "action": "GetPositionsObjectVisibleFrom",
                "objectId": object_id,
                "agentId": 0,
            }
        )
        possible_targets = env.last_event.metadata["actionVector3sReturn"]
        distances = []
        for i, r in enumerate(env.last_event.metadata["actionFloatsReturn"]):
            possible_targets[i]["rotation"] = int(r)
            possible_targets[i]["horizon"] = 30
            distances.append(env.position_dist(object["position"], possible_targets[i]))

        possible_targets = [
            x[1]
            for x in sorted(list(zip(distances, possible_targets)), key=lambda x: x[0])
        ]

        possible_targets = possible_targets[:24]

        if len(possible_targets) < env.num_agents:
            raise Exception(
                "The number of positions from which the object was visible was less than the number of agents."
            )

        # Task data includes the target object id and one agent state/metadata to naviagate to.
        task_data = {"goal_obj_id": object_id, "targets": possible_targets}

        agent.episode = self.episode_class(
            env,
            task_data,
            max_steps=self.max_episode_length,
            num_agents=self.num_agents,
            min_dist_between_agents_to_pickup=self.min_dist_between_agents_to_pickup,
            object_points_set=object_points_set,
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
