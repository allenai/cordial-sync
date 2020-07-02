import itertools
import math
import warnings
from typing import Tuple, Dict, List, Any, Union

import matplotlib as mpl

import constants
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironmentWithGraph
from rl_ai2thor.ai2thor_episodes import MultiAgentAI2ThorEpisode
from rl_ai2thor.ai2thor_utils import manhattan_dists_between_positions
from .furnmove_episodes import are_actions_coordinated, coordination_type_tensor

mpl.use("Agg", force=False)


class JointNavigationEpisode(MultiAgentAI2ThorEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironmentWithGraph,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super(JointNavigationEpisode, self).__init__(
            env=env, task_data=task_data, max_steps=max_steps, **kwargs
        )
        self.initial_agent_metadata = env.get_all_agent_metadata()
        self.target_key_groups = task_data["target_key_groups"]
        self.object_id = task_data["goal_obj_id"]
        self.target_keys_set = set(o for group in self.target_key_groups for o in group)
        self.oracle_length = self.calculate_oracle_length()

    @classmethod
    def class_available_action_groups(cls, **kwargs) -> Tuple[Tuple[str, ...], ...]:
        actions = ("MoveAhead", "RotateLeft", "RotateRight", "Pass")
        return (actions,)

    def _is_goal_object_visible(self, agent_id: int) -> bool:
        return self.task_data["goal_obj_id"] in [
            o["objectId"] for o in self.environment.visible_objects(agent_id=agent_id)
        ]

    def goal_visibility(self) -> List[bool]:
        return [
            self._is_goal_object_visible(agentId)
            for agentId in range(self.environment.num_agents)
        ]

    def goal_visibility_int(self) -> List[int]:
        return [
            int(self._is_goal_object_visible(agentId))
            for agentId in range(self.environment.num_agents)
        ]

    def goal_visible_to_all(self) -> bool:
        return all(self.goal_visibility())

    def info(self):
        return {
            **super(JointNavigationEpisode, self).info(),
            "accuracy": 1 * self.goal_visible_to_all(),
            "goal_visible_to_one_agent": 1 * (sum(self.goal_visibility()) == 1),
        }

    def _step(self, action_as_int: int, agent_id: int) -> Dict[str, Any]:
        action = self.available_actions[action_as_int]
        action_dict = {"action": action, "agentId": agent_id}
        self.environment.step(action_dict)
        return {
            "action": action_as_int,
            "action_success": self.environment.last_event.events[agent_id].metadata[
                "lastActionSuccess"
            ],
        }

    def _closest_target(self, source_key):
        closest_target = None
        closest_steps = float("inf")
        for target_key in self.target_keys_set:
            nsteps = self.environment.shortest_path_length(
                source_state_key=source_key, goal_state_key=target_key
            )
            if closest_target is None or nsteps < closest_steps:
                closest_steps = nsteps
                closest_target = target_key

        return closest_target

    def next_expert_action(self) -> Tuple[Union[int, None], ...]:
        expert_actions = []
        is_goal_visible = self.goal_visibility()
        for agent_id in range(self.environment.num_agents):
            source_key = self.environment.get_key(
                self.environment.last_event.events[agent_id].metadata["agent"]
            )
            if is_goal_visible[agent_id] or source_key in self.target_keys_set:
                action = "Pass"
            else:
                action = self.environment.shortest_path_next_action(
                    source_key, self._closest_target(source_key)
                )
            expert_actions.append(self.available_actions.index(action))
        return tuple(expert_actions)

    def reached_terminal_state(self) -> bool:
        return self.goal_visible_to_all()

    def calculate_oracle_length(self):
        path_lengths = []
        for agent_id in range(self.environment.num_agents):
            source_state_key = self.environment.get_key(
                self.initial_agent_metadata[agent_id]
            )
            # goal_state_key = self.environment.get_key(self.target_metadata[agent_id])
            path_length_single = self.environment.shortest_path_length(
                source_state_key, self._closest_target(source_state_key)
            )
            path_lengths.append(path_length_single)
        return max(path_lengths)


class FurnLiftEpisode(JointNavigationEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironmentWithGraph,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super().__init__(env=env, task_data=task_data, max_steps=max_steps, **kwargs)
        self._item_successfully_picked_up = False
        self._jointly_visible_but_not_picked = 0
        self._picked_but_not_jointly_visible = 0

    @classmethod
    def class_available_action_groups(cls, **kwargs) -> Tuple[Tuple[str, ...], ...]:
        actions = ("MoveAhead", "RotateLeft", "RotateRight", "Pass", "Pickup")
        return (actions,)

    def info(self):
        return {
            **super(JointNavigationEpisode, self).info(),
            "accuracy": 1 * self._item_successfully_picked_up,
            "jointly_visible_but_not_picked": self._jointly_visible_but_not_picked,
            "picked_but_not_jointly_visible": self._picked_but_not_jointly_visible,
        }

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        assert not self.is_paused() and not self.is_complete()
        pickup_index = self.available_actions.index("Pickup")
        visibility = self.goal_visibility()
        visible_to_all = all(visibility)
        agent_tried_pickup = [i == pickup_index for i in actions_as_ints]
        all_pick_up = all(agent_tried_pickup)
        self._jointly_visible_but_not_picked += visible_to_all * (not all_pick_up)

        if visible_to_all and all(i == pickup_index for i in actions_as_ints):
            for i in range(self.environment.num_agents):
                self._increment_num_steps_taken_in_episode()
            step_results = [
                {
                    "action": pickup_index,
                    "action_success": True,
                    "goal_visible": True,
                    "pickup_action_taken": True,
                }
                for _ in range(self.environment.num_agents)
            ]
            object = self.environment.get_object_by_id(self.object_id, agent_id=0)
            self.environment.step(
                {
                    "action": "TeleportObject",
                    **object["position"],
                    "objectId": self.object_id,
                    "y": object["position"]["y"] + 1.0,
                    "rotation": object["rotation"],
                    "agentId": 0,
                }
            )
            self._item_successfully_picked_up = True
        else:
            step_results = []
            for i in range(self._env.num_agents):
                self._increment_num_steps_taken_in_episode()
                step_results.append(self._step(actions_as_ints[i], agent_id=i))
                step_results[-1]["goal_visible"] = visibility[i]
                step_results[-1]["pickup_action_taken"] = agent_tried_pickup[i]
        return step_results

    def _step(self, action_as_int: int, agent_id: int) -> Dict[str, Any]:
        action = self.available_actions[action_as_int]
        if action == "Pickup":
            self._picked_but_not_jointly_visible += 1
            metadata = self.environment.last_event.events[agent_id].metadata
            metadata["lastAction"] = "Pickup"
            metadata["lastActionSuccess"] = False
        else:
            action_dict = {"action": action, "agentId": agent_id}
            self.environment.step(action_dict)
        return {
            "action": action_as_int,
            "action_success": self.environment.last_event.events[agent_id].metadata[
                "lastActionSuccess"
            ],
        }

    def next_expert_action(self) -> Tuple[Union[int, None], ...]:
        expert_actions = []
        is_goal_visible = self.goal_visibility()
        if all(is_goal_visible):
            return tuple(
                [self.available_actions.index("Pickup")] * self.environment.num_agents
            )

        for agent_id in range(self.environment.num_agents):
            source_key = self.environment.get_key(
                self.environment.last_event.events[agent_id].metadata["agent"]
            )
            if is_goal_visible[agent_id] or source_key in self.target_keys_set:
                action = "Pass"
            else:
                action = self.environment.shortest_path_next_action(
                    source_key, self._closest_target(source_key)
                )
            expert_actions.append(self.available_actions.index(action))
        return tuple(expert_actions)

    def reached_terminal_state(self) -> bool:
        return self._item_successfully_picked_up


class FurnLiftNApartStateEpisode(FurnLiftEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironmentWithGraph,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super().__init__(env=env, task_data=task_data, max_steps=max_steps, **kwargs)
        self._min_dist_between_agents_to_pickup = kwargs[
            "min_dist_between_agents_to_pickup"
        ]

        self._pickupable_but_not_picked = 0
        self._picked_but_not_pickupable = 0
        self._picked_but_not_pickupable_distance = 0
        self._picked_but_not_pickupable_visibility = 0

        self.total_reward = 0.0

        self.coordinated_action_checker = are_actions_coordinated
        goal_object = self.environment.get_object_by_id(self.object_id, agent_id=0)
        self.object_location = dict(
            x=round(goal_object["position"]["x"], 2),
            z=round(goal_object["position"]["z"], 2),
        )
        # Initial agent locations, to calculate the expert path lengths
        self._initial_agent_locations = [
            self.environment.get_agent_location(i)
            for i in range(self.environment.num_agents)
        ]

    @staticmethod
    def location_l1_dist(loc1, loc2):
        return abs(loc1["x"] - loc2["x"]) + abs(loc1["z"] - loc2["z"])

    def info(self):
        nagents = self.environment.num_agents
        _final_l1_dist = [
            self.location_l1_dist(
                self.object_location, self.environment.get_agent_location(i)
            )
            for i in range(nagents)
        ]
        mean_final_agent_l1_distance_from_target = round(
            sum(_final_l1_dist) / len(_final_l1_dist), 2
        )

        _init_l1_dist = [
            self.location_l1_dist(self.object_location, ag_loc)
            for ag_loc in self._initial_agent_locations
        ]
        mean_initial_agent_manhattan_steps_from_target = (
            sum(_init_l1_dist) / len(_init_l1_dist)
        ) / self.environment.grid_size

        return {
            **super(JointNavigationEpisode, self).info(),
            "accuracy": 1 * self._item_successfully_picked_up,
            "pickupable_but_not_picked": self._pickupable_but_not_picked,
            "picked_but_not_pickupable": self._picked_but_not_pickupable,
            "picked_but_not_pickupable_distance": self._picked_but_not_pickupable_distance,
            "picked_but_not_pickupable_visibility": self._picked_but_not_pickupable_visibility,
            "reward": self.total_reward,
            "initial_manhattan_steps": mean_initial_agent_manhattan_steps_from_target,
            "final_manhattan_distance_from_target": mean_final_agent_l1_distance_from_target,
            "spl_manhattan": int(self._item_successfully_picked_up)
            * (
                (mean_initial_agent_manhattan_steps_from_target + 0.0001)
                / (
                    max(
                        mean_initial_agent_manhattan_steps_from_target,
                        (self.num_steps_taken_in_episode() / nagents),
                    )
                    + 0.0001
                )
            ),
        }

    def manhattan_dists_between_agents(self):
        return manhattan_dists_between_positions(
            [
                self.environment.get_agent_location(i)
                for i in range(self.environment.num_agents)
            ],
            self.environment.grid_size,
        )

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        assert not self.is_paused() and not self.is_complete()
        pickup_index = self.available_actions.index("Pickup")
        visibility = self.goal_visibility()
        visible_to_all = all(visibility)
        agent_tried_pickup = [i == pickup_index for i in actions_as_ints]
        all_pick_up = all(agent_tried_pickup)
        pairwise_distances = self.manhattan_dists_between_agents()
        sufficiently_far = [
            all(x >= self._min_dist_between_agents_to_pickup for x in dists)
            for dists in pairwise_distances
        ]
        all_sufficiently_far = all(sufficiently_far)

        self._pickupable_but_not_picked += (
            visible_to_all * all_sufficiently_far * (not all_pick_up)
        )

        if visible_to_all and all_sufficiently_far and all_pick_up:
            for i in range(self.environment.num_agents):
                self._increment_num_steps_taken_in_episode()
            step_results = [
                {
                    "action": pickup_index,
                    "action_success": True,
                    "goal_visible": True,
                    "pickup_action_taken": True,
                    "mutually_distant": True,
                    "reward": 1.0,
                }
                for _ in range(self.environment.num_agents)
            ]
            object = self.environment.get_object_by_id(self.object_id, agent_id=0)
            self.environment.step(
                {
                    "action": "TeleportObject",
                    **object["position"],
                    "objectId": self.object_id,
                    "y": object["position"]["y"] + 1.0,
                    "rotation": object["rotation"],
                    "agentId": 0,
                }
            )
            self._item_successfully_picked_up = True
        else:
            step_results = []
            for i in range(self._env.num_agents):
                self._increment_num_steps_taken_in_episode()
                step_results.append(self._step(actions_as_ints[i], agent_id=i))
                step_results[-1]["goal_visible"] = visibility[i]
                step_results[-1]["pickup_action_taken"] = agent_tried_pickup[i]
                self._picked_but_not_pickupable += agent_tried_pickup[i]
                self._picked_but_not_pickupable_distance += (
                    agent_tried_pickup[i] and not all_sufficiently_far
                )
                self._picked_but_not_pickupable_visibility += (
                    agent_tried_pickup[i] and not visible_to_all
                )
                step_results[-1]["mutually_distant"] = sufficiently_far[i]

        self.total_reward += sum(sr["reward"] for sr in step_results)
        return step_results

    def _step(self, action_as_int: int, agent_id: int) -> Dict[str, Any]:
        STEP_PENALITY = -0.01
        FAILED_ACTION_PENALTY = -0.02

        reward = STEP_PENALITY

        action = self.available_actions[action_as_int]
        if action == "Pickup":
            self._picked_but_not_jointly_visible += 1
            metadata = self.environment.last_event.events[agent_id].metadata
            metadata["lastAction"] = "Pickup"
            metadata["lastActionSuccess"] = False
            reward += -0.1
        else:
            action_dict = {"action": action, "agentId": agent_id}
            self.environment.step(action_dict)

        action_success = self.environment.last_event.events[agent_id].metadata[
            "lastActionSuccess"
        ]
        if not action_success:
            reward += FAILED_ACTION_PENALTY
        return {
            "reward": reward,
            "action": action_as_int,
            "action_success": action_success,
        }

    def next_expert_action(self) -> Tuple[Union[int, None], ...]:
        expert_actions = []

        visibility = self.goal_visibility()
        visible_to_all = all(visibility)
        pairwise_distances = self.manhattan_dists_between_agents()
        sufficiently_far = [
            all(x >= self._min_dist_between_agents_to_pickup for x in dists)
            for dists in pairwise_distances
        ]
        all_sufficiently_far = all(sufficiently_far)

        agent_keys = [
            self.environment.get_key(
                self.environment.last_event.events[agent_id].metadata["agent"]
            )
            for agent_id in range(self.environment.num_agents)
        ]

        # In the case that we have reached the expert targets but the expert targets
        # fail to be correct (as can happen if, for example, two agents intersect and
        # block one anothers vision) then we should remove the relevant agent_key_ind_tuple
        # from the set of good target ind tuples and have the agents move somewhere else.
        if all(k in self.target_keys_set for k in agent_keys):
            agent_key_tuple = tuple(sorted(agent_keys))
            if agent_key_tuple in self.target_key_groups and (
                not visible_to_all or not all_sufficiently_far
            ):
                self.target_key_groups.remove(agent_key_tuple)

        if visible_to_all and all_sufficiently_far:
            return tuple(
                [self.available_actions.index("Pickup")] * self.environment.num_agents
            )

        if len(self.target_key_groups) == 0:
            return None

        _dist_cache = {}

        def agent_to_dist_to_target(agent_id, target_key):
            key = (agent_id, target_key)
            if key not in _dist_cache:
                _dist_cache[key] = self.environment.shortest_path_length(
                    source_state_key=agent_keys[agent_id], goal_state_key=target_key
                )
            return _dist_cache[key]

        best_ordered_target_group = None
        best_path_length = None
        for good_target_group in self.target_key_groups:
            for ordered_target_group in itertools.permutations(
                good_target_group, len(good_target_group)
            ):
                path_length = sum(
                    agent_to_dist_to_target(agent_id, target_key)
                    for agent_id, target_key in enumerate(ordered_target_group)
                )
                if best_ordered_target_group is None or best_path_length > path_length:
                    best_path_length = path_length
                    best_ordered_target_group = ordered_target_group

        for agent_id in range(self.environment.num_agents):
            target_key = best_ordered_target_group[agent_id]
            if agent_keys[agent_id] == target_key:
                action = "Pass"
            else:
                action = self.environment.shortest_path_next_action(
                    agent_keys[agent_id], target_key
                )
            expert_actions.append(self.available_actions.index(action))
        return tuple(expert_actions)

    def reached_terminal_state(self) -> bool:
        return self._item_successfully_picked_up

    def coordination_type_tensor(self):
        return coordination_type_tensor(
            self.environment, self.available_actions, self.coordinated_action_checker
        )


class FurnLiftNApartStateLoggingEpisode(FurnLiftNApartStateEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironmentWithGraph,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super().__init__(env=env, task_data=task_data, max_steps=max_steps, **kwargs)
        # Extra logging of step number at which an agent first saw the target
        self._first_view_of_target = [self.max_steps + 1] * self.environment.num_agents
        # Initial agent locations, to calculate the expert path lengths
        self._initial_agent_locations = [
            self.environment.get_agent_location(i)
            for i in range(self.environment.num_agents)
        ]
        self._initial_agent_keys = [
            self.environment.get_key(ag_loc) for ag_loc in self._initial_agent_locations
        ]
        goal_object = self.environment.get_object_by_id(self.object_id, agent_id=0)
        self.object_location = dict(
            x=round(goal_object["position"]["x"], 2),
            z=round(goal_object["position"]["z"], 2),
        )

    @staticmethod
    def location_l2_dist(loc1, loc2):
        return math.sqrt((loc1["x"] - loc2["x"]) ** 2 + (loc1["z"] - loc2["z"]) ** 2)

    def info(self):
        final_agent_distance_from_target = [
            self.location_l2_dist(
                self.object_location, self.environment.get_agent_location(i)
            )
            for i in range(self.environment.num_agents)
        ]

        _dist_cache = {}

        def agent_to_dist_to_target(agent_id, target_key):
            key = (agent_id, target_key)
            if key not in _dist_cache:
                _dist_cache[key] = self.environment.shortest_path_length(
                    source_state_key=self._initial_agent_keys[agent_id],
                    goal_state_key=target_key,
                )
            return _dist_cache[key]

        best_path_lengths = None
        best_path_length = None
        for good_target_group in self.target_key_groups:
            for ordered_target_group in itertools.permutations(
                good_target_group, len(good_target_group)
            ):
                path_lengths = tuple(
                    agent_to_dist_to_target(agent_id, target_key)
                    for agent_id, target_key in enumerate(ordered_target_group)
                )
                path_length = sum(path_lengths)
                if best_path_lengths is None or best_path_length > path_length:
                    best_path_length = best_path_length
                    best_path_lengths = path_lengths

        initial_agent_distance_from_target = [
            self.location_l2_dist(self.object_location, ag_loc)
            for ag_loc in self._initial_agent_locations
        ]
        return {
            **super(JointNavigationEpisode, self).info(),
            "accuracy": 1 * self._item_successfully_picked_up,
            "pickupable_but_not_picked": self._pickupable_but_not_picked,
            "picked_but_not_pickupable": self._picked_but_not_pickupable,
            "picked_but_not_pickupable_distance": self._picked_but_not_pickupable_distance,
            "picked_but_not_pickupable_visibility": self._picked_but_not_pickupable_visibility,
            "reward": self.total_reward,
            "first_view_of_target": self._first_view_of_target,
            "final_distance_between_agents": self.manhattan_dists_between_agents(),
            "best_path_to_target_length": best_path_length,
            "path_to_target_lengths": best_path_lengths,
            "initial_agent_distance_from_target": initial_agent_distance_from_target,
            "final_agent_distance_from_target": final_agent_distance_from_target,
            "object_location": self.object_location,
            "scene_name": self.environment.scene_name,
        }

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        step_results = super(FurnLiftNApartStateLoggingEpisode, self).multi_step(
            actions_as_ints=actions_as_ints
        )

        # Extra logging of step number at which an agent first saw the target
        visibility = self.goal_visibility()
        for i in range(self.environment.num_agents):
            if visibility[i]:
                self._first_view_of_target[i] = min(
                    self._first_view_of_target[i], self.num_steps_taken_in_episode()
                )

        return step_results


class FurnLiftGridStateEpisode(FurnLiftNApartStateEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironmentWithGraph,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super().__init__(env=env, task_data=task_data, max_steps=max_steps, **kwargs)
        self.object_points_set = kwargs["object_points_set"]

    def states_for_agents(self) -> List[Dict[str, Any]]:
        # TODO: Getting the current occupancy matrix like this on every iteration is,
        # TODO: in principle, an expensive operation as it requires doing some thor actions.
        # TODO: We could track state changes without doing this but this requires some annoying bookkeeping.

        # Reachable, unreachable and agent locations are marked
        matrix, point_to_element_map = self.environment.get_current_occupancy_matrix(
            padding=0.5, use_initially_reachable_points_matrix=True
        )
        # Mark goal object locations
        nskipped = 0
        for point_tuple in self.object_points_set:
            if point_tuple not in point_to_element_map:
                nskipped += 1
                continue
            row, col = point_to_element_map[point_tuple]
            matrix[row, col] = constants.GOAL_OBJ_SYM

        if nskipped == len(self.object_points_set):
            raise RuntimeError(
                "Skipped all object points in scene {}.".format(
                    self.environment.scene_name
                )
            )
        elif nskipped > 10:
            warnings.warn(
                "Skipping many object points ({}) in scene {}.".format(
                    nskipped, self.environment.scene_name
                )
            )

        states = []
        for agent_id in range(self.environment.num_agents):
            matrix_ego_per_agent = self.environment.current_matrix_frame(
                agent_id,
                matrix,
                point_to_element_map,
                int(self.environment.visibility_distance // self.environment.grid_size),
                int(self.environment.visibility_distance // self.environment.grid_size),
            )

            last_action = (
                None if self._last_actions is None else self._last_actions[agent_id]
            )
            last_action_success = (
                None
                if self._last_actions_success is None
                else self._last_actions_success[agent_id]
            )
            states.append(
                {
                    "frame": matrix_ego_per_agent,
                    "last_action": last_action,
                    "last_action_success": last_action_success,
                }
            )
        return states
