import itertools
import re
import warnings
from collections import defaultdict
from enum import Enum
from typing import Sequence, Tuple, Callable, Dict, Any, Optional, List, Union

import frozendict
import numpy as np
import scipy.spatial

import constants
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.ai2thor_episodes import MultiAgentAI2ThorEpisode
from rl_ai2thor.ai2thor_gridworld_environment import AI2ThorLiftedObjectGridEnvironment
from utils.misc_util import all_equal

STEP_PENALTY = -0.01
FAILED_ACTION_PENALTY = -0.02
JOINT_MOVE_WITH_OBJECT_KEYWORD = "WithObject"
JOINT_MOVE_OBJECT_KEYWORD = "MoveLifted"
JOINT_ROTATE_KEYWORD = "RotateLifted"
EXPLORATION_BONUS = 0.5
JOINT_PASS_PENALTY = -0.1
CARD_DIR_STRS = ["North", "East", "South", "West"]
EGO_DIR_STRS = ["Ahead", "Right", "Back", "Left"]
ROTATE_OBJECT_ACTIONS = ["RotateLiftedObjectLeft", "RotateLiftedObjectRight"]


def allocentric_action_groups(include_move_obj_actions):
    actions = (
        "MoveNorth",
        "MoveEast",
        "MoveSouth",
        "MoveWest",
        "Pass",
        "MoveAgentsNorthWithObject",
        "MoveAgentsEastWithObject",
        "MoveAgentsSouthWithObject",
        "MoveAgentsWestWithObject",
        "RotateLiftedObjectRight",
    )
    if include_move_obj_actions:
        actions = actions + (
            "MoveLiftedObjectNorth",
            "MoveLiftedObjectEast",
            "MoveLiftedObjectSouth",
            "MoveLiftedObjectWest",
        )
    return (actions,)


def semiallocentric_action_groups(include_move_obj_actions):
    actions = (
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "Pass",
        "MoveAgentsNorthWithObject",
        "MoveAgentsEastWithObject",
        "MoveAgentsSouthWithObject",
        "MoveAgentsWestWithObject",
        "RotateLiftedObjectRight",
    )
    if include_move_obj_actions:
        actions = actions + (
            "MoveLiftedObjectNorth",
            "MoveLiftedObjectEast",
            "MoveLiftedObjectSouth",
            "MoveLiftedObjectWest",
        )
    return (actions,)


def egocentric_action_groups(include_move_obj_actions):
    actions = (
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "Pass",
        "MoveAgentsAheadWithObject",
        "MoveAgentsRightWithObject",
        "MoveAgentsBackWithObject",
        "MoveAgentsLeftWithObject",
        "RotateLiftedObjectRight",
    )
    if include_move_obj_actions:
        actions = actions + (
            "MoveLiftedObjectAhead",
            "MoveLiftedObjectRight",
            "MoveLiftedObjectBack",
            "MoveLiftedObjectLeft",
        )
    return (actions,)


def egocentric_no_rotate_action_groups(include_move_obj_actions):
    actions = (
        "MoveAhead",
        "MoveLeft",
        "MoveRight",
        "MoveBack",
        "Pass",
        "MoveAgentsAheadWithObject",
        "MoveAgentsRightWithObject",
        "MoveAgentsBackWithObject",
        "MoveAgentsLeftWithObject",
        "RotateLiftedObjectRight",
    )
    if include_move_obj_actions:
        actions = actions + (
            "MoveLiftedObjectAhead",
            "MoveLiftedObjectRight",
            "MoveLiftedObjectBack",
            "MoveLiftedObjectLeft",
        )
    return (actions,)


class CoordType(Enum):
    # DO NOT CHANGE THIS WITHOUT CHANGING coordination_type_tensor to match
    # INDIVIDUAL should be the smallest value and equal 0
    INDIVIDUAL = 0

    ROTATE_LIFTED = 1

    MOVE_LIFTED_CARD = 2
    MOVE_LIFTED_EGO = 3

    MOVE_WITH_LIFTED_CARD = 4
    MOVE_WITH_LIFTED_EGO = 5

    PICKUP = 6


ACTION_TO_COORD_TYPE = frozendict.frozendict(
    {
        "Pass": CoordType.INDIVIDUAL,
        #
        **{
            "Move{}".format(dir): CoordType.INDIVIDUAL
            for dir in CARD_DIR_STRS + EGO_DIR_STRS
        },
        **{"Rotate{}".format(dir): CoordType.INDIVIDUAL for dir in ["Left", "Right"]},
        #
        **{
            "RotateLiftedObject{}".format(dir): CoordType.ROTATE_LIFTED
            for dir in ["Left", "Right"]
        },
        #
        **{
            "MoveLiftedObject{}".format(dir): CoordType.MOVE_LIFTED_CARD
            for dir in CARD_DIR_STRS
        },
        #
        **{
            "MoveAgents{}WithObject".format(dir): CoordType.MOVE_WITH_LIFTED_CARD
            for dir in CARD_DIR_STRS
        },
        #
        **{
            "MoveLiftedObject{}".format(dir): CoordType.MOVE_LIFTED_EGO
            for dir in EGO_DIR_STRS
        },
        #
        **{
            "MoveAgents{}WithObject".format(dir): CoordType.MOVE_WITH_LIFTED_EGO
            for dir in EGO_DIR_STRS
        },
        "Pickup": CoordType.PICKUP,
    }
)


def rotate_clockwise(l, n):
    return l[-n:] + l[:-n]


def are_actions_coordinated(env: AI2ThorEnvironment, action_strs: Sequence[str]):
    action_types = [ACTION_TO_COORD_TYPE[a] for a in action_strs]
    if not all_equal(action_types):
        return False

    action_type = action_types[0]

    if action_type == CoordType.INDIVIDUAL:
        return True

    if action_type in [
        CoordType.ROTATE_LIFTED,
        CoordType.MOVE_LIFTED_CARD,
        CoordType.MOVE_WITH_LIFTED_CARD,
        CoordType.PICKUP,
    ]:
        return all_equal(action_strs)
    elif action_type in [CoordType.MOVE_LIFTED_EGO, CoordType.MOVE_WITH_LIFTED_EGO]:
        action_relative_ind = [None] * env.num_agents
        for i, action in enumerate(action_strs):
            for j, dir in enumerate(EGO_DIR_STRS):
                if dir in action:
                    action_relative_ind[i] = j
                    break
            if action_relative_ind[i] is None:
                raise RuntimeError("Ego action but no ego dir in action name?")

        agent_rot_inds = [
            round(env.get_agent_location(agent_id)["rotation"] / 90)
            for agent_id in range(env.num_agents)
        ]

        return all_equal(
            [
                int(dir_rel_ind + agent_rot_ind) % 4
                for dir_rel_ind, agent_rot_ind in zip(
                    action_relative_ind, agent_rot_inds
                )
            ]
        )
    else:
        raise NotImplementedError(
            "Cannot determine if {} actions are coordinated.".format(action_strs)
        )


def are_actions_coordinated_with_pass_conditioning(
    env: AI2ThorEnvironment, action_strs: Sequence[str]
):
    action_types = [ACTION_TO_COORD_TYPE[a] for a in action_strs]
    if not all_equal(action_types):
        return False

    action_type = action_types[0]

    if action_type == CoordType.INDIVIDUAL:
        if "Pass" in action_strs:
            return True
        else:
            return False

    if action_type in [
        CoordType.ROTATE_LIFTED,
        CoordType.MOVE_LIFTED_CARD,
        CoordType.MOVE_WITH_LIFTED_CARD,
        CoordType.PICKUP,
    ]:
        return all_equal(action_strs)
    elif action_type in [CoordType.MOVE_LIFTED_EGO, CoordType.MOVE_WITH_LIFTED_EGO]:
        action_relative_ind = [None] * env.num_agents
        for i, action in enumerate(action_strs):
            for j, dir in enumerate(EGO_DIR_STRS):
                if dir in action:
                    action_relative_ind[i] = j
                    break
            if action_relative_ind[i] is None:
                raise RuntimeError("Ego action but no ego dir in action name?")

        agent_rot_inds = [
            round(env.get_agent_location(agent_id)["rotation"] / 90)
            for agent_id in range(env.num_agents)
        ]

        return all_equal(
            [
                int(dir_rel_ind + agent_rot_ind) % 4
                for dir_rel_ind, agent_rot_ind in zip(
                    action_relative_ind, agent_rot_inds
                )
            ]
        )
    else:
        raise NotImplementedError(
            "Cannot determine if {} actions are coordinated.".format(action_strs)
        )


COORDINATION_TYPE_TENSOR_CACHE = {}


def coordination_type_tensor(
    env,
    action_strings: Tuple[str],
    action_coordination_checker: Callable[[AI2ThorEnvironment, Sequence[str]], bool],
):
    agent_rot_inds = tuple(
        round(env.get_agent_location(i)["rotation"] / 90) % 4
        for i in range(env.num_agents)
    )

    key = (agent_rot_inds, action_strings, action_coordination_checker)
    if key in COORDINATION_TYPE_TENSOR_CACHE:
        return COORDINATION_TYPE_TENSOR_CACHE[key]

    coord_tensor = np.full(
        (len(action_strings),) * env.num_agents, fill_value=-1, dtype=int
    )

    for ind in range(np.product(coord_tensor.shape)):
        multi_ind = np.unravel_index(ind, coord_tensor.shape)
        multi_action = tuple(action_strings[i] for i in multi_ind)

        if action_coordination_checker(env, multi_action):
            coord_tensor[multi_ind] = int(ACTION_TO_COORD_TYPE[multi_action[0]].value)

    COORDINATION_TYPE_TENSOR_CACHE[key] = coord_tensor

    return coord_tensor


def lifted_furniture_step(
    episode, action: str, action_as_int: int, agent_id: int = None, **kwargs
) -> Dict[str, Any]:
    if any(dir in action for dir in CARD_DIR_STRS):
        action_dir = None
        for dir in CARD_DIR_STRS:
            if dir in action:
                action_dir = dir
                break

        joint_actions_nesw_to_local_index = dict(North=0, East=1, South=2, West=3)
        joint_actions_aligned_to_agent = ["Ahead", "Right", "Back", "Left"]

        rotation = episode.environment.get_agent_location(agent_id)["rotation"]
        clock_90 = round(rotation / 90.0)
        joint_actions_aligned_to_agent = rotate_clockwise(
            joint_actions_aligned_to_agent, clock_90
        )

        # joint_actions_aligned_to_agent is Ahead, Right, Back and Left aligned as per
        # agent rotation. Eg. for a 90 degree agent the (relative) joint actions would be:
        # [
        #   "MoveAgentsLeftWithObject",
        #   "MoveAgentsAheadWithObject",
        #   "MoveAgentsRightWithObject",
        #   "MoveAgentsBackWithObject",
        # ]
        # reference:
        #    L
        #    |
        # B ---> A
        #    |
        #    R
        #
        # This converts N,E,S,W (nesw) to 0,1,2,3 and then to L,A,R,B (arbl)
        joint_action_nesw_index = joint_actions_nesw_to_local_index[action_dir]
        assert joint_action_nesw_index >= 0
        joint_action_arbl = joint_actions_aligned_to_agent[joint_action_nesw_index]
        action_dict = {
            "action": action.replace(action_dir, joint_action_arbl),
            "agentId": agent_id,
            **kwargs,
        }
    else:
        action_dict = {"action": action, "agentId": agent_id, **kwargs}
    action_dict["maxAgentsDistance"] = episode.max_distance_from_object
    episode.environment.step(action_dict)
    action_success = episode.environment.last_event.events[agent_id].metadata[
        "lastActionSuccess"
    ]

    episode.environment.last_event.metadata["lastAction"] = action
    if (
        "actionReturn"
        in episode.environment.last_event.events[agent_id].metadata.keys()
    ):
        action_return = episode.environment.last_event.events[agent_id].metadata[
            "actionReturn"
        ]
    else:
        action_return = None
    return {
        "action": action_as_int,
        "action_success": action_success,
        "action_return": action_return,
    }


class MultiAgentMovingWithFurnitureBaseEpisode(MultiAgentAI2ThorEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        frame_type: str = "image",
        grid_output_shape: Optional[Tuple[int, int]] = None,
        first_correct_coord_reward: Optional[float] = None,
        increasing_rotate_penalty: bool = False,
        step_penalty=STEP_PENALTY,
        failed_action_penalty=FAILED_ACTION_PENALTY,
        exploration_bonus=EXPLORATION_BONUS,
        expert_frame_type: Optional[str] = None,
        return_likely_successfuly_move_actions: bool = False,
        return_likely_successfuly_move_actions_for_expert: bool = False,
        pass_conditioned_coordination: bool = False,
        **kwargs,
    ):
        super(MultiAgentMovingWithFurnitureBaseEpisode, self).__init__(
            env=env, task_data=task_data, max_steps=max_steps, **kwargs
        )
        self.frame_type = frame_type
        self.first_correct_coord_reward = first_correct_coord_reward
        self.increasing_rotate_penalty = increasing_rotate_penalty
        self.initial_agent_metadata = env.get_all_agent_metadata()
        self.object_id = task_data["move_obj_id"]
        self.to_object_id = task_data.get("move_to_obj_id")
        self.grid_output_shape = grid_output_shape
        self.visited_xz = set()
        self.visited_xzr = set()
        self._max_distance_from_object = max_distance_from_object
        self.total_reward = 0.0
        self.action_counts = defaultdict(int)
        self.coordinated_actions_taken = set()
        self.agent_num_sequential_rotates = {}

        # For now we assume we're moving a Television
        assert "Television" in self.object_id
        assert self.to_object_id is None or "Dresser" in self.to_object_id

        self.step_penalty = step_penalty
        self.failed_action_penalty = failed_action_penalty
        self.exploration_bonus = exploration_bonus
        self.joint_pass_penalty = JOINT_PASS_PENALTY

        self.expert_frame_type = expert_frame_type
        self.return_likely_successfuly_move_actions = (
            return_likely_successfuly_move_actions
        )
        self.return_likely_successfuly_move_actions_for_expert = (
            return_likely_successfuly_move_actions_for_expert
        )

        self.tv_reachable_positions_tensor = None
        self._tv_reachable_positions_set = None

        self.pass_conditioned_coordination = pass_conditioned_coordination
        self.coordinated_action_checker: Callable[
            [AI2ThorEnvironment, Sequence[str]], bool
        ] = None
        if self.pass_conditioned_coordination:
            self.coordinated_action_checker = (
                are_actions_coordinated_with_pass_conditioning
            )
        else:
            self.coordinated_action_checker = are_actions_coordinated

    @property
    def max_distance_from_object(self):
        return self._max_distance_from_object

    @property
    def tv_reachable_positions_set(self):
        if self._tv_reachable_positions_set is not None:
            return self._tv_reachable_positions_set

        self.environment.step(
            {
                "action": "GetReachablePositionsForObject",
                "objectId": self.object_id,
                "agentId": 0,
            }
        )
        self._tv_reachable_positions_set = set(
            (round(pos["x"], 2), round(pos["z"], 2))
            for pos in self.environment.last_event.metadata["actionReturn"]
        )
        return self._tv_reachable_positions_set

    def _points_set_for_rotation(self, obj_id, obj_points_dict):
        obj = self.environment.get_object_by_id(obj_id, agent_id=0)
        obj_rot = 90 * int(obj["rotation"]["y"] / 90)
        return set(
            (
                round(obj["position"]["x"] + t[0], 2),
                round(obj["position"]["z"] + t[1], 2),
            )
            for t in obj_points_dict[obj_rot]
        )

    def current_object_points_set(self):
        return self._points_set_for_rotation(
            self.object_id, constants.TELEVISION_ROTATION_TO_OCCUPATIONS
        )

    def current_to_object_points_set(self):
        return self._points_set_for_rotation(
            self.to_object_id, constants.TV_STAND_ROTATION_TO_OCCUPATIONS
        )

    def current_distance_between_lifted_and_goal_objects(self):
        move_obj = self.environment.get_object_by_id(self.object_id, agent_id=0)
        move_to_obj = self.environment.get_object_by_id(self.to_object_id, agent_id=0)
        return round(
            abs(move_obj["position"]["x"] - move_to_obj["position"]["x"])
            + abs(move_obj["position"]["z"] - move_to_obj["position"]["z"]),
            2,
        )

    def coordination_type_tensor(self):
        return coordination_type_tensor(
            self.environment, self.available_actions, self.coordinated_action_checker
        )

    def _increment_num_steps_taken_in_episode_by_n(self, n) -> None:
        self._num_steps_taken_in_episode += n

    def info(self):
        def cammel_to_snake(name):
            # See https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        info = {
            **super(MultiAgentMovingWithFurnitureBaseEpisode, self).info(),
            "num_pass": self.action_counts["Pass"],
            "num_rotation": self.action_counts["Rotate"],
            "reward": self.total_reward,
            "percent_points_visited": len(self.visited_xz)
            / len(self.environment.initially_reachable_points)
            * 100.0,
        }

        for k_attempt in list(self.action_counts.keys()):
            if "Attempted" in k_attempt:
                k_base = k_attempt.replace("Attempted", "")
                k_success = k_attempt.replace("Attempted", "Successful")
                n_attempt = self.action_counts[k_attempt]
                n_success = self.action_counts[k_success]

                info[cammel_to_snake(k_base) + "/attempted/count"] = n_attempt
                if n_attempt > 0:
                    info[cammel_to_snake(k_base) + "/successful/percent"] = 100 * (
                        n_success / n_attempt
                    )

        return info

    def add_likely_successful_move_actions(self, states, frame_type):
        assert frame_type == "allocentric-tensor-centered-at-tv"
        tensor = states[0]["frame"]
        agent_reachable = tensor[0]
        tv_reachable = tensor[-2]

        a0_pos_mat = tensor[1:5].sum(0) != 0
        a1_pos_mat = tensor[5:9].sum(0) != 0
        tv_pos_mat = tensor[-3] != 0

        try:
            a0row, a0col = tuple(zip(*np.where(a0_pos_mat)))[0]
            a1row, a1col = tuple(zip(*np.where(a1_pos_mat)))[0]
            all_tv_pos = np.stack(np.where(tv_pos_mat), axis=1)
            tv_row, tv_col = np.array(np.median(all_tv_pos, axis=0), dtype=int)

            would_succeed = []

            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

            # Move with tv
            for m in moves:
                r_off, c_off = m
                try:
                    would_succeed.append(
                        agent_reachable[a0row + r_off, a0col + c_off]
                        and agent_reachable[a1row + r_off, a1col + c_off]
                        and tv_reachable[tv_row + r_off, tv_col + c_off]
                    )
                except IndexError as _:
                    would_succeed.append(False)

            # Move tv alone
            for m in moves:
                r_off, c_off = m
                shifted_tv_pos = all_tv_pos + np.array((m,))

                would_succeed.append(
                    tv_reachable[a0row + r_off, a0col + c_off]
                    and (
                        not np.any(
                            np.abs(shifted_tv_pos - np.array(((a0row, a0col),))).sum(1)
                            == 0
                        )
                    )
                    and (
                        not np.any(
                            np.abs(shifted_tv_pos - np.array(((a1row, a1col),))).sum(1)
                            == 0
                        )
                    )
                )
            would_succeed = [bool(w) for w in would_succeed]
            for s in states:
                s["would_coordinated_action_succeed"] = [bool(w) for w in would_succeed]
        except IndexError as _:
            for s in states:
                s["would_coordinated_action_succeed"] = [False] * 4

            warnings.warn(
                "\nCould not compute whether coordinated actions would succeed.\nIn scene {}, agent positions {} and {}. Current tensor: {}".format(
                    self.environment.scene_name,
                    self.environment.get_agent_location(0),
                    self.environment.get_agent_location(1),
                    tensor,
                )
            )

    def states_for_expert_agents(self):
        states = self.states_for_agents(self.expert_frame_type)
        if self.return_likely_successfuly_move_actions_for_expert:
            if "would_coordinated_action_succeed" not in states[0]:
                self.add_likely_successful_move_actions(
                    states=states, frame_type=self.expert_frame_type
                )
        return states

    def _step(
        self, action_as_int: int, agent_id: int = None, **kwargs
    ) -> Dict[str, Any]:
        return lifted_furniture_step(
            episode=self,
            action=self.available_actions[action_as_int],
            action_as_int=action_as_int,
            agent_id=agent_id,
            **kwargs,
        )

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        assert not self.is_paused() and not self.is_complete()
        self._increment_num_steps_taken_in_episode_by_n(self.environment.num_agents)

        available_actions = self.available_actions
        # if all actions_as_ints are valid for this episode's available_actions
        assert all(
            [action_id < len(available_actions) for action_id in actions_as_ints]
        )
        actions_as_strings = tuple(
            [available_actions[action_id] for action_id in actions_as_ints]
        )

        for agent_id, action in enumerate(actions_as_strings):
            if agent_id not in self.agent_num_sequential_rotates:
                self.agent_num_sequential_rotates[agent_id] = 0
            if [action == ra for ra in ["RotateRight", "RotateLeft"]]:
                self.agent_num_sequential_rotates[agent_id] += 1
            elif action.lower() != "pass":
                self.agent_num_sequential_rotates[agent_id] = 0

        before_object_metadata = self.environment.get_object_by_id(
            object_id=self.object_id, agent_id=0
        )
        before_object_location = {
            "x": before_object_metadata["position"]["x"],
            "y": before_object_metadata["position"]["y"],
            "z": before_object_metadata["position"]["z"],
            "rotation": before_object_metadata["rotation"]["y"],
        }

        step_results = []
        before_info = (
            None
            if self.before_step_function is None
            else self.before_step_function(episode=self)
        )

        for i, action in enumerate(actions_as_strings):
            step_results.append(
                {
                    "action": actions_as_ints[i],
                    "action_as_string": action,
                    "reward": self.step_penalty,
                }
            )

        # If an action is for movement with object
        is_joint_move_with = tuple(
            [
                JOINT_MOVE_WITH_OBJECT_KEYWORD.lower() in action.lower()
                for action in actions_as_strings
            ]
        )

        # If an action is for moving the object
        is_joint_move_object = tuple(
            [
                JOINT_MOVE_OBJECT_KEYWORD.lower() in action.lower()
                for action in actions_as_strings
            ]
        )

        # If joint rotate action
        is_joint_rotate = tuple(
            [
                JOINT_ROTATE_KEYWORD.lower() in action.lower()
                for action in actions_as_strings
            ]
        )

        # Correct coordination?
        correct_coordination = self.coordinated_action_checker(
            self.environment, actions_as_strings
        )

        if correct_coordination and (
            all(is_joint_rotate) or all(is_joint_move_object) or all(is_joint_move_with)
        ):
            self.coordinated_multi_step(
                actions_as_ints=actions_as_ints,
                actions_as_strings=actions_as_strings,
                available_actions=available_actions,
                step_results=step_results,
            )
        else:
            if not self.pass_conditioned_coordination:
                # Handles the old coordination setting
                self.uncoordinated_multi_step(
                    actions_as_strings=actions_as_strings,
                    available_actions=available_actions,
                    is_joint_action=list(
                        any(z)
                        for z in zip(
                            is_joint_move_with, is_joint_move_object, is_joint_rotate
                        )
                    ),
                    step_results=step_results,
                )
            elif "Pass" in actions_as_strings:
                # (Pass, X) can always go through the uncoordinated_multi_step
                # If X is joint, it would lead to failed_penalty error
                self.uncoordinated_multi_step(
                    actions_as_strings=actions_as_strings,
                    available_actions=available_actions,
                    is_joint_action=list(
                        any(z)
                        for z in zip(
                            is_joint_move_with, is_joint_move_object, is_joint_rotate
                        )
                    ),
                    step_results=step_results,
                )
            else:
                # (Non pass, Non pass) actions which aren't coordindated (joint, joint)
                for agent_id, action in enumerate(actions_as_strings):
                    step_results[agent_id]["reward"] += self.failed_action_penalty
                    step_results[agent_id]["action_success"] = False

        self.total_reward += sum(sr["reward"] for sr in step_results)
        after_object_metadata = self.environment.get_object_by_id(
            object_id=self.object_id, agent_id=0
        )
        after_object_location = {
            "x": after_object_metadata["position"]["x"],
            "y": after_object_metadata["position"]["y"],
            "z": after_object_metadata["position"]["z"],
            "rotation": after_object_metadata["rotation"]["y"],
        }
        object_location = {
            self.object_id: {
                "before_location": before_object_location,
                "after_location": after_object_location,
            }
        }
        for sr in step_results:
            sr["object_location"] = object_location

        if self.after_step_function is not None:
            self.after_step_function(
                step_results=step_results, before_info=before_info, episode=self
            )
        return step_results

    # @profile
    def coordinated_multi_step(
        self, actions_as_ints, actions_as_strings, available_actions, step_results
    ):
        if self.first_correct_coord_reward is not None:
            assert type(actions_as_ints) == tuple
            if actions_as_ints not in self.coordinated_actions_taken:
                self.coordinated_actions_taken.add(actions_as_ints)
                for sr in step_results:
                    sr["reward"] += self.first_correct_coord_reward

        found_substring = False
        action_count_string = None
        for substr in CARD_DIR_STRS + EGO_DIR_STRS:
            if substr in actions_as_strings[0]:
                found_substring = True
                action_count_string = (
                    actions_as_strings[0].replace(substr, "") + "Attempted"
                )
                break
        if not found_substring:
            raise Exception("Could not construct action_count_string")
        self.action_counts[action_count_string] += self.environment.num_agents

        step_result = self._step(
            action_as_int=available_actions.index(actions_as_strings[0]),
            agent_id=0,
            objectId=self.object_id,
            maxAgentsDistance=self._max_distance_from_object,
        )
        action_success = step_result["action_success"]
        self.action_counts[action_count_string.replace("Attempted", "Successful")] += (
            action_success * self.environment.num_agents
        )

        object = self.environment.get_object_by_id(self.object_id, agent_id=0)
        object_position = object["position"]
        object_pos_rot_tuple = (
            round(object_position["x"], 2),
            round(object_position["z"], 2),
            round(object["rotation"]["y"] / 90) % 2,
        )
        object_pos_tuple = object_pos_rot_tuple[:2]

        additional_reward = self.failed_action_penalty * (1 - action_success)
        is_new_object_rotation = False
        if (
            object_pos_tuple in self.visited_xz
            and object_pos_rot_tuple not in self.visited_xzr
        ):
            is_new_object_rotation = True
            additional_reward += 0.5 * self.exploration_bonus

        is_new_object_position = False
        if object_pos_tuple not in self.visited_xz:
            is_new_object_position = True
            additional_reward += self.exploration_bonus

        self.visited_xz.add(object_pos_tuple)
        self.visited_xzr.add(object_pos_rot_tuple)

        for srs in step_results:
            srs["successfully_coordinated"] = True
            srs["object_pos_rot_tuple"] = object_pos_rot_tuple
            srs["is_new_object_position"] = is_new_object_position
            srs["is_new_object_rotation"] = is_new_object_rotation
            srs["action_success"] = action_success
            srs["reward"] += additional_reward

        return step_results

    # @profile
    def uncoordinated_multi_step(
        self, actions_as_strings, available_actions, is_joint_action, step_results
    ):
        for agent_id, action in enumerate(actions_as_strings):
            additional_reward = 0.0
            action_success = False
            if is_joint_action[agent_id]:
                additional_reward += self.failed_action_penalty
            elif action == "Pass":
                additional_reward += 0.0
                action_success = True
                self.action_counts["Pass"] += 1
                self._step(
                    action_as_int=available_actions.index(action), agent_id=agent_id
                )
            elif action == "RotateLeft" or action == "RotateRight":
                if self.increasing_rotate_penalty:
                    additional_reward += max(
                        -1,
                        self.step_penalty
                        * (
                            2 ** ((self.agent_num_sequential_rotates[agent_id] // 4))
                            - 1
                        ),
                    )

                action_success = True
                self.action_counts["Rotate"] += 1
                self._step(
                    action_as_int=available_actions.index(action), agent_id=agent_id
                )
            elif "Move" in action:
                # Based on the order of conditions, this will be executed for single
                # agent move actions, bucketed under MoveAheadAttempted/Successful
                # Could be:
                # MoveAhead, MoveRight, MoveBack, MoveLeft or
                # MoveNorth, MoveEast, MoveSouth, MoveWest
                self.action_counts["MoveAheadAttempted"] += 1
                sr = self._step(
                    action_as_int=available_actions.index(action),
                    agent_id=agent_id,
                    maxAgentsDistance=self._max_distance_from_object,
                    objectId=self.object_id,
                )
                action_success = sr["action_success"]
                self.action_counts["MoveAheadSuccessful"] += 1 * action_success
                additional_reward += self.failed_action_penalty * (1 - action_success)
            else:
                raise Exception(
                    "Something wrong with conditions for action {}".format(action)
                )
            step_results[agent_id]["reward"] += additional_reward
            step_results[agent_id]["action_success"] = action_success
        return step_results

    def expert_states_for_agents(self):
        return self.states_for_agents(frame_type=self.expert_frame_type)

    def states_for_agents(self, frame_type: Optional[str] = None):
        frames = []
        if frame_type is None:
            frame_type = self.frame_type
        if frame_type == "image":
            return super(
                MultiAgentMovingWithFurnitureBaseEpisode, self
            ).states_for_agents()
        elif frame_type == "grid-matrix":
            # Reachable, unreachable and agent locations are marked
            (
                matrix_all_agents,
                point_to_element_map,
            ) = self.environment.get_current_occupancy_matrices_two_agents(
                padding=0.5, use_initially_reachable_points_matrix=True
            )

            # TODO: Edit to have generic n-agent version of this
            # Configured for two agents only
            assert self.environment.num_agents == 2
            # Mark visited locations
            nskipped = 0
            for point_tuple in self.visited_xz:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                # Shouldn't overwrite agent embedding information or goal object information
                if not (
                    matrix_all_agents[0][row, col]
                    in [
                        constants.REACHABLE_SYM,
                        constants.UNREACHABLE_SYM,
                        constants.NO_INFO_SYM,
                    ]
                    and matrix_all_agents[1][row, col]
                    in [
                        constants.REACHABLE_SYM,
                        constants.UNREACHABLE_SYM,
                        constants.NO_INFO_SYM,
                    ]
                ):
                    continue
                matrix_all_agents[0][row, col] = constants.VISITED_SYM
                matrix_all_agents[1][row, col] = constants.VISITED_SYM
            if nskipped > 10:
                warnings.warn(
                    "Skipping many object's visited points ({}) in scene {}.".format(
                        nskipped, self.environment.scene_name
                    )
                )

            # Mark goal object locations
            nskipped = 0
            object_points_set = self.current_object_points_set()
            agent_position_consts = {
                constants.AGENT_OTHER_0,
                constants.AGENT_OTHER_90,
                constants.AGENT_OTHER_180,
                constants.AGENT_OTHER_270,
                constants.AGENT_SELF_0,
                constants.AGENT_SELF_90,
                constants.AGENT_SELF_180,
                constants.AGENT_SELF_270,
            }
            for point_tuple in object_points_set:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                if matrix_all_agents[0][row, col] not in agent_position_consts:
                    matrix_all_agents[0][row, col] = constants.GOAL_OBJ_SYM
                if matrix_all_agents[1][row, col] not in agent_position_consts:
                    matrix_all_agents[1][row, col] = constants.GOAL_OBJ_SYM

            if nskipped == len(object_points_set):
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

            if self.to_object_id is not None:
                raise NotImplementedError(
                    "to_object_id must be None when using frame_tupe=='grid-matrix'"
                )

            for agent_id in range(self.environment.num_agents):
                matrix_allo_per_agent = self.environment.current_allocentric_matrix_frame(
                    agent_id, matrix_all_agents[agent_id], point_to_element_map, 10
                )

                frames.append(matrix_allo_per_agent)
        elif frame_type in [
            "allocentric-tensor",
            "allocentric-tensor-centered-at-tv",
            "egocentric-tensor",
            "allocentric-tensor-no-rotations",
        ]:
            if frame_type in [
                "allocentric-tensor",
                "allocentric-tensor-centered-at-tv",
                "egocentric-tensor",
            ]:
                # Reachable, unreachable and agent locations are marked
                (
                    state_tensor_per_agent,
                    point_to_element_map,
                ) = self.environment.get_current_multi_agent_occupancy_tensors(
                    padding=1.0, use_initially_reachable_points_matrix=True
                )
            elif frame_type in ["allocentric-tensor-no-rotations"]:
                (
                    state_tensor_per_agent,
                    point_to_element_map,
                ) = self.environment.get_current_multi_agent_occupancy_tensors_no_rot(
                    padding=1.0, use_initially_reachable_points_matrix=True
                )
            else:
                raise Exception("Check conditions!")
            nrow_ncol = state_tensor_per_agent[0].shape[-2:]

            # TODO: Edit to have generic n-agent version of this
            # Configured for two agents only
            assert self.environment.num_agents == 2
            # Mark visited locations
            nskipped = 0
            visited_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
            for point_tuple in self.visited_xz:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                visited_tensor[0, row, col] = True
            if nskipped > 10:
                warnings.warn(
                    "Skipping many object's visited points ({}) in scene {}.".format(
                        nskipped, self.environment.scene_name
                    )
                )

            # Mark goal object locations
            nskipped = 0
            object_points_set = self.current_object_points_set()
            object_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
            for point_tuple in object_points_set:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                object_tensor[0, row, col] = True

            if nskipped == len(object_points_set):
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

            if self.tv_reachable_positions_tensor is None:
                self.tv_reachable_positions_tensor = np.zeros(
                    (1, *nrow_ncol), dtype=bool
                )
                nskipped = 0
                for point_tuple in self.tv_reachable_positions_set:
                    if point_tuple not in point_to_element_map:
                        nskipped += 1
                        continue
                    row, col = point_to_element_map[point_tuple]
                    self.tv_reachable_positions_tensor[0, row, col] = True

            to_object_tensor_tuple = tuple()
            if self.to_object_id is not None:
                nskipped = 0
                to_object_points_set = self.current_to_object_points_set()
                to_object_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
                for point_tuple in to_object_points_set:
                    if point_tuple not in point_to_element_map:
                        nskipped += 1
                        continue
                    row, col = point_to_element_map[point_tuple]
                    to_object_tensor[0, row, col] = True

                if nskipped == len(to_object_points_set):
                    raise RuntimeError(
                        "Skipped all to_object points in scene {}.".format(
                            self.environment.scene_name
                        )
                    )
                elif nskipped > 10:
                    warnings.warn(
                        "Skipping many to_object points ({}) in scene {}.".format(
                            nskipped, self.environment.scene_name
                        )
                    )
                to_object_tensor_tuple = (to_object_tensor,)

            if frame_type == "egocentric-tensor":
                reachable = state_tensor_per_agent[0][:1]
                agent_rot_inds = []
                other_agent_position_tensors = []
                for agent_id, state in enumerate(state_tensor_per_agent):
                    agent_rot_inds.append(
                        round(
                            self.environment.get_agent_location(agent_id)["rotation"]
                            / 90
                        )
                        % 4
                    )

                    other_agent_position_tensor = state[5:9]

                    if agent_rot_inds[-1] == 0:
                        order = [0, 1, 2, 3]
                    elif agent_rot_inds[-1] == 1:
                        order = [1, 2, 3, 0]
                    elif agent_rot_inds[-1] == 2:
                        order = [2, 3, 0, 1]
                    elif agent_rot_inds[-1] == 3:
                        order = [3, 0, 1, 2]
                    else:
                        raise NotImplementedError()

                    other_agent_position_tensors.append(
                        np.stack(
                            [other_agent_position_tensor[i] for i in order], axis=0
                        )
                    )

                for agent_id in range(self.environment.num_agents):
                    visibility_mask = np.zeros((1, *nrow_ncol), dtype=bool)
                    assert (
                        len(
                            self.environment.last_event.events[agent_id].metadata[
                                "visibleRange"
                            ]
                        )
                        == 26
                    )
                    visible_tuples = [
                        (p["x"], p["z"])
                        for p in self.environment.last_event.events[agent_id].metadata[
                            "visibleRange"
                        ]
                    ]
                    visible_hull = scipy.spatial.Delaunay(np.array(visible_tuples))
                    for point_tuple in point_to_element_map:
                        if visible_hull.find_simplex(point_tuple) >= 0:
                            row, col = point_to_element_map[point_tuple]
                            visibility_mask[0, row, col] = True

                    tensor = np.concatenate(
                        (
                            self.tv_reachable_positions_tensor,
                            reachable,
                            visited_tensor,
                            visibility_mask,
                            other_agent_position_tensors[agent_id],
                            object_tensor,
                        )
                        + to_object_tensor_tuple,
                        axis=0,
                    )

                    tensor *= visibility_mask

                    outsize = 15
                    padding = outsize - 1

                    assert outsize % 2 == 1
                    tensor = np.pad(
                        tensor,
                        [(0, 0), (padding, padding), (padding, padding)],
                        "constant",
                        constant_values=False,
                    )

                    agent_pos = self.environment.get_agent_location(agent_id=agent_id)
                    agent_x = round(agent_pos["x"], 2)
                    agent_z = round(agent_pos["z"], 2)
                    pos_tuple = (agent_x, agent_z)
                    row, col = point_to_element_map[pos_tuple]
                    row = row + padding
                    col = col + padding

                    half_pad = padding // 2
                    if agent_rot_inds[agent_id] == 0:
                        egocentric_tensor = tensor[
                            :,
                            (row - padding) : (row + 1),
                            (col - half_pad) : (col + half_pad + 1),
                        ]
                    elif agent_rot_inds[agent_id] == 1:
                        egocentric_tensor = tensor[
                            :,
                            (row - half_pad) : (row + half_pad + 1),
                            col : (col + padding + 1),
                        ]
                        egocentric_tensor = np.rot90(egocentric_tensor, axes=(1, 2))
                    elif agent_rot_inds[agent_id] == 2:
                        egocentric_tensor = tensor[
                            :,
                            row : (row + padding + 1),
                            (col - half_pad) : (col + half_pad + 1),
                        ]
                        egocentric_tensor = np.rot90(
                            egocentric_tensor, k=2, axes=(1, 2)
                        )
                    elif agent_rot_inds[agent_id] == 3:
                        egocentric_tensor = tensor[
                            :,
                            (row - half_pad) : (row + half_pad + 1),
                            (col - padding) : (col + 1),
                        ]
                        egocentric_tensor = np.rot90(
                            egocentric_tensor, k=3, axes=(1, 2)
                        )
                    else:
                        raise NotImplementedError()

                    frames.append(np.array(egocentric_tensor, dtype=float))
            else:
                state_tensor_per_agent = [
                    np.concatenate(
                        (
                            state,
                            visited_tensor,
                            object_tensor,
                            self.tv_reachable_positions_tensor,
                        )
                        + to_object_tensor_tuple,
                        axis=0,
                    )
                    for state in state_tensor_per_agent
                ]
                if frame_type in [
                    "allocentric-tensor",
                    "allocentric-tensor-no-rotations",
                ]:
                    for agent_id in range(self.environment.num_agents):
                        agent_pos = self.environment.get_agent_location(
                            agent_id=agent_id
                        )
                        agent_x = round(agent_pos["x"], 2)
                        agent_z = round(agent_pos["z"], 2)
                        pos_tuple = (agent_x, agent_z)
                        row, col = point_to_element_map[pos_tuple]
                        tensor = np.pad(
                            state_tensor_per_agent[agent_id],
                            [(0, 0), (10, 10), (10, 10)],
                            "constant",
                            constant_values=False,
                        )
                        allocentric_tensor = tensor[
                            :, row : (row + 21), col : (col + 21)
                        ]
                        frames.append(np.array(allocentric_tensor, dtype=float))
                elif frame_type == "allocentric-tensor-centered-at-tv":
                    object_metadata = self.environment.get_object_by_id(
                        object_id=self.object_id, agent_id=0
                    )
                    object_location = {
                        "x": object_metadata["position"]["x"],
                        "y": object_metadata["position"]["y"],
                        "z": object_metadata["position"]["z"],
                        "rotation": object_metadata["rotation"]["y"],
                    }
                    for agent_id in range(self.environment.num_agents):
                        object_x = round(object_location["x"], 2)
                        object_z = round(object_location["z"], 2)
                        pos_tuple = (object_x, object_z)

                        row, col = point_to_element_map[pos_tuple]

                        tensor = np.pad(
                            state_tensor_per_agent[agent_id],
                            [(0, 0), (10, 10), (10, 10)],
                            "constant",
                            constant_values=False,
                        )
                        allocentric_tensor = tensor[
                            :, row : (row + 21), col : (col + 21)
                        ]
                        frames.append(np.array(allocentric_tensor, dtype=float))
                else:
                    raise Exception("something wrong with conditions")
        else:
            raise Exception("Invalid frame type {}.".format(frame_type))

        states = []
        for agent_id in range(self.environment.num_agents):
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
                    "last_action": last_action,
                    "last_action_success": last_action_success,
                    "frame": frames[agent_id],
                }
            )

        if (
            frame_type == self.frame_type
            and self.return_likely_successfuly_move_actions
        ) or (
            frame_type == self.expert_frame_type
            and self.return_likely_successfuly_move_actions_for_expert
        ):
            self.add_likely_successful_move_actions(
                states=states, frame_type=frame_type
            )

        return states


class FurnMoveEpisode(MultiAgentMovingWithFurnitureBaseEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        min_dist_to_to_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            **kwargs,
        )
        self.include_move_obj_actions = include_move_obj_actions
        self.reached_target_reward = reached_target_reward
        self.moved_closer_reward = moved_closer_reward
        self.min_dist_to_to_object = min_dist_to_to_object

        self.closest_to_object_achieved = (
            self.current_distance_between_lifted_and_goal_objects()
        )
        self.initial_move_obj = self.environment.get_object_by_id(
            self.object_id, agent_id=0
        )
        self.initial_move_to_obj = self.environment.get_object_by_id(
            self.to_object_id, agent_id=0
        )

    @property
    def available_action_groups(self) -> Tuple[Tuple[str, ...], ...]:
        return self.class_available_action_groups(
            include_move_obj_actions=self.include_move_obj_actions
        )

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        raise NotImplementedError

    def info(self):
        info = super(FurnMoveEpisode, self).info()
        info["navigation/reached_target"] = self.reached_terminal_state()
        info[
            "navigation/final_distance"
        ] = self.current_distance_between_lifted_and_goal_objects()
        initial_manhattan_distance = round(
            abs(
                self.initial_move_obj["position"]["x"]
                - self.initial_move_to_obj["position"]["x"]
            )
            + abs(
                self.initial_move_obj["position"]["z"]
                - self.initial_move_to_obj["position"]["z"]
            ),
            2,
        )
        initial_manhattan_steps = round(
            initial_manhattan_distance / self.environment.grid_size
        )
        path_length = self.num_steps_taken_in_episode() / self.environment.num_agents
        info["navigation/spl_manhattan"] = info["navigation/reached_target"] * (
            (initial_manhattan_steps + 0.0001)
            / (max(initial_manhattan_steps, path_length) + 0.0001)
        )
        info["navigation/initial_manhattan_steps"] = initial_manhattan_steps
        return info

    def reached_terminal_state(self) -> bool:
        return self.closest_to_object_achieved < self.min_dist_to_to_object

    def next_expert_action(self) -> Tuple[Union[int, None], ...]:
        # No expert supervision for exploring the environment.
        expert_actions = [None] * self.environment.num_agents
        return tuple(expert_actions)

    def coordinated_multi_step(
        self, actions_as_ints, actions_as_strings, available_actions, step_results
    ):
        super(FurnMoveEpisode, self).coordinated_multi_step(
            actions_as_ints=actions_as_ints,
            actions_as_strings=actions_as_strings,
            available_actions=available_actions,
            step_results=step_results,
        )

        dist = self.current_distance_between_lifted_and_goal_objects()
        additional_reward = 0.0

        if self.reached_terminal_state():
            additional_reward += self.reached_target_reward
        elif dist < self.closest_to_object_achieved:
            additional_reward += self.moved_closer_reward

        self.closest_to_object_achieved = min(dist, self.closest_to_object_achieved)

        for sr in step_results:
            sr["reward"] += additional_reward

        return step_results


class FurnMoveAllocentricEpisode(FurnMoveEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        assert kwargs["frame_type"] == "allocentric-tensor"
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            include_move_obj_actions=include_move_obj_actions,
            reached_target_reward=reached_target_reward,
            moved_closer_reward=moved_closer_reward,
            **kwargs,
        )

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        return semiallocentric_action_groups(
            include_move_obj_actions=include_move_obj_actions
        )


class FurnMoveEgocentricEpisode(FurnMoveEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        assert "frame_type" in kwargs
        assert kwargs["frame_type"] in ["egocentric-tensor", "image"]
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            include_move_obj_actions=include_move_obj_actions,
            reached_target_reward=reached_target_reward,
            moved_closer_reward=moved_closer_reward,
            **kwargs,
        )
        if "track_time_averaged_action_pairs" in kwargs:
            self.track_time_averaged_action_pairs = kwargs[
                "track_time_averaged_action_pairs"
            ]
            num_actions = len(self.available_actions)
            self.action_pairs_matrix_attempted = np.zeros(
                (num_actions, num_actions), dtype=float
            )
            self.action_pairs_matrix_successful = np.zeros(
                (num_actions, num_actions), dtype=float
            )
        else:
            self.track_time_averaged_action_pairs = False

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        return egocentric_action_groups(
            include_move_obj_actions=include_move_obj_actions
        )

    def multi_step(self, actions_as_ints: Tuple[int, ...]):
        step_results = super(FurnMoveEgocentricEpisode, self).multi_step(
            actions_as_ints
        )
        if self.track_time_averaged_action_pairs:
            assert self.environment.num_agents == 2
            self.action_pairs_matrix_attempted[
                actions_as_ints[0], actions_as_ints[1]
            ] += 1
            self.action_pairs_matrix_successful[
                actions_as_ints[0], actions_as_ints[1]
            ] += int(all([sr["action_success"] for sr in step_results]))
        return step_results

    def info(self):
        info = super(FurnMoveEgocentricEpisode, self).info()
        if self.track_time_averaged_action_pairs:
            total_actions = np.sum(self.action_pairs_matrix_attempted)
            info[
                "action_pairs_matrix_attempted"
            ] = self.action_pairs_matrix_attempted / (total_actions + 0.0001)
            info[
                "action_pairs_matrix_successful"
            ] = self.action_pairs_matrix_successful / (total_actions + 0.0001)
        return info


class FurnMoveEgocentricFastGridEpisode(FurnMoveEpisode):
    def __init__(
        self,
        env: AI2ThorLiftedObjectGridEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        assert kwargs["frame_type"] in [
            "fast-egocentric-tensor",
            "fast-egocentric-relative-tensor",
        ]
        if "visualize_test_gridworld" in kwargs:
            self.visualize_test_gridworld = kwargs["visualize_test_gridworld"]
            assert "visualizing_ms" in kwargs
            self.visualizing_ms = kwargs["visualizing_ms"]

        else:
            self.visualize_test_gridworld = False
            self.visualizing_ms = None
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            include_move_obj_actions=include_move_obj_actions,
            reached_target_reward=reached_target_reward,
            moved_closer_reward=moved_closer_reward,
            **kwargs,
        )
        if "track_time_averaged_action_pairs" in kwargs:
            self.track_time_averaged_action_pairs = kwargs[
                "track_time_averaged_action_pairs"
            ]
            num_actions = len(self.available_actions)
            self.action_pairs_matrix_attempted = np.zeros(
                (num_actions, num_actions), dtype=float
            )
            self.action_pairs_matrix_successful = np.zeros(
                (num_actions, num_actions), dtype=float
            )
        else:
            self.track_time_averaged_action_pairs = False

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        return egocentric_action_groups(
            include_move_obj_actions=include_move_obj_actions
        )

    @property
    def tv_reachable_positions_set(self):
        if self._tv_reachable_positions_set is not None:
            return self._tv_reachable_positions_set

        self.environment.step(
            {
                "action": "GetReachablePositionsForObject",
                "objectId": self.object_id,
                "agentId": 0,
            }
        )
        self._tv_reachable_positions_set = set(
            (round(pos["x"], 2), round(pos["z"], 2))
            for pos in itertools.chain.from_iterable(
                self.environment.last_event.metadata["actionReturn"].values()
            )
        )
        return self._tv_reachable_positions_set

    def states_for_agents(self, frame_type: Optional[str] = None):
        frames = []
        if frame_type is None:
            frame_type = self.frame_type
        if frame_type in ["fast-egocentric-tensor", "fast-egocentric-relative-tensor"]:
            # Reachable, unreachable and agent locations are marked
            # Since it's the same cost, may as well set
            # use_initially_reachable_points_matrix to False
            (
                state_tensor_per_agent,
                point_to_element_map,
            ) = self.environment.get_current_multi_agent_occupancy_tensors(
                use_initially_reachable_points_matrix=False
            )
            nrow_ncol = state_tensor_per_agent[0].shape[-2:]

            # Mark visited locations
            nskipped = 0
            visited_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
            for point_tuple in self.visited_xz:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                visited_tensor[0, row, col] = True
            if nskipped > 10:
                warnings.warn(
                    "Skipping many object's visited points ({}) in scene {}.".format(
                        nskipped, self.environment.scene_name
                    )
                )

            # Mark goal object locations
            nskipped = 0
            object_points_set = self.current_object_points_set()
            object_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
            for point_tuple in object_points_set:
                if point_tuple not in point_to_element_map:
                    nskipped += 1
                    continue
                row, col = point_to_element_map[point_tuple]
                object_tensor[0, row, col] = True

            if nskipped == len(object_points_set):
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

            # TODO: This can be replaced with rotation_to_lifted_object_reachable_position_masks
            if self.tv_reachable_positions_tensor is None:
                self.tv_reachable_positions_tensor = np.zeros(
                    (1, *nrow_ncol), dtype=bool
                )
                nskipped = 0
                for point_tuple in self.tv_reachable_positions_set:
                    if point_tuple not in point_to_element_map:
                        nskipped += 1
                        continue
                    row, col = point_to_element_map[point_tuple]
                    self.tv_reachable_positions_tensor[0, row, col] = True

            to_object_tensor_tuple = tuple()
            if self.to_object_id is not None:
                nskipped = 0
                to_object_points_set = self.current_to_object_points_set()
                to_object_tensor = np.zeros((1, *nrow_ncol), dtype=bool)
                for point_tuple in to_object_points_set:
                    if point_tuple not in point_to_element_map:
                        nskipped += 1
                        continue
                    row, col = point_to_element_map[point_tuple]
                    to_object_tensor[0, row, col] = True

                if nskipped == len(to_object_points_set):
                    raise RuntimeError(
                        "Skipped all to_object points in scene {}.".format(
                            self.environment.scene_name
                        )
                    )
                elif nskipped > 10:
                    warnings.warn(
                        "Skipping many to_object points ({}) in scene {}.".format(
                            nskipped, self.environment.scene_name
                        )
                    )
                to_object_tensor_tuple = (to_object_tensor,)

            if frame_type == "fast-egocentric-tensor":
                output_tensor_per_agent = [
                    np.concatenate(
                        (
                            state,
                            visited_tensor,
                            object_tensor,
                            self.tv_reachable_positions_tensor,
                        )
                        + to_object_tensor_tuple,
                        axis=0,
                    )
                    for state in state_tensor_per_agent
                ]
            elif frame_type == "fast-egocentric-relative-tensor":
                reachable = state_tensor_per_agent[0][:1]
                agent_rot_inds = []
                other_agent_position_tensors_list = []
                for agent_id, state in enumerate(state_tensor_per_agent):
                    agent_rot_inds.append(
                        round(
                            self.environment.get_agent_location(agent_id)["rotation"]
                            / 90
                        )
                        % 4
                    )

                    other_agent_positions = state[5:]

                    if agent_rot_inds[-1] == 0:
                        order = [0, 1, 2, 3]
                    elif agent_rot_inds[-1] == 1:
                        order = [1, 2, 3, 0]
                    elif agent_rot_inds[-1] == 2:
                        order = [2, 3, 0, 1]
                    elif agent_rot_inds[-1] == 3:
                        order = [3, 0, 1, 2]
                    else:
                        raise NotImplementedError()

                    other_agent_position_tensors_list.append(
                        other_agent_positions[
                            sum(
                                [
                                    [o + 4 * i for o in order]
                                    for i in range(self.environment.num_agents - 1)
                                ],
                                [],
                            )
                        ]
                    )

                output_tensor_per_agent = [
                    np.concatenate(
                        (
                            self.tv_reachable_positions_tensor,
                            reachable,
                            visited_tensor,
                            other_agent_position_tensors_list[agent_id],
                            object_tensor,
                        )
                        + to_object_tensor_tuple,
                        axis=0,
                    )
                    for agent_id in range(self.environment.num_agents)
                ]
            else:
                raise Exception("something wrong with conditions, check!")

            for agent_id in range(self.environment.num_agents):
                agent_pos = self.environment.get_agent_location(agent_id=agent_id)
                agent_x = round(agent_pos["x"], 2)
                agent_z = round(agent_pos["z"], 2)
                pos_tuple = (agent_x, agent_z)
                row, col = point_to_element_map[pos_tuple]
                agent_location_in_mask = self.environment.get_agent_location_in_mask(
                    agent_id=agent_id
                )
                if not row == agent_location_in_mask["row"]:
                    print(
                        "row: {} | agent_location_in_mask[row]: {}".format(
                            row, agent_location_in_mask["row"]
                        )
                    )
                if not col == agent_location_in_mask["col"]:
                    print(
                        "col: {} | agent_location_in_mask[col]: {}".format(
                            col, agent_location_in_mask["col"]
                        )
                    )
                rot = round(agent_location_in_mask["rot"] / 90) % 4

                outsize = 15
                padding = outsize - 1
                assert outsize % 2 == 1
                tensor = np.pad(
                    output_tensor_per_agent[agent_id],
                    [(0, 0), (padding, padding), (padding, padding)],
                    "constant",
                    constant_values=False,
                )
                row = row + padding
                col = col + padding

                half_pad = padding // 2
                if rot == 0:
                    egocentric_tensor = tensor[
                        :,
                        (row - padding) : (row + 1),
                        (col - half_pad) : (col + half_pad + 1),
                    ]
                elif rot == 1:
                    egocentric_tensor = tensor[
                        :,
                        (row - half_pad) : (row + half_pad + 1),
                        col : (col + padding + 1),
                    ]
                    egocentric_tensor = np.rot90(egocentric_tensor, axes=(1, 2))
                elif rot == 2:
                    egocentric_tensor = tensor[
                        :,
                        row : (row + padding + 1),
                        (col - half_pad) : (col + half_pad + 1),
                    ]
                    egocentric_tensor = np.rot90(egocentric_tensor, k=2, axes=(1, 2))
                elif rot == 3:
                    egocentric_tensor = tensor[
                        :,
                        (row - half_pad) : (row + half_pad + 1),
                        (col - padding) : (col + 1),
                    ]
                    egocentric_tensor = np.rot90(egocentric_tensor, k=3, axes=(1, 2))
                else:
                    raise NotImplementedError()
                # TODO: See if this copy is needed?
                frames.append(np.array(egocentric_tensor, dtype=float))
        else:
            raise Exception("Invalid frame type {}.".format(frame_type))

        states = []
        for agent_id in range(self.environment.num_agents):
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
                    "last_action": last_action,
                    "last_action_success": last_action_success,
                    "frame": frames[agent_id],
                }
            )

        if (
            frame_type == self.frame_type
            and self.return_likely_successfuly_move_actions
        ) or (
            frame_type == self.expert_frame_type
            and self.return_likely_successfuly_move_actions_for_expert
        ):
            self.add_likely_successful_move_actions(
                states=states, frame_type=frame_type
            )

        return states

    def multi_step(self, actions_as_ints: Tuple[int, ...]):
        step_results = super(FurnMoveEgocentricFastGridEpisode, self).multi_step(
            actions_as_ints
        )
        if self.visualize_test_gridworld:
            self.environment.visualize(self.visualizing_ms)
        if self.track_time_averaged_action_pairs:
            assert self.environment.num_agents == 2
            self.action_pairs_matrix_attempted[
                actions_as_ints[0], actions_as_ints[1]
            ] += 1
            self.action_pairs_matrix_successful[
                actions_as_ints[0], actions_as_ints[1]
            ] += int(all([sr["action_success"] for sr in step_results]))
        return step_results

    def info(self):
        info = super(FurnMoveEgocentricFastGridEpisode, self).info()
        if self.track_time_averaged_action_pairs:
            total_actions = np.sum(self.action_pairs_matrix_attempted)
            info[
                "action_pairs_matrix_attempted"
            ] = self.action_pairs_matrix_attempted / (total_actions + 0.0001)
            info[
                "action_pairs_matrix_successful"
            ] = self.action_pairs_matrix_successful / (total_actions + 0.0001)
        return info


class FurnMoveEgocentricNoRotationsFastGridEpisode(FurnMoveEgocentricFastGridEpisode):
    def __init__(
        self,
        env: AI2ThorLiftedObjectGridEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            include_move_obj_actions=include_move_obj_actions,
            reached_target_reward=reached_target_reward,
            moved_closer_reward=moved_closer_reward,
            **kwargs,
        )

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        return egocentric_no_rotate_action_groups(
            include_move_obj_actions=include_move_obj_actions
        )


class FurnMoveAllocentricTVCenteredEpisode(FurnMoveEpisode):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        max_distance_from_object: float,
        include_move_obj_actions: bool = False,
        reached_target_reward: float = 1.0,
        moved_closer_reward: float = 0.1,
        **kwargs,
    ):
        assert "frame_type" not in kwargs
        kwargs["frame_type"] = "allocentric-tensor-centered-at-tv"
        super().__init__(
            env=env,
            task_data=task_data,
            max_steps=max_steps,
            max_distance_from_object=max_distance_from_object,
            include_move_obj_actions=include_move_obj_actions,
            reached_target_reward=reached_target_reward,
            moved_closer_reward=moved_closer_reward,
            **kwargs,
        )

    @classmethod
    def class_available_action_groups(
        cls, include_move_obj_actions: bool = False, **kwargs
    ) -> Tuple[Tuple[str, ...], ...]:
        return semiallocentric_action_groups(
            include_move_obj_actions=include_move_obj_actions
        )
