"""A wrapper for engaging with the THOR environment."""

import copy
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping

import ai2thor.server
import networkx as nx
import numpy as np
from ai2thor.controller import Controller

import constants
from rl_ai2thor.ai2thor_utils import pad_matrix_to_size_center, pad_matrix
from utils.misc_util import round_to_factor


class AI2ThorEnvironment(object):
    def __init__(
        self,
        docker_enabled: bool = False,
        x_display: str = None,
        local_thor_build: str = None,
        time_scale: float = 1.0,
        visibility_distance: float = constants.VISIBILITY_DISTANCE,
        fov: float = constants.FOV,
        restrict_to_initially_reachable_points: bool = False,
        num_agents: int = 1,
        visible_agents: bool = True,
        render_depth_image: bool = False,
        headless: bool = False,
        always_return_visible_range: bool = False,
        allow_agents_to_intersect: bool = False,
    ) -> None:
        self.num_agents = num_agents
        self.controller = Controller(headless=headless)
        self.controller.local_executable_path = local_thor_build
        self.controller.docker_enabled = docker_enabled
        self.x_display = x_display
        self._initially_reachable_points: Optional[List[Dict]] = None
        self._initially_reachable_points_set: Optional[Set[Dict]] = None
        self._started = False
        self.move_mag: Optional[float] = None
        self.grid_size: Optional[float] = None
        self._grid_size_digits: Optional[float] = None
        self.time_scale = time_scale
        self.visibility_distance = visibility_distance
        self.fov = fov
        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.visible_agents = visible_agents
        self.render_depth_image = render_depth_image
        self.headless = headless
        self.always_return_visible_range = always_return_visible_range
        self.allow_agents_to_intersect = allow_agents_to_intersect

    @property
    def scene_name(self) -> str:
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self) -> np.ndarray:
        return self.controller.last_event.frame

    @property
    def current_frames(self) -> Tuple[np.ndarray, ...]:
        return tuple(
            self.controller.last_event.events[i].frame for i in range(self.num_agents)
        )

    @property
    def current_depth_frames(self) -> Tuple[np.ndarray, ...]:
        if not self.render_depth_image:
            raise Exception(
                "Depth frames are not available, "
                "must set render_depth_image to true before initializing."
            )
        return tuple(
            self.controller.last_event.events[i].depth_frame
            for i in range(self.num_agents)
        )

    @property
    def last_event(self) -> ai2thor.server.Event:
        return self.controller.last_event

    @property
    def started(self) -> bool:
        return self._started

    def start(
        self,
        scene_name: Optional[str],
        move_mag: float = 0.25,
        player_screen_width=300,
        player_screen_height=300,
        quality="Very Low",
    ) -> None:
        if self.headless and (
            player_screen_height != 300 or player_screen_height != 300
        ):
            warnings.warn(
                "In headless mode but choosing non-default player screen width/height, will be ignored."
            )

        if player_screen_width < 300 or player_screen_height < 300:
            self.controller.start(
                x_display=self.x_display,
                player_screen_width=300,
                player_screen_height=300,
            )
            if not self.headless:
                self.controller.step(
                    {
                        "action": "ChangeResolution",
                        "x": player_screen_width,
                        "y": player_screen_height,
                    }
                )
        else:
            self.controller.start(
                x_display=self.x_display,
                player_screen_width=player_screen_width,
                player_screen_height=player_screen_height,
            )

        self.controller.step({"action": "ChangeQuality", "quality": quality})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            raise Exception("Failed to change quality to: {}.".format(quality))

        self._started = True
        self.reset(scene_name=scene_name, move_mag=move_mag)

    def stop(self) -> None:
        try:
            self.controller.stop_unity()
        except Exception as e:
            warnings.warn(str(e))
        finally:
            self._started = False

    def reset(
        self, scene_name: Optional[str], move_mag: float = 0.25,
    ):
        self.move_mag = move_mag
        self.grid_size = self.move_mag
        self._grid_size_digits = [
            i
            for i in range(2, 10)
            if abs(round(self.grid_size, i) - self.grid_size) < 1e-9
        ][0]
        assert self._grid_size_digits != 9, (
            "Bad grid size chosen. " "Should have a finite decimal expansion."
        )

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]
        self.controller.reset(scene_name)

        tmp_stderr = sys.stderr
        sys.stderr = open(
            os.devnull, "w"
        )  # TODO: HACKILY BLOCKING sequenceId print errors
        self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self.grid_size,
                "visibilityDistance": self.visibility_distance,
                "fov": self.fov,
                "timeScale": self.time_scale,
                # "sequenceId": 0,  # TODO: WHY IS THIS NECESSARY?
                "agentCount": self.num_agents,
                "makeAgentsVisible": self.visible_agents,
                "renderDepthImage": self.render_depth_image,
                "alwaysReturnVisibleRange": self.always_return_visible_range,
            }
        )
        sys.stderr.close()
        sys.stderr = tmp_stderr

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.controller.last_event.metadata[
            "reachablePositions"
        ]

    def teleport_agent_to(
        self,
        x: float,
        y: float,
        z: float,
        rotation: float,
        horizon: float,
        standing: Optional[bool] = None,
        force_action: bool = False,
        only_initially_reachable: bool = False,
        agent_id: int = None,
        render_image=True,
    ) -> None:
        if self.num_agents == 1 and agent_id == -1:
            agent_id = 0

        if standing is None:
            standing = self.last_event.metadata["isStanding"]
        if only_initially_reachable:
            reachable_points = self.initially_reachable_points
            target = {"x": x, "y": y, "z": z}
            reachable = False
            for p in reachable_points:
                if self.position_dist(target, p) < 0.01:
                    reachable = True
                    break
            if not reachable:
                self.last_event.metadata["lastAction"] = "TeleportFull"
                self.last_event.metadata[
                    "errorMessage"
                ] = "Target position was not initially reachable."
                self.last_event.metadata["lastActionSuccess"] = False
                return
        self.controller.step(
            dict(
                action="TeleportFull",
                x=x,
                y=y,
                z=z,
                rotation={"x": 0.0, "y": rotation, "z": 0.0},
                horizon=horizon,
                standing=standing,
                forceAction=force_action,
                agentId=agent_id,
                render_image=render_image,
            )
        )

    def random_reachable_state(
        self,
        seed: int = None,
        specific_rotations=(0, 90, 180, 270),
        specific_horizons=(0, 30, 60, 330),
        only_initially_reachable: bool = False,
    ) -> Dict:
        if seed is not None:
            random.seed(seed)
        if only_initially_reachable:
            xyz = random.choice(self.initially_reachable_points)
        else:
            xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice(specific_rotations)
        horizon = random.choice(specific_horizons)
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(
        self,
        seed: int = None,
        partial_position: Optional[Dict[str, float]] = None,
        agent_id: int = None,
        only_initially_reachable: bool = False,
    ) -> Dict:
        if partial_position is None:
            partial_position = {}
        k = 0
        while k == 0 or (not self.last_event.metadata["lastActionSuccess"] and k < 10):
            state = self.random_reachable_state(
                seed=seed, only_initially_reachable=only_initially_reachable
            )
            self.teleport_agent_to(**{**state, **partial_position}, agent_id=agent_id)
            k += 1

        if not self.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                (
                    "Randomize agent location in scene {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, seed, partial_position)
            )
            self.teleport_agent_to(
                **{**state, **partial_position}, force_action=True, agent_id=agent_id
            )
            assert self.last_event.metadata["lastActionSuccess"]
        return state

    @property
    def initially_reachable_points(self) -> List[Dict[str, float]]:
        assert self._initially_reachable_points is not None
        return copy.deepcopy(self._initially_reachable_points)  # type:ignore

    @property
    def initially_reachable_points_set(self) -> Set[Tuple[float, float]]:
        if self._initially_reachable_points_set is None:
            self._initially_reachable_points_set = set()
            for p in self.initially_reachable_points:
                self._initially_reachable_points_set.add(
                    self._agent_location_to_tuple(p)
                )

        return self._initially_reachable_points_set

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        self.step({"action": "GetReachablePositions", "agentId": 0})
        return self.last_event.metadata["reachablePositions"]  # type:ignore

    def refresh_initially_reachable(self):
        self._initially_reachable_points = self.currently_reachable_points
        self._initially_reachable_points_set = None

    def currently_reachable_points_set(self) -> Set[Tuple[float, float]]:
        currently_reachable_points = self.currently_reachable_points
        currently_reachable_points_set = set()
        for p in currently_reachable_points:
            currently_reachable_points_set.add(self._agent_location_to_tuple(p))

        return currently_reachable_points_set

    def get_initially_unreachable_points(self) -> Set[Tuple[float, float]]:
        unreachable_points = set([])
        x_vals = set([p["x"] for p in self.initially_reachable_points])
        z_vals = set([p["z"] for p in self.initially_reachable_points])

        for x in x_vals:
            for z in z_vals:
                if (x, z) not in self.initially_reachable_points:
                    unreachable_points.add((x, z))
        return unreachable_points

    def _points_to_matrix(
        self, points: List[Dict[str, float]], padding: float = 0.0
    ) -> Tuple[List[List[bool]], Dict[Tuple, Tuple]]:
        xz_set = set(
            (
                round(p["x"], self._grid_size_digits),
                round(p["z"], self._grid_size_digits),
            )
            for p in points
        )

        xs = [p["x"] for p in points]
        zs = [p["z"] for p in points]
        n = 1.0 / self.grid_size
        x_min = math.floor(n * (min(*xs) - padding)) / n
        x_max = math.ceil(n * (max(*xs) + padding)) / n
        z_min = math.floor(n * (min(*zs) - padding)) / n
        z_max = math.ceil(n * (max(*zs) + padding)) / n

        x_vals = list(np.linspace(x_min, x_max, round(1 + (x_max - x_min) * 4)))
        z_vals = list(
            reversed(np.linspace(z_min, z_max, round(1 + (z_max - z_min) * 4)))
        )

        point_to_element_map = dict()
        matrix = [[False for _ in range(len(x_vals))] for _ in range(len(z_vals))]
        for i, x in enumerate(x_vals):
            for j, z in enumerate(z_vals):
                matrix[j][i] = (x, z) in xz_set
                point_to_element_map[(x, z)] = (j, i)
        return matrix, point_to_element_map

    def get_currently_reachable_points_matrix(
        self, padding: float = 0.0
    ) -> Tuple[List[List[bool]], Dict[Tuple, Tuple]]:
        return self._points_to_matrix(self.currently_reachable_points, padding=padding)

    def get_initially_reachable_points_matrix(
        self, padding: float = 0.0
    ) -> Tuple[List[List[bool]], Dict[Tuple, Tuple]]:
        return self._points_to_matrix(self.initially_reachable_points, padding=padding)

    def get_current_occupancy_matrix(
        self, padding: float = 0.0, use_initially_reachable_points_matrix: bool = False
    ) -> Tuple[np.ndarray, Dict[Tuple, Tuple]]:
        if use_initially_reachable_points_matrix:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_initially_reachable_points_matrix(padding=padding)
        else:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_currently_reachable_points_matrix(padding=padding)
        # 0/1 reachable point matrix
        matrix_bool = np.array(matrix_bool, dtype=bool)
        matrix = np.full(matrix_bool.shape, fill_value=constants.UNREACHABLE_SYM)
        matrix[matrix_bool] = constants.REACHABLE_SYM

        for i in range(self.num_agents):
            agent_location = self.get_agent_location(agent_id=i)
            xz_val = (
                round(agent_location["x"], self._grid_size_digits),
                round(agent_location["z"], self._grid_size_digits),
            )
            if xz_val in point_to_element_map:
                # TODO: FIX THIS?
                rowcol_val = point_to_element_map[xz_val]
                matrix[rowcol_val[0], rowcol_val[1]] = constants.AGENT_SYM
        return matrix, point_to_element_map

    def get_current_occupancy_matrices_two_agents(
        self, padding: float = 0.0, use_initially_reachable_points_matrix: bool = False
    ) -> Tuple[List[np.ndarray], Dict[Tuple, Tuple]]:
        if use_initially_reachable_points_matrix:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_initially_reachable_points_matrix(padding=padding)
        else:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_currently_reachable_points_matrix(padding=padding)
        # 0/1 reachable point matrix
        matrix_bool = np.array(matrix_bool, dtype=bool)
        matrix = np.full(matrix_bool.shape, fill_value=constants.UNREACHABLE_SYM)
        matrix[matrix_bool] = constants.REACHABLE_SYM
        matrix_all_agents = [copy.deepcopy(matrix) for _ in range(self.num_agents)]

        assert self.num_agents == 2
        my_symbols = [
            constants.AGENT_SELF_0,
            constants.AGENT_SELF_90,
            constants.AGENT_SELF_180,
            constants.AGENT_SELF_270,
        ]
        your_symbols = [
            constants.AGENT_OTHER_0,
            constants.AGENT_OTHER_90,
            constants.AGENT_OTHER_180,
            constants.AGENT_OTHER_270,
        ]

        for i in range(2):
            agent_location = self.get_agent_location(agent_id=i)
            xz_val = (
                round(agent_location["x"], self._grid_size_digits),
                round(agent_location["z"], self._grid_size_digits),
            )
            clock_90 = (int(agent_location["rotation"]) % 360) // 90
            if xz_val in point_to_element_map:
                # TODO: FIX THIS?
                rowcol_val = point_to_element_map[xz_val]
                matrix_all_agents[i][rowcol_val[0], rowcol_val[1]] = my_symbols[
                    clock_90
                ]
                matrix_all_agents[1 - i][rowcol_val[0], rowcol_val[1]] = your_symbols[
                    clock_90
                ]
        return matrix_all_agents, point_to_element_map

    def get_current_multi_agent_occupancy_tensors(
        self, padding: float = 0.0, use_initially_reachable_points_matrix: bool = False
    ) -> Tuple[List[np.ndarray], Dict[Tuple, Tuple]]:
        if use_initially_reachable_points_matrix:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_initially_reachable_points_matrix(padding=padding)
        else:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_currently_reachable_points_matrix(padding=padding)
        # 0/1 reachable point matrix
        reachable_tensor = np.array([matrix_bool], dtype=bool)

        positions_tensors = [
            np.zeros((4 * self.num_agents, *reachable_tensor.shape[-2:]), dtype=float)
            for _ in range(self.num_agents)
        ]

        for i in range(self.num_agents):
            agent_location = self.get_agent_location(agent_id=i)
            xz_val = (
                round(agent_location["x"], self._grid_size_digits),
                round(agent_location["z"], self._grid_size_digits),
            )
            clock_90 = (int(agent_location["rotation"]) % 360) // 90

            for j in range(self.num_agents):
                if xz_val in point_to_element_map:
                    # TODO: FIX THIS?
                    rowcol_val = point_to_element_map[xz_val]
                    positions_tensors[j][
                        clock_90 + 4 * ((i - j) % self.num_agents),
                        rowcol_val[0],
                        rowcol_val[1],
                    ] = 1.0

        return (
            [
                np.concatenate((reachable_tensor, pt), axis=0)
                for pt in positions_tensors
            ],
            point_to_element_map,
        )

    def get_current_multi_agent_occupancy_tensors_no_rot(
        self, padding: float = 0.0, use_initially_reachable_points_matrix: bool = False
    ) -> Tuple[List[np.ndarray], Dict[Tuple, Tuple]]:
        if use_initially_reachable_points_matrix:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_initially_reachable_points_matrix(padding=padding)
        else:
            (
                matrix_bool,
                point_to_element_map,
            ) = self.get_currently_reachable_points_matrix(padding=padding)
        # 0/1 reachable point matrix
        reachable_tensor = np.array([matrix_bool], dtype=bool)

        positions_tensors = [
            np.zeros((self.num_agents, *reachable_tensor.shape[-2:]), dtype=float)
            for _ in range(self.num_agents)
        ]

        for i in range(self.num_agents):
            agent_location = self.get_agent_location(agent_id=i)
            xz_val = (
                round(agent_location["x"], self._grid_size_digits),
                round(agent_location["z"], self._grid_size_digits),
            )

            for j in range(self.num_agents):
                if xz_val in point_to_element_map:
                    rowcol_val = point_to_element_map[xz_val]
                    positions_tensors[j][
                        (i - j) % self.num_agents, rowcol_val[0], rowcol_val[1]
                    ] = 1.0

        # TODO: Simplify with `positions_tensors[i] = np.roll(positions_tensors[0], 4*i, axis=0)`?

        return (
            [
                np.concatenate((reachable_tensor, pt), axis=0)
                for pt in positions_tensors
            ],
            point_to_element_map,
        )

    def get_agent_location(self, agent_id: int = None) -> Dict[str, float]:
        if self.num_agents == 1:
            metadata = self.controller.last_event.metadata
        else:
            metadata = self.controller.last_event.events[agent_id].metadata
        location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
        }
        return location

    def _agent_location_to_tuple(self, p):
        return (
            round(p["x"], self._grid_size_digits),
            round(p["z"], self._grid_size_digits),
        )

    def get_agent_locations(self) -> Tuple[Dict[str, float], ...]:
        """Gets all agents' locations."""
        return tuple(self.get_agent_location(i) for i in range(self.num_agents))

    def get_agent_metadata(self, agent_id: int = 0) -> Dict[str, Any]:
        """Gets agent's metadata."""
        return self.controller.last_event.events[agent_id].metadata["agent"]

    def get_all_agent_metadata(self) -> Tuple[Dict[str, Any], ...]:
        """Gets all agents' locations."""
        return tuple(self.get_agent_metadata(i) for i in range(self.num_agents))

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        action = action_dict["action"]
        agent_id = action_dict.get("agentId")
        if agent_id is not None:
            assert type(agent_id) == int and 0 <= agent_id <= self.num_agents
        elif action == "RandomlyCreateLiftedFurniture":
            pass
        else:
            assert self.num_agents == 1
            agent_id = 0

        if (
            self.allow_agents_to_intersect
            and "allowAgentsToIntersect" not in action_dict
        ):
            action_dict["allowAgentsToIntersect"] = True

        if "MoveAgents" in action:
            assert "Object" in action

            action_dict = {
                **action_dict,
                "moveMagnitude": self.move_mag,
            }  # type: ignore
            start_agent_locations = [
                self.get_agent_location(agent_id=i) for i in range(self.num_agents)
            ]
            object_info = self.get_object_by_id(action_dict["objectId"], agent_id=0)

            self.controller.step(action_dict)
            if self.restrict_to_initially_reachable_points:
                end_locations = [
                    self._agent_location_to_tuple(self.get_agent_location(agent_id=i))
                    for i in range(self.num_agents)
                ]
                if any(
                    t not in self.initially_reachable_points_set for t in end_locations
                ):
                    for i in range(self.num_agents):
                        self.teleport_agent_to(
                            **start_agent_locations[i], agent_id=i, force_action=True
                        )
                    self.controller.step(
                        {
                            "action": "TeleportObject",
                            "objectId": action_dict["objectId"],
                            **object_info["position"],
                            "rotation": object_info["rotation"],
                            "forceAction": True,
                        }
                    )
                    self.last_event.events[agent_id].metadata["lastAction"] = action
                    self.last_event.events[agent_id].metadata[
                        "lastActionSuccess"
                    ] = False
                    self.last_event.events[agent_id].metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
                    self.last_event.metadata = self.last_event.events[agent_id].metadata

        elif "Move" in action and "Hand" not in action:  # type: ignore
            action_dict = {
                **action_dict,
                "moveMagnitude": self.move_mag,
            }  # type: ignore
            start_location = self.get_agent_location(agent_id=agent_id)
            self.controller.step(action_dict)

            if self.restrict_to_initially_reachable_points:
                end_location_tuple = self._agent_location_to_tuple(
                    self.get_agent_location(agent_id=agent_id)
                )
                if end_location_tuple not in self.initially_reachable_points_set:
                    self.teleport_agent_to(
                        **start_location, agent_id=agent_id, force_action=True
                    )
                    self.last_event.metadata["lastAction"] = action
                    self.last_event.metadata["lastActionSuccess"] = False
                    self.last_event.metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
        elif "RandomizeHideSeekObjects" in action:
            last_positions = [
                self.get_agent_location(agent_id=i) for i in range(self.num_agents)
            ]
            self.controller.step(action_dict)
            metadata = self.last_event.metadata
            for i, lp in enumerate(last_positions):
                if self.position_dist(lp, self.get_agent_location(agent_id=i)) > 0.001:
                    self.teleport_agent_to(**lp, agent_id=agent_id, force_action=True)
                    warnings.warn(
                        "In scene {}, after randomization of hide and seek objects, agent {} moved.".format(
                            self.scene_name, i
                        )
                    )

            self.controller.step({"action": "GetReachablePositions"})
            self._initially_reachable_points = self.controller.last_event.metadata[
                "reachablePositions"
            ]
            self._initially_reachable_points_set = None
            self.controller.last_event.metadata["lastAction"] = action
            self.controller.last_event.metadata["lastActionSuccess"] = metadata[
                "lastActionSuccess"
            ]
            self.controller.last_event.metadata["reachablePositions"] = []
        else:
            return self.controller.step(action_dict)

    @staticmethod
    def position_dist(
        p0: Mapping[str, Any], p1: Mapping[str, Any], use_l1: bool = False
    ) -> float:
        if use_l1:
            return (
                abs(p0["x"] - p1["x"]) + abs(p0["y"] - p1["y"]) + abs(p0["z"] - p1["z"])
            )
        else:
            return math.sqrt(
                (p0["x"] - p1["x"]) ** 2
                + (p0["y"] - p1["y"]) ** 2
                + (p0["z"] - p1["z"]) ** 2
            )

    def closest_object_with_properties(
        self, properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        min_dist = float("inf")
        closest = None
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                d = self.position_dist(agent_pos, o["position"])
                if d < min_dist:
                    min_dist = d
                    closest = o
        return closest

    def closest_visible_object_of_type(self, type: str) -> Optional[Dict[str, Any]]:
        properties = {"visible": True, "objectType": type}
        return self.closest_object_with_properties(properties)

    def closest_object_of_type(self, type: str) -> Optional[Dict[str, Any]]:
        properties = {"objectType": type}
        return self.closest_object_with_properties(properties)

    def closest_reachable_point_to_position(
        self, position: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        target = np.array([position["x"], position["z"]])
        min_dist = float("inf")
        closest_point = None
        for pt in self.initially_reachable_points:
            dist = np.linalg.norm(target - np.array([pt["x"], pt["z"]]))
            if dist < min_dist:
                closest_point = pt
                min_dist = dist
                if min_dist < 1e-3:
                    break
        assert closest_point is not None
        return closest_point, min_dist

    @staticmethod
    def _angle_from_to(a_from: float, a_to: float) -> float:
        a_from = a_from % 360
        a_to = a_to % 360
        min_rot = min(a_from, a_to)
        max_rot = max(a_from, a_to)
        rot_across_0 = (360 - max_rot) + min_rot
        rot_not_across_0 = max_rot - min_rot
        rot_err = min(rot_across_0, rot_not_across_0)
        if rot_across_0 == rot_err:
            rot_err *= -1 if a_to > a_from else 1
        else:
            rot_err *= 1 if a_to > a_from else -1
        return rot_err

    def agent_xz_to_scene_xz(self, agent_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()

        x_rel_agent = agent_xz["x"]
        z_rel_agent = agent_xz["z"]
        scene_x = agent_pos["x"]
        scene_z = agent_pos["z"]
        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            scene_x += x_rel_agent
            scene_z += z_rel_agent
        elif abs(rotation - 90) < 1e-5:
            scene_x += z_rel_agent
            scene_z += -x_rel_agent
        elif abs(rotation - 180) < 1e-5:
            scene_x += -x_rel_agent
            scene_z += -z_rel_agent
        elif abs(rotation - 270) < 1e-5:
            scene_x += -z_rel_agent
            scene_z += x_rel_agent
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": scene_x, "z": scene_z}

    def scene_xz_to_agent_xz(self, scene_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()
        x_err = scene_xz["x"] - agent_pos["x"]
        z_err = scene_xz["z"] - agent_pos["z"]

        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            agent_x = x_err
            agent_z = z_err
        elif abs(rotation - 90) < 1e-5:
            agent_x = -z_err
            agent_z = x_err
        elif abs(rotation - 180) < 1e-5:
            agent_x = -x_err
            agent_z = -z_err
        elif abs(rotation - 270) < 1e-5:
            agent_x = z_err
            agent_z = -x_err
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": agent_x, "z": agent_z}

    def current_matrix_frame(
        self,
        agent_id: int,
        matrix: np.ndarray,
        point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
        d_ahead: int,
        d_side: int,
    ) -> np.ndarray:
        padded_matrix, point_to_pad_element_map = pad_matrix(
            matrix, pad=max(d_ahead, d_side), point_to_element_map=point_to_element_map
        )
        agent_pos = self.get_agent_location(agent_id=agent_id)
        agent_x = round(agent_pos["x"], self._grid_size_digits)
        agent_z = round(agent_pos["z"], self._grid_size_digits)
        (agent_row, agent_col) = point_to_pad_element_map[(agent_x, agent_z)]
        rotation = int(agent_pos["rotation"]) % 360

        if rotation == 0:
            local_ego_matrix = padded_matrix[
                agent_row - d_ahead : agent_row,
                agent_col - d_side : agent_col + d_side + 1,
            ]
        elif rotation == 90:
            local_matrix = padded_matrix[
                agent_row - d_side : agent_row + d_side + 1,
                agent_col + 1 : agent_col + d_ahead + 1,
            ]
            local_ego_matrix = np.rot90(local_matrix, 1)
        elif rotation == 180:
            local_matrix = padded_matrix[
                agent_row + 1 : agent_row + d_ahead + 1,
                agent_col - d_side : agent_col + d_side + 1,
            ]
            local_ego_matrix = np.rot90(local_matrix, 2)
        elif rotation == 270:
            local_matrix = padded_matrix[
                agent_row - d_side : agent_row + d_side + 1,
                agent_col - d_ahead : agent_col,
            ]
            local_ego_matrix = np.rot90(local_matrix, 3)
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")
        assert local_ego_matrix.shape == (d_ahead, 2 * d_side + 1)
        return local_ego_matrix

    def current_allocentric_matrix_frame(
        self,
        agent_id: int,
        matrix: np.ndarray,
        point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
        d_each_side: int,
    ) -> np.ndarray:
        padded_matrix, point_to_pad_element_map = pad_matrix(
            matrix, pad=d_each_side, point_to_element_map=point_to_element_map
        )
        agent_pos = self.get_agent_location(agent_id=agent_id)
        agent_x = round(agent_pos["x"], self._grid_size_digits)
        agent_z = round(agent_pos["z"], self._grid_size_digits)
        (agent_row, agent_col) = point_to_pad_element_map[(agent_x, agent_z)]

        local_allo_matrix = padded_matrix[
            agent_row - d_each_side : agent_row + d_each_side + 1,
            agent_col - d_each_side : agent_col + d_each_side + 1,
        ]
        assert local_allo_matrix.shape == (2 * d_each_side + 1, 2 * d_each_side + 1)
        return local_allo_matrix

    def current_allocentric_matrix_frame_full_range_center(
        self,
        matrix: np.ndarray,
        point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
        desired_output_shape: Tuple[int, int],
    ) -> np.ndarray:
        global_allo_matrix, point_to_pad_element_map = pad_matrix_to_size_center(
            matrix,
            desired_output_shape=desired_output_shape,
            point_to_element_map=point_to_element_map,
        )
        assert global_allo_matrix.shape == desired_output_shape
        return global_allo_matrix

    def all_objects(self, agent_id: int = None) -> List[Dict[str, Any]]:
        if self.num_agents == 1:
            agent_id = 0
        return self.controller.last_event.events[agent_id].metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any], agent_id: int = None
    ) -> List[Dict[str, Any]]:
        if self.num_agents == 1:
            agent_id = 0

        objects = []
        for o in self.all_objects(agent_id=agent_id):
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self, agent_id: int = None) -> List[Dict[str, Any]]:
        if self.num_agents == 1:
            agent_id = 0
        return self.all_objects_with_properties({"visible": True}, agent_id=agent_id)

    def get_object_by_id(
        self, object_id: str, agent_id: Optional[int] = None
    ) -> Dict[str, Any]:
        if self.num_agents == 0:
            agent_id = 0
        return [
            o
            for o in self.last_event.events[agent_id].metadata["objects"]
            if o["objectId"] == object_id
        ][0]


class AI2ThorEnvironmentWithGraph(AI2ThorEnvironment):
    _cached_graphs: Dict[str, nx.DiGraph] = {}

    def __init__(
        self,
        docker_enabled: bool = False,
        x_display: str = None,
        local_thor_build: str = None,
        time_scale: float = 1.0,
        visibility_distance: float = constants.VISIBILITY_DISTANCE,
        fov: float = constants.FOV,
        restrict_to_initially_reachable_points: bool = False,
        num_agents: int = 1,
        visible_agents: bool = True,
        render_depth_image: bool = False,
        override_graph: Union[
            nx.classes.digraph.DiGraph, nx.classes.graph.Graph
        ] = None,
        **kwargs,
    ):
        super(AI2ThorEnvironmentWithGraph, self).__init__(
            docker_enabled=docker_enabled,
            x_display=x_display,
            local_thor_build=local_thor_build,
            time_scale=time_scale,
            visibility_distance=visibility_distance,
            fov=fov,
            restrict_to_initially_reachable_points=restrict_to_initially_reachable_points,
            num_agents=num_agents,
            visible_agents=visible_agents,
            render_depth_image=render_depth_image,
            **kwargs,
        )
        if override_graph:
            self._cached_graphs[self.scene_name] = override_graph

    def initially_reachable_points_with_rotations(self, horizon):
        points_slim = self.initially_reachable_points
        points = []
        for r in [0, 90, 180, 270]:
            for p in points_slim:
                p = copy.copy(p)
                p["rotation"] = r
                p["horizon"] = horizon
                points.append(p)
        return points

    def refresh_initially_reachable(self):
        self._initially_reachable_points = self.currently_reachable_points
        self._initially_reachable_points_set = None

        if self.scene_name in self._cached_graphs:
            g = self._cached_graphs[self.scene_name]
            initially_reachable_keys_set = set(
                self.get_key(p)
                for p in self.initially_reachable_points_with_rotations(horizon=30)
            )

            for n in list(g.nodes()):
                if n not in initially_reachable_keys_set:
                    g.remove_node(n)
            for n in initially_reachable_keys_set:
                if n not in g:
                    self._add_node_to_graph(g, n)

    def update_graph_with_failed_action(self, failed, agent_id):
        source_key = self.get_key(self.last_event.events[agent_id].metadata["agent"])
        e_dict = self.graph[source_key]
        to_remove_key = None
        for t_key in self.graph[source_key]:
            if e_dict[t_key]["action"] == failed:
                to_remove_key = t_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(self, g, s, t):
        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot, s_hor = s
        t_x, t_z, t_rot, t_hor = t

        dist = round(math.sqrt((s_x - t_x) ** 2 + (s_z - t_z) ** 2), 5)
        angle_dist = abs(s_rot - t_rot) % 360

        if dist == 0 and angle_dist == 90:
            if (s_rot + 90) % 360 == s_rot % 360:
                action = "RotateRight"
            else:
                action = "RotateLeft"
            g.add_edge(s, t, action=action)
        elif dist == 0.25 and s_rot == t_rot:
            if (
                (s_rot == 0 and ae(t_z - s_z, 0.25))
                or (s_rot == 90 and ae(t_x - s_x, 0.25))
                or (s_rot == 180 and ae(t_z - s_z, -0.25))
                or (s_rot == 270 and ae(t_x - s_x, -0.25))
            ):
                g.add_edge(s, t, action="MoveAhead")

    def _add_node_to_graph(self, graph: nx.DiGraph, s: Tuple[float, float, int, int]):
        if s in graph:
            return

        existing_nodes = list(graph.nodes())
        graph.add_node(s)
        for t in existing_nodes:
            self._add_from_to_edge(graph, s, t)
            self._add_from_to_edge(graph, t, s)

    @property
    def graph(self):
        if self.scene_name not in self._cached_graphs:
            g = nx.DiGraph()
            points = self.initially_reachable_points_with_rotations(horizon=30)
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._cached_graphs[self.scene_name] = g
        return self._cached_graphs[self.scene_name]

    @graph.setter
    def graph(self, g):
        self._cached_graphs[self.scene_name] = g

    def location_for_key(self, key, y_value=0.0):
        x, z, rot, hor = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot, horizon=hor)
        return loc

    def get_key(self, input) -> Tuple[float, float, int, int]:
        if "x" in input:
            x = input["x"]
            z = input["z"]
            rot = input["rotation"]
            hor = input["horizon"]
        else:
            x = input["position"]["x"]
            z = input["position"]["z"]
            rot = input["rotation"]["y"]
            hor = input["cameraHorizon"]

        return (
            round(x, 2),
            round(z, 2),
            round_to_factor(rot, 90),
            round_to_factor(hor, 30),
        )

    def is_neighbor_and_facing(self, p, possible_neighbor) -> bool:
        def ae(x, y):
            return abs(x - y) < 0.001

        p0 = p
        p1 = possible_neighbor

        return (
            (ae(p1["x"] - p0["x"], 0.25) and ae(p1["rotation"], 270))
            or (ae(p1["x"] - p0["x"], -0.25) and ae(p1["rotation"], 90))
            or (ae(p1["z"] - p0["z"], 0.25) and ae(p1["rotation"], 180))
            or (ae(p1["z"] - p0["z"], -0.25) and ae(p1["rotation"], 0))
        )

    def _check_contains_key(self, key: Tuple[float, float, int, int], add_if_not=True):
        if key not in self.graph:
            warnings.warn(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            self._add_node_to_graph(self.graph, key)

    def shortest_state_path(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        path = nx.shortest_path(self.graph, source_state_key, goal_state_key)
        return path

    def shortest_path_next_state(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        if source_state_key == goal_state_key:
            raise Exception("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)

        # FIXME: Make this generic for any action_space, currently hardcoded to "MoveAhead", "RotateRight", "RotateLeft", "LookUp", "Lookdown"
        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        if not self.graph.has_edge(source_state_key, next_state_key):
            print(
                "source_state_key: "
                + str(source_state_key)
                + "\ngoal_state_key: "
                + str(goal_state_key)
                + "\nnext_state_key: "
                + str(next_state_key)
            )
            raise Exception(
                "calculated next state is not reachable from source state, check!"
            )
        source_loc = self.location_for_key(source_state_key)
        next_loc = self.location_for_key(next_state_key)
        diff = defaultdict(lambda: None)
        # Not so clean way to check that strictly one of x,z,rotation or horizon changes.
        diff_detected = False
        for key in source_loc.keys():
            if source_loc[key] != next_loc[key]:
                if diff_detected:
                    raise Exception(
                        "More than one basic action required to move to next node state, check!"
                    )
                diff[key] = next_loc[key] - source_loc[key]
                diff_detected = key
        if diff_detected == "x" or diff_detected == "z":
            return "MoveAhead"
        elif diff_detected == "rotation":
            if (source_loc["rotation"] + 90) % 360 == next_loc["rotation"]:
                return "RotateRight"
            elif (source_loc["rotation"] - 90) % 360 == next_loc["rotation"]:
                return "RotateLeft"
            else:
                raise Exception("Cannot reach next state in one rotate action")
        elif diff_detected == "horizon":
            source_horizon = round(source_loc["horizon"] / 30) * 30.0
            next_horizon = round(next_loc["horizon"] / 30) * 30.0
            if source_horizon + 30 == next_horizon:
                return "LookDown"
            elif source_horizon - 30 == next_horizon:
                return "LookUp"
            else:
                raise Exception("Cannot reach next state in one look up/down action")
        else:
            raise Exception("no change in x, z, rotation or camera, check!")

    def shortest_path_length(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")
