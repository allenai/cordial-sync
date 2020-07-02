import copy
import itertools
import math
# noinspection PyUnresolvedReferences
import random
import re
import warnings
from typing import List, Dict, Optional, Any, Set, Tuple, Union

import ai2thor.server
import cv2
import numpy as np
from ai2thor.server import Event, MultiAgentEvent
from scipy.ndimage.measurements import label

import constants
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment

MOVE_MAP = {
    0: dict(row=-1, col=0),
    90: dict(row=0, col=1),
    180: dict(row=1, col=0),
    270: dict(row=0, col=-1),
}

SMALL_TELEVISION_TEMPLATE_STRING = """
0 0 2 0 0
0 2 1 2 0
0 2 1 1 2 
0 2 1 2 0
0 0 2 0 0 
"""


class GridWorldController(object):
    def __init__(
        self,
        agent_initially_reachable_pos: List[Dict[str, float]],
        rotation_to_lifted_object_reachable_pos: Dict[int, List[Dict[str, float]]],
        lifted_object_id: str,
        lifted_object_type: str,
        object_template_string: str,
        scene_name,
        min_steps_between_agents: int = 1,
        grid_size=0.25,
        remove_unconnected_positions=False,
    ):
        # Initially reachble pos and mask don't change over the course of episodes
        self.agent_initially_reachable_pos_tuple = tuple(agent_initially_reachable_pos)
        self.agent_initially_reachable_positions_mask = None
        self.rotation_to_lifted_object_reachable_pos = {
            int(k): rotation_to_lifted_object_reachable_pos[k]
            for k in rotation_to_lifted_object_reachable_pos
        }
        self.grid_size = grid_size
        self.remove_unconnected_positions = remove_unconnected_positions
        self.lifted_object_id = lifted_object_id
        self.lifted_object_type = lifted_object_type
        self.lifted_object: Optional[TrackedObject] = None
        self.lifted_object_template = None
        self.scene_name = scene_name
        self.agents: List[AgentObject] = []
        self.agent_count = 0

        self.lifted_object_template = self.parse_template_to_mask(
            object_template_string
        )
        self.min_steps_between_agents = min_steps_between_agents

        # Run only once, in the initialization
        self._build_grid_world(
            padding_units=max(3, *self.lifted_object_template.shape)
            if not remove_unconnected_positions
            else max(
                2,
                (self.lifted_object_template == 1).sum(1).max() // 2,
                (self.lifted_object_template == 1).sum(0).max() // 2,
            )
        )

        self.last_event = None
        self.steps_taken = 0

        self.tracked_objects: Dict[str, TrackedObject] = dict()

        self._error_message = ""

    def start(self):
        pass

    def reset(self, scene_name):
        assert self.scene_name is None or scene_name == self.scene_name
        self.scene_name = scene_name
        self.last_event = None
        self.agent_count = 1
        self.agents = []
        self.steps_taken = 0
        self._error_message = ""

    def parse_template_to_mask(self, template):
        tv_tmpl = []
        for line in template.strip().split("\n"):
            row = map(lambda x: int(x.strip()), line.split())
            tv_tmpl.append(list(row))

        return np.array(tv_tmpl, dtype=np.uint8)

    def empty_mask(self):
        return np.zeros((self._nrows, self._ncols), dtype=np.bool)

    def _build_grid_world(self, padding_units):
        # Initializes a lot of the basic chasis.
        # Doesn't set the location for agents or lifted object or target

        self._min_x = 2 ** 32
        self._max_x = -1 * 2 ** 32
        self._min_z = 2 ** 32
        self._max_z = -1 * 2 ** 32

        for point in self.agent_initially_reachable_pos_tuple:
            if point["x"] < self._min_x:
                self._min_x = point["x"]

            if point["z"] < self._min_z:
                self._min_z = point["z"]

            if point["z"] > self._max_z:
                self._max_z = point["z"]

            if point["x"] > self._max_x:
                self._max_x = point["x"]

        for point in sum(self.rotation_to_lifted_object_reachable_pos.values(), []):
            if point["x"] < self._min_x:
                self._min_x = point["x"]

            if point["z"] < self._min_z:
                self._min_z = point["z"]

            if point["z"] > self._max_z:
                self._max_z = point["z"]

            if point["x"] > self._max_x:
                self._max_x = point["x"]

        # adding buffer of 6 (1.0 / grid_size) points to allow for the origin
        # of the object to be at the edge
        self._max_z += padding_units * self.grid_size
        self._max_x += padding_units * self.grid_size
        self._min_z -= padding_units * self.grid_size
        self._min_x -= padding_units * self.grid_size

        self._ncols = int((self._max_x - self._min_x) / self.grid_size) + 1
        self._nrows = int((self._max_z - self._min_z) / self.grid_size) + 1

        self.agent_reachable_positions_mask = self.empty_mask()
        self.rotation_to_lifted_object_reachable_position_masks = {
            rot: self.empty_mask()
            for rot in self.rotation_to_lifted_object_reachable_pos
        }

        for rot in self.rotation_to_lifted_object_reachable_pos:
            self._build_points_mask(
                self.rotation_to_lifted_object_reachable_pos[rot],
                self.rotation_to_lifted_object_reachable_position_masks[rot],
            )
        self._build_points_mask(
            self.agent_initially_reachable_pos_tuple,
            self.agent_reachable_positions_mask,
        )

        if self.remove_unconnected_positions:
            flat_masks = np.stack(
                self.agent_reachable_positions_mask
                + list(
                    self.rotation_to_lifted_object_reachable_position_masks.values()
                ),
                axis=0,
            ).any(0)

            labels, ncomponents = label(
                flat_masks, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            )

            if ncomponents > 1:
                reachable_point = np.argwhere(self.agent_reachable_positions_mask)[0]
                good_label = labels[tuple(reachable_point)]

                connected_mask = labels == good_label

                row_span = np.argwhere(connected_mask.any(axis=1))
                row_min, row_max = (
                    max(row_span.min() - padding_units, 0),
                    min(row_span.max() + padding_units, connected_mask.shape[0]),
                )
                row_slice = slice(row_min, row_max + 1)

                col_span = np.argwhere(connected_mask.any(axis=0))
                col_min, col_max = (
                    max(col_span.min() - padding_units, 0),
                    min(col_span.max() + padding_units, connected_mask.shape[1]),
                )
                col_slice = slice(col_min, col_max + 1)

                for (k, v) in list(
                    self.rotation_to_lifted_object_reachable_position_masks.items()
                ):
                    self.rotation_to_lifted_object_reachable_position_masks[
                        k
                    ] = np.logical_and(v, connected_mask)[row_slice, col_slice]

                self.agent_reachable_positions_mask = self.agent_reachable_positions_mask[
                    row_slice, col_slice
                ]

                new_xz_mins = self.rowcol_to_xz((row_max, col_min))
                new_xz_maxes = self.rowcol_to_xz((row_min, col_max))

                self._max_x, self._max_z = tuple(float(w) for w in new_xz_maxes)
                self._min_x, self._min_z = tuple(float(w) for w in new_xz_mins)

                (self._nrows, self._ncols,) = self.agent_reachable_positions_mask.shape

        self.agent_initially_reachable_positions_mask = copy.deepcopy(
            self.agent_reachable_positions_mask
        )

    @property
    def min_x(self):
        return self._min_x

    @property
    def max_x(self):
        return self._max_x

    @property
    def min_z(self):
        return self._min_z

    @property
    def max_z(self):
        return self._max_z

    def rowcol_to_xz(self, rowcol):
        row, col = rowcol
        x = (col * self.grid_size) + self._min_x
        z = (-row * self.grid_size) + self._max_z
        return x, z

    def xz_to_rowcol(self, xz):
        x, z = xz
        row = round((self._max_z - z) / self.grid_size)
        col = round((x - self._min_x) / self.grid_size)
        return int(row), int(col)

    def _build_points_mask(self, points, mask):
        for point in points:
            row, col = self.xz_to_rowcol((point["x"], point["z"]))
            mask[row, col] = True

    def viz_mask(self, mask):
        viz_scale = 20
        viz_image = (
            np.ones(
                (self._nrows * viz_scale, self._ncols * viz_scale, 3), dtype=np.uint8
            )
            * 255
        )

        for point in np.argwhere(mask):
            cv2.circle(
                viz_image,
                (point[1] * viz_scale, point[0] * viz_scale),
                4,
                (255, 0, 0),
                -1,
            )

        cv2.imshow("aoeu", viz_image)
        cv2.waitKey(2000)

    def get_boundary_of_mask(self, mask):
        mask_points = np.argwhere(mask)

        if len(mask_points) <= 1:
            raise Exception("Too few valid mask points")

        leftmost_ind = np.argmin(mask_points[:, 1])

        leftmost_point = mask_points[leftmost_ind, :]

        path = [leftmost_point]

        up = np.array([-1, 0])
        right = np.array([0, 1])
        down = np.array([1, 0])
        left = np.array([0, -1])
        dirs = [up, right, down, left]

        def inbounds(p):
            return (
                0 <= p[0] < mask.shape[0]
                and 0 <= p[1] < mask.shape[1]
                and mask[p[0], p[1]]
            )

        dir_ind = 0
        while dir_ind < 6:
            move_success = False
            dir = dirs[dir_ind % 4]

            p = path[-1] + dir
            if inbounds(p):
                move_success = True
                path.append(p)

            if not move_success:
                p = path[-1] + dirs[(dir_ind + 1) % 4]
                if inbounds(p):
                    move_success = True
                    path.append(p)

            if not move_success:
                dir_ind += 1

            if len(path) > 1 and np.all(path[0] == path[-1]):
                break

        if dir_ind == 6:
            raise Exception("Error occurred!")

        return np.array(path)

    def draw_path_from_row_cols(
        self,
        points,
        viz_image,
        viz_scale,
        color,
        thickness,
        expand_from_center: Optional[np.ndarray] = None,
    ):
        if expand_from_center is not None:
            points = points - 0.5 * (points < expand_from_center.reshape((1, -1)))
            points = points + 0.5 * (points > expand_from_center.reshape((1, -1)))

        for ind in range(len(points) - 1):
            p0 = points[ind]
            p1 = points[ind + 1]
            cv2.line(
                viz_image,
                (int(round(p0[1] * viz_scale)), int(round(p0[0] * viz_scale))),
                (int(round(p1[1] * viz_scale)), int(round(p1[0] * viz_scale))),
                color,
                thickness,
            )

    def viz_world(self, viz_scale=20, do_wait=True, wait_key=0, array_only=False):
        viz_image = (
            np.ones(
                (self._nrows * viz_scale, self._ncols * viz_scale, 3), dtype=np.uint8
            )
            * 255
        )

        for p in np.argwhere(self.agent_reachable_positions_mask):
            tl = (p[1] * viz_scale - viz_scale // 4, p[0] * viz_scale - viz_scale // 4)
            br = (p[1] * viz_scale + viz_scale // 4, p[0] * viz_scale + viz_scale // 4)

            cv2.rectangle(viz_image, tl, br, (210, 210, 210), -1)

        masks = [
            self.rotation_to_lifted_object_reachable_position_masks[rot]
            for rot in sorted(
                list(self.rotation_to_lifted_object_reachable_position_masks.keys())
            )
        ]
        for p in np.argwhere((np.stack(masks, axis=0)).any(0) != 0):
            color = np.array([0, 0, 0])
            for i, mask in enumerate(masks):
                if mask[p[0], p[1]] and i < 3:
                    color[i] = 255
                elif mask[p[0], p[1]]:
                    color = color // 2

            offset = viz_scale // 10 + viz_scale // 4
            tl = (p[1] * viz_scale - offset, p[0] * viz_scale - offset)
            br = (p[1] * viz_scale + offset, p[0] * viz_scale + offset)

            cv2.rectangle(
                viz_image,
                tl,
                br,
                tuple(int(i) for i in color),
                thickness=viz_scale // 10,
            )

        for object_id, tracked_object in self.tracked_objects.items():
            if object_id == self.lifted_object_id:
                continue
            else:
                if hasattr(tracked_object, "object_mask"):
                    object_mask = tracked_object.object_mask
                    row = tracked_object.row
                    col = tracked_object.col
                    row_rad, col_rad = (
                        object_mask.shape[0] // 2,
                        object_mask.shape[1] // 2,
                    )
                    obj_mask = self.empty_mask()

                    obj_mask[
                        (row - row_rad) : (row + row_rad + 1),
                        (col - col_rad) : (col + col_rad + 1),
                    ] = object_mask

                    boundary_points = self.get_boundary_of_mask(obj_mask)

                    self.draw_path_from_row_cols(
                        points=boundary_points,
                        viz_image=viz_image,
                        viz_scale=viz_scale,
                        color=(255, 165, 0),
                        thickness=max(viz_scale // 10, 1),
                        expand_from_center=np.array([row, col]),
                    )

        if self.lifted_object is not None:
            self.draw_path_from_row_cols(
                points=self.get_boundary_of_mask(self.current_lifted_object_mask()),
                viz_image=viz_image,
                viz_scale=viz_scale,
                color=(255, 0, 255),
                thickness=2,
                expand_from_center=np.array(
                    [self.lifted_object.row, self.lifted_object.col]
                ),
            )

            self.draw_path_from_row_cols(
                points=self.get_boundary_of_mask(
                    self.current_near_lifted_object_mask()
                ),
                viz_image=viz_image,
                viz_scale=viz_scale,
                color=(180, 0, 180, 180),
                thickness=max(viz_scale // 10, 1),
                expand_from_center=np.array(
                    [self.lifted_object.row, self.lifted_object.col]
                ),
            )

        agent_colors = [  # Red + blue + tableau 10 colors
            (255, 0, 0),
            (0, 0, 255),
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
        ]
        for i, a in enumerate(self.agents):
            forward_dir = MOVE_MAP[a.rot]
            right_dir = MOVE_MAP[(a.rot + 90) % 360]
            cv2.drawContours(
                viz_image,
                [
                    np.array(
                        [
                            (
                                a.col * viz_scale
                                + forward_dir["col"] * (viz_scale // 3),
                                a.row * viz_scale
                                + forward_dir["row"] * (viz_scale // 3),
                            ),
                            (
                                a.col * viz_scale
                                - (
                                    right_dir["col"] * viz_scale // 4
                                    + forward_dir["col"] * viz_scale // 3
                                ),
                                a.row * viz_scale
                                - (
                                    right_dir["row"] * viz_scale // 4
                                    + forward_dir["row"] * viz_scale // 3
                                ),
                            ),
                            (
                                a.col * viz_scale
                                - (
                                    -right_dir["col"] * viz_scale // 4
                                    + forward_dir["col"] * viz_scale // 3
                                ),
                                a.row * viz_scale
                                - (
                                    -right_dir["row"] * viz_scale // 4
                                    + forward_dir["row"] * viz_scale // 3
                                ),
                            ),
                        ]
                    )
                ],
                0,
                agent_colors[i],
                -1,
            )

        if not array_only:
            cv2.imshow("aoeu", viz_image)
            if do_wait:
                return str(chr(cv2.waitKey(wait_key) & 255))
            else:
                cv2.waitKey(100)

        return viz_image

    def viz_ego_agent_views(
        self,
        viz_scale=20,
        view_shape=(15, 15),
        do_wait=True,
        wait_key=0,
        array_only=False,
    ):
        world = self.viz_world(viz_scale=viz_scale, array_only=True)

        assert view_shape[0] == view_shape[1]
        pad = viz_scale * view_shape[0]
        world = np.pad(
            world,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        def to_pix(a):
            return int(round(a * viz_scale + viz_scale * view_shape[0]))

        forward_width, side_width = view_shape
        ego_views = []
        for agent in self.agents:
            row, col, rot = agent.row, agent.col, int(agent.rot)

            if rot == 0:
                row -= 1 / 2
                row_pix_slice = slice(to_pix(row - forward_width), to_pix(row) + 1)
                col_pix_slice = slice(
                    to_pix(col - side_width / 2), to_pix(col + side_width / 2) + 1
                )
            elif rot == 180:
                row += 1 / 2
                row_pix_slice = slice(to_pix(row), to_pix(row + forward_width) + 1)
                col_pix_slice = slice(
                    to_pix(col - side_width / 2), to_pix(col + side_width / 2) + 1
                )
            elif rot == 90:
                col += 1 / 2
                row_pix_slice = slice(
                    to_pix(row - side_width / 2), to_pix(row + side_width / 2) + 1
                )
                col_pix_slice = slice(to_pix(col), to_pix(col + forward_width) + 1)
            elif rot == 270:
                col -= 1 / 2
                row_pix_slice = slice(
                    to_pix(row - side_width / 2), to_pix(row + side_width / 2) + 1
                )
                col_pix_slice = slice(to_pix(col - forward_width), to_pix(col) + 1)
            else:
                raise NotImplementedError

            ego_views.append(np.rot90(world[row_pix_slice, col_pix_slice], k=rot // 90))

        if not array_only:
            cv2.imshow("aoeu", np.concatenate(ego_views, axis=0))
            if do_wait:
                return str(chr(cv2.waitKey(wait_key) & 255))
            else:
                cv2.waitKey(100)

        return ego_views

    def Initialize(self, action):
        self.agent_count = action["agentCount"]
        for i in range(self.agent_count):
            self.agents.append(
                AgentObject(
                    self,
                    len(self.agents),
                    min_steps_between_agents=self.min_steps_between_agents,
                )
            )

        return (True, None)

    def GetReachablePositions(self, action):
        # We don't need this for now
        # If we have to use this:
        # The usage concurs to "currently reachable" agent positions
        # Initially reachable points can be accessed by the object variable
        # current_reachable_pos: List[Dict[str, float]] = []
        # for p in np.argwhere(self.agent_reachable_positions_mask):
        #     x, z = self._rowcol_to_xz((p[0], p[1]))
        #     current_reachable_pos.append(
        #         {
        #             "x": x,
        #             "z": z
        #         }
        #     )
        # return (True, current_reachable_pos)
        raise NotImplementedError

    def GetCurrentReachablePositionsSet(self, action):
        # More efficient than GetReachablePositions as it directly gives
        # sets instead of a list of dict and then convert to set

        # The usage concurs to "currently reachable" agent positions
        # Initially reachable points can be accessed by the object variable
        current_reachable_pos_set: Set[Tuple[float, float]] = set()
        for p in np.argwhere(self.agent_reachable_positions_mask):
            x, z = self.rowcol_to_xz((p[0], p[1]))
            current_reachable_pos_set.add((x, z))
        return (True, current_reachable_pos_set)

    def TeleportFull(self, action):
        agent = self.agents[action["agentId"]]

        row, col = self.xz_to_rowcol((action["x"], action["z"]))

        if action.get("makeReachable"):
            self.agent_reachable_positions_mask[row, col] = True

        if (not action.get("forceAction")) and (
            not agent.is_valid_new_position(
                new_row=row,
                new_col=col,
                additional_mask=self.current_near_lifted_object_mask()
                if self.lifted_object is not None
                else None,
                allow_agent_intersection=False
                if "allowAgentIntersection" not in action
                else action["allowAgentIntersection"],
            )
        ):
            return False, None

        agent.row, agent.col = row, col
        agent.rot = action["rotation"]["y"]
        return True, None

    def TeleportObject(self, action):
        # Updates and sets the lifted object position and agents' location too

        objectId = action["objectId"]
        obj = self.tracked_objects[objectId]

        to_row, to_col = self.xz_to_rowcol((action["x"], action["z"]))
        to_rot = 90 * (round(action["rotation"]["y"] / 90) % 4)

        old_rot = obj.rot
        obj.rot = to_rot

        # TODO: Assertion is only necessary as objects do not currently
        #  store their own masks
        assert objectId == self.lifted_object_id
        if self._move_object(
            obj=obj,
            delta={"row": to_row - obj.row, "col": to_col - obj.col},
            valid_mask=self.rotation_to_lifted_object_reachable_position_masks[
                int(self.lifted_object.rot)
            ],
            skip_valid_check=action["forceAction"]
            if "forceAction" in action
            else False,
        ):
            return True, None
        else:
            obj.rot = old_rot
            return False, None

    def CreateLiftedFurnitureAtLocation(self, action):
        # Updates and sets the lifted object position and agents' location too

        assert action["objectType"] == self.lifted_object_type

        rowcol = self.xz_to_rowcol((action["x"], action["z"]))

        rot = 90 * (round(action["rotation"]["y"] / 90) % 4)

        valid_position = self.rotation_to_lifted_object_reachable_position_masks[rot][
            rowcol[0], rowcol[1]
        ]
        if not valid_position:
            if action.get("forceAction"):
                self.rotation_to_lifted_object_reachable_position_masks[rot][
                    rowcol[0], rowcol[1]
                ] = True
            else:
                return False, None

        self.lifted_object = TrackedObject(
            self, self.lifted_object_id, self.lifted_object_type
        )
        row, col = rowcol
        self.lifted_object.row = row
        self.lifted_object.col = col
        self.lifted_object.rot = rot

        self.tracked_objects[self.lifted_object_id] = self.lifted_object

        if not all(
            agent.is_valid_new_position(
                new_row=agent.row,
                new_col=agent.col,
                additional_mask=self.current_near_lifted_object_mask(),
                allow_agent_intersection=True,
            )
            for agent in self.agents
        ):
            self.lifted_object = None
            del self.tracked_objects[self.lifted_object_id]
            return False, None

        return (True, self.lifted_object_id)

    def RandomlyCreateLiftedFurniture(self, action):
        # Updates and sets the lifted object position and agents' location too

        assert action["objectType"] == self.lifted_object_type

        # pick random reachable spot in object_reachable_pos
        # random.seed(0)
        for i in range(10):
            agent_points = []
            point = random.choice(
                # np.argwhere(self.rotation_to_lifted_object_reachable_position_masks[rotation])
                np.argwhere(self.agent_reachable_positions_mask)
            )
            possible_rotations = [
                rot
                for rot in self.rotation_to_lifted_object_reachable_position_masks
                if self.rotation_to_lifted_object_reachable_position_masks[rot][
                    point[0], point[1]
                ]
            ]
            if len(possible_rotations) == 0:
                continue
            rotation = random.choice(possible_rotations)

            self.lifted_object = TrackedObject(
                self, self.lifted_object_id, self.lifted_object_type
            )
            row, col = point
            self.lifted_object.row = row
            self.lifted_object.col = col
            self.lifted_object.rot = rotation
            self.tracked_objects[self.lifted_object_id] = self.lifted_object

            current_state = self.empty_mask()
            object_mask = self.lifted_object_template == 1
            interactable_positions = self.lifted_object_template == 2
            rotations = int((360 - rotation) / 90)
            if rotations < 4:
                object_mask = np.rot90(object_mask, k=rotations)
                interactable_positions = np.rot90(interactable_positions, k=rotations)

            mask_buffer_row, mask_buffer_col = (
                object_mask.shape[0] // 2,
                object_mask.shape[1] // 2,
            )

            rlow, rhigh = row - mask_buffer_row, row + mask_buffer_row + 1
            clow, chigh = col - mask_buffer_col, col + mask_buffer_col + 1

            rlowdelta, rhighdelta = (
                max(-rlow, 0),
                max(rhigh - current_state.shape[0], 0),
            )
            clowdelta, chighdelta = (
                max(-clow, 0),
                max(chigh - current_state.shape[1], 0),
            )
            current_state[
                rlow + rlowdelta : rhigh - rhighdelta,
                clow + clowdelta : chigh - chighdelta,
            ] = interactable_positions[
                rlowdelta : interactable_positions.shape[0] - rhighdelta,
                clowdelta : interactable_positions.shape[1] - chighdelta,
            ]
            current_state &= self.agent_reachable_positions_mask

            agent_points = []
            if self.min_steps_between_agents == 1:
                agent_points = random.sample(
                    list(np.argwhere(current_state)), k=self.agent_count
                )

                # XXX need to retry if we can't put the agent in a location
                if len(agent_points) != self.agent_count:
                    continue
            else:
                rows_and_cols = np.argwhere(current_state)
                if len(rows_and_cols) < self.agent_count:
                    continue

                np.random.shuffle(rows_and_cols)

                for count, items in enumerate(
                    itertools.combinations(
                        list(range(rows_and_cols.shape[0])), self.agent_count
                    )
                ):
                    if count > 100:
                        break

                    subset: np.ndarray = rows_and_cols[items, :]
                    diag = np.expand_dims(
                        np.diag([self.min_steps_between_agents] * self.agent_count), -1
                    )
                    if (
                        diag
                        + np.abs(subset.reshape(-1, 1, 2) - subset.reshape(1, -1, 2))
                    ).max(-1).min() >= self.min_steps_between_agents:
                        np.random.shuffle(subset)
                        agent_points = subset
                        break

                if len(agent_points) != self.agent_count:
                    break

        if len(agent_points) != self.agent_count:
            return (False, None)

        for i, agent in enumerate(self.agents):
            agent.row = agent_points[i][0]
            agent.col = agent_points[i][1]
            if random.random() < 0.5:
                if agent.row > self.lifted_object.row:
                    agent.rot = 0
                elif agent.row < self.lifted_object.row:
                    agent.rot = 180
                else:
                    agent.rot = random.choice([0, 180])
            else:
                if agent.col < self.lifted_object.col:
                    agent.rot = 90
                elif agent.col > self.lifted_object.col:
                    agent.rot = 270
                else:
                    agent.rot = random.choice([90, 270])

        return (True, self.lifted_object_id)

    def GetReachablePositionsForObject(self, action):
        assert action["objectId"] == self.lifted_object_id
        return (True, self.rotation_to_lifted_object_reachable_pos)

    def MoveAhead(self, action):
        return self._move_agent(action, 0)

    def _move_object(self, obj, delta, valid_mask, skip_valid_check=False):
        if skip_valid_check or obj.is_valid_new_position(
            obj.row + delta["row"], obj.col + delta["col"], valid_mask
        ):
            obj.row += delta["row"]
            obj.col += delta["col"]
            return True
        else:
            return False

    def _move_agents_with_lifted(self, action, r):
        assert action["objectId"] == self.lifted_object_id

        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        obj = self.lifted_object
        next_obj_z = obj.row + delta["row"]
        next_obj_x = obj.col + delta["col"]
        success = True
        if obj.is_valid_new_position(
            next_obj_z,
            next_obj_x,
            self.rotation_to_lifted_object_reachable_position_masks[
                int(self.lifted_object.rot)
            ],
        ):
            imask = self.current_near_lifted_object_mask(row=next_obj_z, col=next_obj_x)
            for a in self.agents:
                if not a.is_valid_new_position(
                    a.row + delta["row"],
                    a.col + delta["col"],
                    imask,
                    allow_agent_intersection=True,
                ):
                    success = False
                    break
        else:
            success = False

        if success:
            assert self._move_object(
                self.lifted_object,
                delta,
                self.rotation_to_lifted_object_reachable_position_masks[
                    int(self.lifted_object.rot)
                ],
                skip_valid_check=True,
            )
            for a in self.agents:
                assert self._move_object(
                    a,
                    delta,
                    self.current_near_lifted_object_mask(),
                    skip_valid_check=True,
                )

        return (success, None)

    def _move_lifted(self, action, r):
        assert action["objectId"] == self.lifted_object_id

        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        obj = self.lifted_object
        next_obj_z = obj.row + delta["row"]
        next_obj_x = obj.col + delta["col"]
        success = True
        if obj.is_valid_new_position(
            next_obj_z,
            next_obj_x,
            self.rotation_to_lifted_object_reachable_position_masks[
                int(self.lifted_object.rot)
            ],
        ):
            imask = self.current_near_lifted_object_mask(row=next_obj_z, col=next_obj_x)
            for a in self.agents:
                if not imask[a.row, a.col]:
                    success = False
        else:
            success = False

        if success:
            self._move_object(
                self.lifted_object,
                delta,
                self.rotation_to_lifted_object_reachable_position_masks[
                    int(self.lifted_object.rot)
                ],
            )

        return (success, None)

    def _move_agent(self, action, r):
        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        success = self._move_object(
            agent, delta, self.current_near_lifted_object_mask()
        )
        return (success, None)

    def MoveLeft(self, action):
        return self._move_agent(action, -90)

    def MoveRight(self, action):
        return self._move_agent(action, 90)

    def MoveBack(self, action):
        return self._move_agent(action, 180)

    def RotateRight(self, action):
        agent = self.agents[action["agentId"]]
        agent.rot = (agent.rot + 90) % 360
        return (True, None)

    def RotateLeft(self, action):
        agent = self.agents[action["agentId"]]
        agent.rot = (agent.rot - 90) % 360
        return (True, None)

    def current_near_lifted_object_mask(self, row=None, col=None, rotation=None):
        if rotation is None:
            rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        interactable_mask = self.lifted_object_template == 2
        interactable_mask = np.rot90(interactable_mask, k=rotations)

        if col is None or row is None:
            row = self.lifted_object.row
            col = self.lifted_object.col

        mask_buffer_row, mask_buffer_col = (
            interactable_mask.shape[0] // 2,
            interactable_mask.shape[1] // 2,
        )

        current_state = self.empty_mask()

        rlow, rhigh = row - mask_buffer_row, row + mask_buffer_row + 1
        clow, chigh = col - mask_buffer_col, col + mask_buffer_col + 1

        rlowdelta, rhighdelta = (
            max(-rlow, 0),
            max(rhigh - current_state.shape[0], 0),
        )
        clowdelta, chighdelta = max(-clow, 0), max(chigh - current_state.shape[1], 0)
        current_state[
            rlow + rlowdelta : rhigh - rhighdelta,
            clow + clowdelta : chigh - chighdelta,
        ] = interactable_mask[
            rlowdelta : interactable_mask.shape[0] - rhighdelta,
            clowdelta : interactable_mask.shape[1] - chighdelta,
        ]

        return current_state

    def current_lifted_object_mask(self):
        rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        object_mask = self.lifted_object_template == 1
        object_mask = np.rot90(object_mask, k=rotations)

        row = self.lifted_object.row
        col = self.lifted_object.col
        mask_buffer_row, mask_buffer_col = (
            object_mask.shape[0] // 2,
            object_mask.shape[1] // 2,
        )

        current_state = self.empty_mask()

        rlow, rhigh = row - mask_buffer_row, row + mask_buffer_row + 1
        clow, chigh = col - mask_buffer_col, col + mask_buffer_col + 1

        rlowdelta, rhighdelta = (
            max(-rlow, 0),
            max(rhigh - current_state.shape[0], 0),
        )
        clowdelta, chighdelta = max(-clow, 0), max(chigh - current_state.shape[1], 0)
        current_state[
            rlow + rlowdelta : rhigh - rhighdelta,
            clow + clowdelta : chigh - chighdelta,
        ] = object_mask[
            rlowdelta : object_mask.shape[0] - rhighdelta,
            clowdelta : object_mask.shape[1] - chighdelta,
        ]

        return current_state

    def _rotate_lifted(self, new_rotation):
        if not self.rotation_to_lifted_object_reachable_position_masks[new_rotation][
            self.lifted_object.row, self.lifted_object.col
        ]:
            self._error_message = (
                "Lifted object colliding with non-agent after rotation."
            )
            return False

        imask = self.current_near_lifted_object_mask(rotation=new_rotation)
        for a in self.agents:
            if not imask[a.row, a.col]:
                self._error_message = (
                    "Lifted object colliding with agent after rotation."
                )
                return False

        self.lifted_object.rot = new_rotation
        return True

    def agent_inside_range(
        self, agent_id, top_row, bottom_row, left_column, right_column
    ):
        row = self.agents[agent_id].row
        col = self.agents[agent_id].col
        return top_row <= row <= bottom_row and left_column <= col <= right_column

    def CreateAndPlaceObjectOnFloorAtLocation(self, action):
        # Places object on the floor, but also updates the
        # agent_reachable_positions_mask
        object_mask = action["object_mask"]
        object_type = action["objectType"]
        force_action = False if "forceAction" not in action else action["forceAction"]

        rot = 90 * (round(action["rotation"]["y"] / 90) % 4)
        mask = np.rot90(object_mask, k=-(rot // 90))

        row, col = self.xz_to_rowcol((action["x"], action["z"]))

        row_rad, col_rad = mask.shape[0] // 2, mask.shape[1] // 2
        reachable_subset = self.agent_reachable_positions_mask[
            row - row_rad : row + row_rad + 1, col - col_rad : col + col_rad + 1
        ]
        if force_action or (np.logical_and(reachable_subset, mask) == mask).all():
            if (not force_action) and np.any(
                [
                    self.agent_inside_range(
                        agent_id,
                        row - row_rad,
                        row + row_rad,
                        col - col_rad,
                        col + col_rad,
                    )
                    for agent_id in range(len(self.agents))
                ]
            ):
                # TODO: THIS CURRENTLY ONLY WORKS FOR RECTANGULARLY SHAPED OBJECTS
                return False, None

            # update the agent_reachable_positions_mask
            self.agent_reachable_positions_mask[
                row - row_rad : row + row_rad + 1, col - col_rad : col + col_rad + 1
            ] &= np.logical_not(mask)
            xz = self.rowcol_to_xz((row, col))

            object_id = object_type + "|{}".format(len(self.tracked_objects) + 1)
            floor_object = TrackedObject(self, object_id, object_type)
            floor_object.row = row
            floor_object.col = col
            floor_object.rot = rot
            floor_object.object_mask = mask
            self.tracked_objects[object_id] = floor_object

            return (
                True,
                {
                    "position": {"x": xz[0], "y": math.nan, "z": xz[1]},
                    "row": floor_object.row,
                    "col": floor_object.col,
                    "rotation": floor_object.rot,
                    "objectId": object_id,
                },
            )
        return False, None

    def RandomlyCreateAndPlaceObjectOnFloor(self, action):
        # Places object on the floor, but also updates the
        # agent_reachable_positions_mask
        object_mask = action["object_mask"]
        object_type = action["objectType"]

        object_masks = [(k, np.rot90(object_mask, k=k)) for k in range(4)]

        positions = np.argwhere(self.agent_reachable_positions_mask)
        for i in np.random.permutation(positions.shape[0]):
            row, col = positions[i]
            random.shuffle(object_masks)
            for k, mask in object_masks:
                row_rad, col_rad = mask.shape[0] // 2, mask.shape[1] // 2
                reachable_subset = self.agent_reachable_positions_mask[
                    row - row_rad : row + row_rad + 1, col - col_rad : col + col_rad + 1
                ]
                if (np.logical_and(reachable_subset, mask) == mask).all():
                    if np.any(
                        [
                            self.agent_inside_range(
                                agent_id,
                                row - row_rad,
                                row + row_rad,
                                col - col_rad,
                                col + col_rad,
                            )
                            for agent_id in range(len(self.agents))
                        ]
                    ):
                        continue

                    # update the agent_reachable_positions_mask
                    self.agent_reachable_positions_mask[
                        row - row_rad : row + row_rad + 1,
                        col - col_rad : col + col_rad + 1,
                    ] &= np.logical_not(mask)
                    xz = self.rowcol_to_xz((row, col))

                    object_id = object_type + "|{}".format(
                        len(self.tracked_objects) + 1
                    )
                    floor_object = TrackedObject(self, object_id, object_type)
                    floor_object.row = row
                    floor_object.col = col
                    floor_object.rot = 90 * k
                    floor_object.object_mask = mask
                    self.tracked_objects[object_id] = floor_object

                    return (
                        True,
                        {
                            "position": {"x": xz[0], "y": math.nan, "z": xz[1]},
                            "row": row,
                            "col": col,
                            "rotation": 90 * k,
                            "objectId": object_id,
                        },
                    )
        return False, None

    def RotateLiftedObjectLeft(self, action):
        new_rotation = (self.lifted_object.rot - 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def RotateLiftedObjectRight(self, action):
        new_rotation = (self.lifted_object.rot + 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def MoveAgentsRightWithObject(self, action):
        return self._move_agents_with_lifted(action, 90)

    def MoveAgentsAheadWithObject(self, action):
        return self._move_agents_with_lifted(action, 0)

    def MoveAgentsBackWithObject(self, action):
        return self._move_agents_with_lifted(action, 180)

    def MoveAgentsLeftWithObject(self, action):
        return self._move_agents_with_lifted(action, -90)

    def MoveLiftedObjectRight(self, action):
        return self._move_lifted(action, 90)

    def MoveLiftedObjectAhead(self, action):
        return self._move_lifted(action, 0)

    def MoveLiftedObjectBack(self, action):
        return self._move_lifted(action, 180)

    def MoveLiftedObjectLeft(self, action):
        return self._move_lifted(action, -90)

    def Pass(self, action):
        return (True, None)

    def step(self, action, raise_for_failure=False):
        self.steps_taken += 1
        # XXX should have invalid action
        # print("running method %s" % action)
        method = getattr(self, action["action"])
        success, result = method(action)
        events = []
        for a in self.agents:
            events.append(
                Event(
                    self._generate_metadata(
                        a, self.lifted_object, result, action, success
                    )
                )
            )

        self.last_event = MultiAgentEvent(
            action.get("agentId") if "agentId" in action else 0, events
        )
        self.last_event.metadata["errorMessage"] = self._error_message
        self._error_message = ""
        return self.last_event

    def _generate_metadata(
        self,
        agent: "AgentObject",
        lifted_object: "GridObject",
        result: Any,
        action: str,
        success: bool,
    ):
        metadata = dict()
        metadata["agent"] = dict(position=agent.position, rotation=agent.rotation)
        metadata["objects"] = []
        if len(self.tracked_objects) > 0:
            for object_id, tracked_object in self.tracked_objects.items():
                metadata["objects"].append(
                    dict(
                        position=tracked_object.position,
                        rotation=tracked_object.rotation,
                        objectType=tracked_object.object_type,
                        objectId=tracked_object.object_id,
                    )
                )
        metadata["actionReturn"] = result
        metadata["lastAction"] = action["action"]
        metadata["lastActionSuccess"] = success
        metadata["sceneName"] = self.scene_name
        metadata["screenHeight"] = 300
        metadata["screenWidth"] = 300
        metadata["colors"] = []

        return metadata


class GridObject(object):
    def __init__(self, controller):
        self.controller = controller
        self.col = 0
        self.row = 0
        self.rot = 0.0

    @property
    def position(self):
        cx = (self.col * self.controller.grid_size) + self.controller.min_x
        cz = (-self.row * self.controller.grid_size) + self.controller.max_z
        return dict(x=cx, y=1.0, z=cz)

    @property
    def x(self):
        return (self.col * self.controller.grid_size) + self.controller.min_x

    @property
    def z(self):
        return (-self.row * self.controller.grid_size) + self.controller.max_z

    @property
    def rotation(self):
        return dict(x=0.0, y=self.rot, z=0.0)


class AgentObject(GridObject):
    def __init__(
        self,
        controller: GridWorldController,
        agent_id,
        min_steps_between_agents: int = 1,
    ):
        super().__init__(controller)
        self.agent_id = agent_id
        self.min_steps_between_agents = min_steps_between_agents

    def is_valid_new_position(
        self, new_row, new_col, additional_mask, allow_agent_intersection=False
    ):
        if additional_mask is not None:
            additional_mask &= self.controller.agent_reachable_positions_mask
        else:
            additional_mask = copy.deepcopy(
                self.controller.agent_reachable_positions_mask
            )

        # mark spots occupied by agents as False
        if not allow_agent_intersection:
            for a in self.controller.agents:
                if a is not self:
                    d = self.min_steps_between_agents
                    additional_mask[
                        (a.row - d + 1) : (a.row + d), (a.col - d + 1) : (a.col + d)
                    ] = False

        return additional_mask[new_row, new_col]


class TrackedObject(GridObject):
    def __init__(self, controller, object_id, object_type):
        super().__init__(controller)
        self.object_id = object_id
        self.object_type = object_type

    def is_valid_new_position(self, new_row, new_col, additional_mask):
        assert additional_mask is not None
        return additional_mask[new_row, new_col]


def run_demo(controller: GridWorldController):
    key_to_action = {
        "w": "MoveAhead",
        "a": "MoveLeft",
        "s": "MoveBack",
        "d": "MoveRight",
        "z": "RotateLeft",
        "x": "RotateRight",
        "i": "MoveAgentsAheadWithObject",
        "j": "MoveAgentsLeftWithObject",
        "k": "MoveAgentsBackWithObject",
        "l": "MoveAgentsRightWithObject",
        "m": "RotateLiftedObjectLeft",
        ",": "RotateLiftedObjectRight",
        "t": "MoveLiftedObjectAhead",
        "f": "MoveLiftedObjectLeft",
        "g": "MoveLiftedObjectBack",
        "h": "MoveLiftedObjectRight",
    }

    print("Controls:")
    print("'q':\tQuit")
    print("'0' ('1', '2', ...):\tChange to controlling agent 0.")

    for k in "wasdzxijklm,tfgh":
        print(
            "'{}':\t{}".format(
                k,
                " ".join(
                    [
                        word.lower() if i != 0 else word
                        for i, word in enumerate(
                            re.split("([A-Z][^A-Z]*)", key_to_action[k])
                        )
                        if word != ""
                    ]
                ),
            )
        )

    controlling_agent_id = 0
    trying_to_quit = False
    while True:
        c = controller.viz_world()
        # controller.viz_ego_agent_views()

        if c in ["0", "1", "2", "3", "4", "5", "6"]:
            trying_to_quit = False
            controlling_agent_id = int(c)
            print("Switched to agent {}".format(c))
        elif c == "q":
            print("Are you sure you wish to exit the demo? (y/n)")
            trying_to_quit = True
        elif trying_to_quit and c == "y":
            return
        elif c in key_to_action:
            trying_to_quit = False
            controller.step(
                {
                    "action": key_to_action[c],
                    "agentId": controlling_agent_id,
                    "objectId": "Television|1",
                }
            )
            print(
                "Taking action {}\nAction {}\n".format(
                    key_to_action[c],
                    "success"
                    if controller.last_event.metadata["lastActionSuccess"]
                    else "failure",
                )
            )

            for agent_id, agent in enumerate(controller.agents):
                print("Agent {} position".format(agent_id), agent.position)
            print("Object position", controller.lifted_object.position)
            print("")
        else:
            trying_to_quit = False
            print('Invalid key "{}"'.format(c))


class AI2ThorLiftedObjectGridEnvironment(object):
    def __init__(
        self,
        lifted_object_height: float,
        max_dist_to_lifted_object: float,
        object_type: str,
        min_steps_between_agents: int = 1,
        docker_enabled: bool = False,
        x_display: str = None,
        local_thor_build: str = None,
        time_scale: float = 1.0,
        visibility_distance: float = constants.VISIBILITY_DISTANCE,
        fov: float = constants.FOV,
        num_agents: int = 1,
        visible_agents: bool = True,
        headless: bool = True,
        remove_unconnected_positions: bool = False,
    ) -> None:
        assert object_type == "Television"

        self.lifted_object_height = lifted_object_height
        self.max_dist_to_lifted_object = max_dist_to_lifted_object
        self.object_type = object_type

        self.num_agents = num_agents
        self._thor_env = AI2ThorEnvironment(
            docker_enabled=docker_enabled,
            x_display=x_display,
            local_thor_build=local_thor_build,
            time_scale=time_scale,
            visibility_distance=visibility_distance,
            fov=fov,
            restrict_to_initially_reachable_points=True,
            num_agents=1,
            visible_agents=True,
            headless=headless,
        )

        self.controller: Optional[GridWorldController] = None
        self.x_display = x_display
        self._started = False
        self.move_mag: Optional[float] = None
        self.grid_size: Optional[float] = None
        self.time_scale = time_scale
        self.visibility_distance = visibility_distance
        self.visible_agents = visible_agents
        self.headless = headless
        self.min_steps_between_agents = min_steps_between_agents
        self.remove_unconnected_positions = remove_unconnected_positions

        self.cached_initially_reachable_positions = {}
        self.cached_rotation_to_lifted_object_reachable_positions = {}
        self.cached_controller = {}
        self.object_unreachable_silhouette_template_string = None

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
    def last_event(self) -> ai2thor.server.Event:
        return self.controller.last_event

    @property
    def started(self) -> bool:
        return self._started

    def start(
        self,
        scene_name: Optional[str],
        move_mag: float = 0.25,
        player_screen_width=128,
        player_screen_height=128,
        quality="Very Low",
    ) -> None:
        self._thor_env.start(
            scene_name=scene_name,
            move_mag=move_mag,
            player_screen_width=player_screen_width,
            player_screen_height=player_screen_height,
            quality=quality,
        )

        self._started = True
        self.reset(scene_name=scene_name, move_mag=move_mag)

    def stop(self) -> None:
        try:
            self._thor_env.stop()
        except Exception as e:
            warnings.warn(str(e))
        finally:
            self._started = False

    def reset(self, scene_name: Optional[str], move_mag: float = 0.25):
        # While called to 'RandomlyCreateLiftedFurniture' are made via the
        # thor_env, it is only to obtain the reachable locations. For the
        # gridworld environment the lifted object is not placed, nor are the
        # agents places. That is done by executing:
        # env.controller.step({action: "RandomlyCreateLiftedFurniture}

        assert move_mag == 0.25

        self.move_mag = move_mag
        self.grid_size = self.move_mag

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]

        if scene_name not in self.cached_initially_reachable_positions:
            self._thor_env.reset(scene_name, move_mag=move_mag)

            self.cached_initially_reachable_positions[scene_name] = copy.deepcopy(
                self._thor_env.initially_reachable_points
            )

            self._thor_env.step(
                dict(
                    action="RandomlyCreateLiftedFurniture",
                    objectType=self.object_type,
                    objectVariation=1,
                    y=self.lifted_object_height,
                    z=self.max_dist_to_lifted_object,
                )
            )
            assert self._thor_env.last_event.metadata["lastActionSuccess"]
            object_id = self._thor_env.last_event.metadata["actionReturn"]

            self._thor_env.step(
                {
                    "action": "GetReachablePositionsForObject",
                    "objectId": object_id,
                    "positions": self.cached_initially_reachable_positions[scene_name],
                }
            )
            assert self._thor_env.last_event.metadata["lastActionSuccess"]

            self.cached_rotation_to_lifted_object_reachable_positions[
                scene_name
            ] = self._thor_env.last_event.metadata["actionReturn"]

            if self.object_unreachable_silhouette_template_string is None:
                self._thor_env.step(
                    {
                        "action": "GetUnreachableSilhouetteForObject",
                        "objectId": object_id,
                        "z": self.max_dist_to_lifted_object,
                    }
                )
                assert self._thor_env.last_event.metadata["lastActionSuccess"]
                self.object_unreachable_silhouette_template_string = self._thor_env.last_event.metadata[
                    "actionReturn"
                ]

                # Remove rows / cols where both the object isn't present and the agents' cannot go
                mat = [
                    l.strip().split(" ")
                    for l in self.object_unreachable_silhouette_template_string.strip().split(
                        "\n"
                    )
                ]
                any_removed = True
                while any_removed:
                    any_removed = False

                    if all(mat[0][i] == "0" for i in range(len(mat[0]))) and all(
                        mat[-1][i] == "0" for i in range(len(mat[0]))
                    ):
                        any_removed = True
                        mat.pop(0)
                        mat.pop(-1)

                    if all(mat[i][0] == "0" for i in range(len(mat))) and all(
                        mat[i][-1] == "0" for i in range(len(mat))
                    ):
                        any_removed = True
                        for l in mat:
                            l.pop(0)
                            l.pop(-1)

                assert len(mat) % 2 == 1 and len(mat[0]) % 2 == 1

                self.object_unreachable_silhouette_template_string = "\n".join(
                    [" ".join(l) for l in mat]
                )

            self.cached_controller[scene_name] = copy.deepcopy(
                GridWorldController(
                    agent_initially_reachable_pos=self.cached_initially_reachable_positions[
                        scene_name
                    ],
                    rotation_to_lifted_object_reachable_pos=self.cached_rotation_to_lifted_object_reachable_positions[
                        scene_name
                    ],
                    lifted_object_id=self.object_type + "|1",
                    lifted_object_type=self.object_type,
                    min_steps_between_agents=self.min_steps_between_agents,
                    grid_size=self.grid_size,
                    object_template_string=self.object_unreachable_silhouette_template_string,
                    scene_name=scene_name,
                    remove_unconnected_positions=self.remove_unconnected_positions,
                )
            )

        self.controller = copy.deepcopy(self.cached_controller[scene_name])

        self.controller.step({"action": "Initialize", "agentCount": self.num_agents})

    @property
    def initially_reachable_points(self) -> Tuple[Dict[str, float]]:
        return self.controller.agent_initially_reachable_pos_tuple  # type:ignore

    def get_current_multi_agent_occupancy_tensors(
        self, use_initially_reachable_points_matrix: bool = False
    ) -> Tuple[List[np.ndarray], Dict[Tuple, Tuple]]:
        # Padding is already incorporated in the controller level
        # Check if that needs to improved.
        # The reachable tensor is as index 0
        point_to_element_map = dict()
        if use_initially_reachable_points_matrix:
            reachable_tensor = np.expand_dims(
                copy.deepcopy(self.controller.agent_initially_reachable_positions_mask),
                axis=0,
            )
            for point in self.controller.agent_initially_reachable_pos_tuple:
                xz = (point["x"], point["z"])
                point_to_element_map[xz] = self.controller.xz_to_rowcol(xz)
        else:
            reachable_tensor = np.expand_dims(
                copy.deepcopy(self.controller.agent_reachable_positions_mask), axis=0
            )
            for point in self.controller.agent_initially_reachable_pos_tuple:
                xz = (point["x"], point["z"])
                point_to_element_map[xz] = self.controller.xz_to_rowcol(xz)
            for point in sum(
                self.controller.rotation_to_lifted_object_reachable_pos.values(), []
            ):
                xz = (point["x"], point["z"])
                point_to_element_map[xz] = self.controller.xz_to_rowcol(xz)
        # 0/1 reachable point matrix

        positions_tensors = [
            np.zeros((4 * self.num_agents, *reachable_tensor.shape[-2:]), dtype=float)
            for _ in range(self.num_agents)
        ]
        for i in range(self.num_agents):
            agent_location_in_grid = self.get_agent_location_in_mask(agent_id=i)
            # This is now in sync with the quantization done in visual hull processing
            clock_90 = round(agent_location_in_grid["rot"] / 90) % 4
            rowcol_val = (agent_location_in_grid["row"], agent_location_in_grid["col"])

            for j in range(self.num_agents):
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
        }
        return location

    def get_agent_location_in_mask(self, agent_id: int = None) -> Dict[str, float]:
        if self.num_agents <= 1:
            agent_id = 0
        a = self.controller.agents[agent_id]
        return {"row": a.row, "col": a.col, "rot": a.rot}

    def _agent_location_to_tuple(self, p: Dict[str, float]):
        return (round(p["x"], 2), round(p["z"], 2))

    def get_agent_locations(self) -> Tuple[Dict[str, float], ...]:
        """Gets all agents' locations."""
        return tuple(self.get_agent_location(i) for i in range(self.num_agents))

    def get_agent_metadata(self, agent_id: int = 0) -> Dict[str, Any]:
        """Gets agent's metadata."""
        return self.controller.last_event.events[agent_id].metadata["agent"]

    def get_all_agent_metadata(self) -> Tuple[Dict[str, Any], ...]:
        """Gets all agents' locations."""
        return tuple(self.get_agent_metadata(i) for i in range(self.num_agents))

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

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        action = action_dict["action"]
        agent_id = action_dict.get("agentId")

        if agent_id is not None:
            assert type(agent_id) == int and 0 <= agent_id <= self.num_agents
        else:
            assert self.num_agents == 1
            action_dict["agentId"] = 0

        return self.controller.step(action_dict)

    def visualize(self, wait_key):
        self.controller.viz_world(wait_key)


if __name__ == "__main__":
    import random
    import constants

    env = AI2ThorLiftedObjectGridEnvironment(
        lifted_object_height=1.3,
        max_dist_to_lifted_object=1,
        min_steps_between_agents=2,
        object_type="Television",
        num_agents=2,
        local_thor_build=constants.ABS_PATH_TO_LOCAL_THOR_BUILD,
        headless=False,
        remove_unconnected_positions=True,
    )

    scenes = constants.TRAIN_SCENE_NAMES[20:40]
    env.start(scenes[0], player_screen_height=500, player_screen_width=500)

    dresser_silhouette_string = """1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    """

    dresser_mask = env.controller.parse_template_to_mask(dresser_silhouette_string)

    # Run demo
    while True:
        env.reset(random.choice(scenes))
        env.step(
            {
                "action": "RandomlyCreateAndPlaceObjectOnFloor",
                "agentId": 0,
                "object_mask": dresser_mask,
                "objectType": "Dresser",
            }
        )
        assert env.last_event.metadata["lastActionSuccess"]
        # print(env.last_event.metadata["actionReturn"])

        env.step(
            {
                "action": "RandomlyCreateLiftedFurniture",
                "objectType": "Television",
                "agentId": 0,
            }
        )
        for i in range(10):
            if not env.last_event.metadata["lastActionSuccess"]:
                env.step(
                    {
                        "action": "RandomlyCreateLiftedFurniture",
                        "objectType": "Television",
                        "agentId": 0,
                    }
                )

        run_demo(env.controller)
        # print(env.controller.steps_taken)
        env.cached_controller.clear()
        env.cached_initially_reachable_positions.clear()
        env.cached_rotation_to_lifted_object_reachable_positions.clear()
