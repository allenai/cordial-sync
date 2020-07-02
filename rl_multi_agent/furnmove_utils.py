import glob
import inspect
import json
import math
import os
import time
import warnings
from collections import defaultdict
from typing import List, Dict, Any

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

from constants import ABS_PATH_TO_ANALYSIS_RESULTS_DIR, ABS_PATH_TO_DATA_DIR
from rl_ai2thor.ai2thor_episodes import MultiAgentAI2ThorEpisode
from rl_base import A3CAgent
from rl_multi_agent import MultiAgent
from utils.visualization_utils import get_agent_map_data, visualize_agent_path


def step_results_to_data_to_save_for_furnmove_experiment(
    episode: MultiAgentAI2ThorEpisode, step_results: List[List[Dict]]
):
    if len(step_results) == 0:
        return {}

    episode_info = episode.info()
    nagents = len(step_results[0])
    info = {
        "ep_length": len(step_results) * nagents,
        "final_distance": episode_info["navigation/final_distance"],
        "reached_target": episode_info["navigation/reached_target"],
        "spl_manhattan": episode_info["navigation/spl_manhattan"],
        "initial_manhattan_steps": episode_info["navigation/initial_manhattan_steps"],
    }
    avail_actions = episode.available_actions

    action_taken_matrix = np.zeros((len(avail_actions),) * nagents)
    action_taken_success_matrix = np.zeros((len(avail_actions),) * nagents)

    agent_action_info = [defaultdict(lambda: []) for _ in range(nagents)]
    total_reward = 0
    for srs in step_results:
        action_inds = []
        success_so_far = None

        for agent_id, sr in enumerate(srs):
            agent_info = agent_action_info[agent_id]
            total_reward += sr["reward"]

            action_ind = sr["action"]
            action = avail_actions[sr["action"]]
            action_success = sr["action_success"]
            action_inds.append(action_ind)

            agent_info["action_ind"].append(action_ind)
            agent_info["action"].append(action)
            agent_info["action_success"].append(action_success)

            if "extra_before_info" in sr:
                agent_info["dresser_visible"].append(
                    sr["extra_before_info"]["dresser_visible"]
                )
                agent_info["tv_visible"].append(sr["extra_before_info"]["tv_visible"])

            if success_so_far is None:
                success_so_far = action_success
            elif success_so_far != action_success:
                success_so_far = math.nan

        action_inds = tuple(action_inds)
        if success_so_far is math.nan:
            if all(
                "Pass" != agent_action_info[i]["action"][-1] for i in range(nagents)
            ):
                warnings.warn("One agent was successful while the other was not!")
        elif success_so_far:
            action_taken_success_matrix[action_inds] += 1
        action_taken_matrix[action_inds] += 1

    info["step_results"] = step_results
    info["reward"] = total_reward
    info["agent_action_info"] = [dict(info) for info in agent_action_info]
    info["action_taken_matrix"] = action_taken_matrix.tolist()
    info["action_taken_success_matrix"] = action_taken_success_matrix.tolist()
    return info


class SaveFurnMoveMixin(object):
    def simple_name(self):
        raise NotImplementedError()

    def create_episode_summary(
        self,
        agent: A3CAgent,
        additional_metrics: Dict[str, float],
        step_results: List[List[Dict]],
    ) -> Any:
        data_to_save = step_results_to_data_to_save_for_furnmove_experiment(
            episode=agent.episode, step_results=step_results
        )
        data_to_save["episode_init_data"] = self.init_train_agent.episode_init_data

        if len(agent.eval_results) != 0:
            for i in range(agent.environment.num_agents):
                for k in ["talk_probs", "reply_probs"]:
                    data_to_save["a{}_{}".format(i, k)] = []
                    for er in agent.eval_results:
                        if k in er:
                            data_to_save["a{}_{}".format(i, k)].append(
                                er[k][i].cpu().detach().view(-1).numpy().tolist()
                            )
            if "randomization_probs" in agent.eval_results[0]:
                data_to_save["randomization_probs"] = []
                for er in agent.eval_results:
                    data_to_save["randomization_probs"].append(
                        torch.exp(er["randomization_log_probs"])
                        .cpu()
                        .detach()
                        .view(-1)
                        .numpy()
                        .tolist()
                    )

        return {**data_to_save, **additional_metrics}

    def data_save_dir(self) -> str:
        return os.path.join(ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__{}/{}").format(
            self.episode_init_queue_file_name.split("__")[-1].replace(".json", ""),
            self.simple_name(),
        )

    def save_episode_summary(self, data_to_save: Any):
        self.total_saved += 1

        scene = data_to_save["episode_init_data"]["scene"]
        rep = data_to_save["episode_init_data"]["replication"]
        print(
            "Saving data corresponding to scene {} and replication {}. {}/{} completed.".format(
                scene, rep, self.total_saved, self.total_to_save
            ),
            flush=True,
        )

        write_path = os.path.join(
            self.data_save_dir(), "{}__{}.json".format(scene, rep)
        )

        with open(write_path, "w") as f:
            json.dump(data_to_save, f)

    def create_episode_init_queue(self, mp_module) -> torch.multiprocessing.Queue:
        q = mp_module.Queue()
        self.total_saved = 0

        with open(
            os.path.join(ABS_PATH_TO_DATA_DIR, "{}").format(
                self.episode_init_queue_file_name
            ),
            "r",
        ) as f:
            scene_to_rep_to_ep_data = json.load(f)

        keys_remaining = {
            (scene, rep)
            for scene in scene_to_rep_to_ep_data
            for rep in scene_to_rep_to_ep_data[scene]
        }

        os.makedirs(self.data_save_dir(), exist_ok=True)

        for p in glob.glob(os.path.join(self.data_save_dir(), "*.json")):
            file_name = os.path.basename(p)
            keys_remaining.remove(tuple(file_name.replace(".json", "").split("__")))

        k = 0
        for scene, rep in keys_remaining:
            data = scene_to_rep_to_ep_data[scene][rep]
            data["scene"] = scene
            data["replication"] = rep
            q.put(data)
            k += 1

        print("{} episodes added to queue".format(k))
        self.total_to_save = k

        return q

    def stopping_criteria_reached(self):
        return self.total_saved >= self.total_to_save


def step_results_to_data_to_save_for_furnlift_experiment(
    episode: MultiAgentAI2ThorEpisode, step_results: List[List[Dict]]
):
    if len(step_results) == 0:
        return {}

    episode_info = episode.info()

    nagents = len(step_results[0])
    info = {
        "ep_length": len(step_results) * nagents,
        "picked_but_not_pickupable_distance": episode_info[
            "picked_but_not_pickupable_distance"
        ],
        "picked_but_not_pickupable": episode_info["picked_but_not_pickupable"],
        "picked_but_not_pickupable_visibility": episode_info[
            "picked_but_not_pickupable_visibility"
        ],
        "pickupable_but_not_picked": episode_info["pickupable_but_not_picked"],
        "accuracy": episode_info["accuracy"],
        "final_distance": episode_info["final_manhattan_distance_from_target"],
        "initial_manhattan_steps": episode_info["initial_manhattan_steps"],
        "spl_manhattan": episode_info["spl_manhattan"],
    }
    avail_actions = episode.available_actions

    action_taken_matrix = np.zeros((len(avail_actions),) * nagents)

    agent_action_info = [defaultdict(lambda: []) for _ in range(nagents)]
    total_reward = 0
    for srs in step_results:
        action_inds = []
        success_so_far = None

        for agent_id, sr in enumerate(srs):
            agent_info = agent_action_info[agent_id]
            total_reward += sr["reward"]

            action_ind = sr["action"]
            action = avail_actions[sr["action"]]
            action_success = sr["action_success"]
            action_inds.append(action_ind)

            agent_info["action_ind"].append(action_ind)
            agent_info["action"].append(action)
            agent_info["action_success"].append(action_success)

            if success_so_far is None:
                success_so_far = action_success
            elif success_so_far != action_success:
                success_so_far = math.nan

        action_inds = tuple(action_inds)
        action_taken_matrix[action_inds] += 1

    info["step_results"] = step_results
    info["reward"] = total_reward
    info["agent_action_info"] = [dict(info) for info in agent_action_info]
    info["action_taken_matrix"] = action_taken_matrix.tolist()
    return info


class SaveFurnLiftMixin(object):
    def simple_name(self):
        raise NotImplementedError()

    def create_episode_summary(
        self,
        agent: A3CAgent,
        additional_metrics: Dict[str, float],
        step_results: List[List[Dict]],
    ) -> Any:
        data_to_save = step_results_to_data_to_save_for_furnlift_experiment(
            episode=agent.episode, step_results=step_results
        )
        data_to_save["episode_init_data"] = self.init_train_agent.episode_init_data

        if len(agent.eval_results) != 0:
            for i in range(agent.environment.num_agents):
                for k in ["talk_probs", "reply_probs"]:
                    data_to_save["a{}_{}".format(i, k)] = []
                    for er in agent.eval_results:
                        if k in er:
                            data_to_save["a{}_{}".format(i, k)].append(
                                er[k][i].cpu().detach().view(-1).numpy().tolist()
                            )
            if "randomization_probs" in agent.eval_results[0]:
                data_to_save["randomization_probs"] = []
                for er in agent.eval_results:
                    data_to_save["randomization_probs"].append(
                        torch.exp(er["randomization_log_probs"])
                        .cpu()
                        .detach()
                        .view(-1)
                        .numpy()
                        .tolist()
                    )

        return {**data_to_save, **additional_metrics}

    def data_save_dir(self) -> str:
        unique_dir = os.path.basename(inspect.getfile(type(self))).replace(
            "_config.py", ""
        )
        return os.path.join(
            ABS_PATH_TO_DATA_DIR, "furnlift_evaluations__{}/{}__{}"
        ).format(
            self.episode_init_queue_file_name.split("__")[-1].replace(".json", ""),
            self.simple_name(),
            unique_dir,
        )

    def save_episode_summary(self, data_to_save: Any):
        self.total_saved += 1

        scene = data_to_save["episode_init_data"]["scene"]
        rep = data_to_save["episode_init_data"]["replication"]
        print(
            (
                "Saving data corresponding to scene {} and replication {}."
                " {}/{} completed."
            ).format(scene, rep, self.total_saved, self.total_to_save),
            flush=True,
        )

        write_path = os.path.join(
            self.data_save_dir(), "{}__{}.json".format(scene, rep)
        )

        with open(write_path, "w") as f:
            json.dump(data_to_save, f)

    def create_episode_init_queue(self, mp_module) -> torch.multiprocessing.Queue:
        q = mp_module.Queue()
        self.total_saved = 0

        with open(
            os.path.join(ABS_PATH_TO_DATA_DIR, "{}").format(
                self.episode_init_queue_file_name
            ),
            "r",
        ) as f:
            scene_to_rep_to_ep_data = json.load(f)

        keys_remaining = {
            (scene, rep)
            for scene in scene_to_rep_to_ep_data
            for rep in scene_to_rep_to_ep_data[scene]
        }

        os.makedirs(self.data_save_dir(), exist_ok=True)

        for p in glob.glob(os.path.join(self.data_save_dir(), "*.json")):
            file_name = os.path.basename(p)
            keys_remaining.remove(tuple(file_name.replace(".json", "").split("__")))

        k = 0
        for scene, rep in keys_remaining:
            data = scene_to_rep_to_ep_data[scene][rep]
            data["scene"] = scene
            data["replication"] = rep
            q.put(data)
            k += 1

        print("{} episodes added to queue".format(k))
        self.total_to_save = k

        return q

    def stopping_criteria_reached(self):
        return self.total_saved >= self.total_to_save


def save_agents_path_without_frame_png(
    agent: MultiAgent, k=0, sampling_rate=5, **kwargs
):
    angle_to_dx = {0: 0.0, 90: 0.1, 180: 0, 270: -0.1}
    angle_to_dz = {0: 0.1, 90: 0.0, 180: -0.1, 270: 0.0}
    available_actions = agent.episode.available_actions
    step_results = agent.step_results
    plt.figure(figsize=(14, 14))
    plt.tight_layout()
    plt.axis("equal")
    colors = mpl.cm.rainbow(np.linspace(0, 1, agent.episode.environment.num_agents))

    agent_marker_type = "o"
    agent_marker_size = 6
    object_marker_type = "*"
    object_marker_size = 10
    short_arrow_length_factor = 1.8
    long_arrow_length_factor = 1.2
    longer_arrow_length_factor = 0.8
    object_color = "cyan"

    object_before_positions = [
        sr[0]["object_location"][agent.episode.object_id]["before_location"]
        for sr in step_results
    ]
    object_after_positions = [
        sr[0]["object_location"][agent.episode.object_id]["after_location"]
        for sr in step_results
    ]
    plt.plot(
        object_before_positions[0]["x"],
        object_before_positions[0]["z"],
        color="blue",
        marker="x",
        markersize=12,
    )
    plt.arrow(
        object_before_positions[0]["x"],
        object_before_positions[0]["z"],
        angle_to_dx[round(object_before_positions[0]["rotation"] / 90.0) * 90] / 0.5,
        angle_to_dz[round(object_before_positions[0]["rotation"] / 90.0) * 90] / 0.5,
        color="blue",
        head_width=0.08,
    )
    if hasattr(agent.episode, "object_points_set"):
        object_points_set = agent.episode.object_points_set()
        object_points_set_x = [point_tuple[0] for point_tuple in object_points_set]
        object_points_set_z = [point_tuple[1] for point_tuple in object_points_set]
        plt.plot(
            object_points_set_x,
            object_points_set_z,
            color="blue",
            marker="x",
            markersize=6,
            linestyle="None",
        )
    for agent_id in range(agent.environment.num_agents):
        agent_color = colors[agent_id]
        before_positions = [sr[agent_id]["before_location"] for sr in step_results]
        after_positions = [sr[agent_id]["after_location"] for sr in step_results]
        plt.plot(
            before_positions[0]["x"],
            before_positions[0]["z"],
            color=agent_color,
            marker="x",
            markersize=10,
        )
        plt.arrow(
            before_positions[0]["x"],
            before_positions[0]["z"],
            angle_to_dx[round(object_before_positions[0]["rotation"] / 90.0) * 90]
            / 0.5,
            angle_to_dz[round(object_before_positions[0]["rotation"] / 90.0) * 90]
            / 0.5,
            color=agent_color,
            head_width=0.08,
        )
        for sr_i, sr in enumerate(step_results):
            before_position_i = before_positions[sr_i]
            after_position_i = after_positions[sr_i]
            object_before_position_i = object_before_positions[sr_i]
            object_after_position_i = object_after_positions[sr_i]
            object_before_angle_i = (
                round((object_before_position_i["rotation"] % 360) / 90.0) * 90.0
            )
            object_after_angle_i = (
                round((object_after_position_i["rotation"] % 360) / 90.0) * 90.0
            )
            before_angle_i = round((before_position_i["rotation"] % 360) / 90.0) * 90.0
            after_angle_i = round((after_position_i["rotation"] % 360) / 90.0) * 90.0
            action_string = available_actions[sr[agent_id]["action"]]
            agent_arrow = {
                "color": ["black"],
                "length_factor": [short_arrow_length_factor],
                "angle": [after_angle_i],
            }
            object_arrow = {
                "color": ["black"],
                "length_factor": [short_arrow_length_factor],
                "angle": [object_after_angle_i],
            }

            if not sr[agent_id]["action_success"]:
                # No marks for unsuccessful actions
                continue
            if action_string == "MoveAhead":
                # No extra marks for individual move aheads
                pass
            elif action_string in ["RotateLeft", "RotateRight"]:
                # overwite with a long arrow to mark rotations
                agent_arrow["color"].append("yellow")
                agent_arrow["length_factor"].append(long_arrow_length_factor)
                agent_arrow["angle"].append(after_angle_i)
            elif action_string in ["RotateLiftedRight"]:
                # overwite with a long arrow to mark rotations
                object_arrow["color"].append("yellow")
                object_arrow["length_factor"].append(long_arrow_length_factor)
                object_arrow["angle"].append(object_after_angle_i)
            elif action_string in [
                "MoveLiftedObjectAhead",
                "MoveLiftedObjectRight",
                "MoveLiftedObjectBack",
                "MoveLiftedObjectLeft",
            ]:
                object_arrow["color"].append("green")
                object_arrow["length_factor"].append(longer_arrow_length_factor)
                action_to_rot_ind_offset = {
                    "MoveLiftedObjectAhead": 0,
                    "MoveLiftedObjectRight": 1,
                    "MoveLiftedObjectBack": 2,
                    "MoveLiftedObjectLeft": 3,
                }
                object_rot_ind = round((object_after_angle_i % 360) / 90.0)
                adjusted_rot_ind = (
                    action_to_rot_ind_offset[action_string] + object_rot_ind
                ) % 4
                adjusted_rot = adjusted_rot_ind * 90
                object_arrow["angle"].append(adjusted_rot)
            elif action_string in [
                "MoveAgentsAheadWithObject",
                "MoveAgentsRightWithObject",
                "MoveAgentsBackWithObject",
                "MoveAgentsLeftWithObject",
            ]:
                agent_arrow["color"].append(agent_color)
                agent_arrow["length_factor"].append(longer_arrow_length_factor)

                action_to_rot_ind_offset = {
                    "MoveAgentsAheadWithObject": 0,
                    "MoveAgentsRightWithObject": 1,
                    "MoveAgentsBackWithObject": 2,
                    "MoveAgentsLeftWithObject": 3,
                }
                agent_rot_ind = round((after_angle_i % 360) / 90.0)
                adjusted_rot_ind = (
                    action_to_rot_ind_offset[action_string] + agent_rot_ind
                ) % 4
                adjusted_rot = adjusted_rot_ind * 90
                agent_arrow["angle"].append(adjusted_rot)
            elif action_string in [
                "MoveAgentsNorthWithObject",
                "MoveAgentsEastWithObject",
                "MoveAgentsSouthWithObject",
                "MoveAgentsWestWithObject",
            ]:
                agent_arrow["color"].append(agent_color)
                agent_arrow["length_factor"].append(longer_arrow_length_factor)
                action_to_rot = {
                    "MoveAgentsNorthWithObject": 0,
                    "MoveAgentsEastWithObject": 90,
                    "MoveAgentsSouthWithObject": 180,
                    "MoveAgentsWestWithObject": 270,
                }
                agent_arrow["angle"].append(action_to_rot[action_string])
            elif action_string == "Pass":
                continue
            plt.plot(
                after_position_i["x"],
                after_position_i["z"],
                color=agent_color,
                marker=agent_marker_type,
                markersize=agent_marker_size,
                alpha=0.2,
            )
            plt.plot(
                object_after_position_i["x"],
                object_after_position_i["z"],
                color=object_color,
                marker=object_marker_type,
                markersize=object_marker_size,
                alpha=0.2,
            )
            assert agent_arrow.keys() == object_arrow.keys()
            arrow_keys = list(agent_arrow.keys())
            assert all(
                len(agent_arrow[key]) == len(agent_arrow[arrow_keys[0]])
                for key in arrow_keys
            )
            assert all(
                len(object_arrow[key]) == len(object_arrow[arrow_keys[0]])
                for key in arrow_keys
            )
            for arrow_i in range(len(agent_arrow[arrow_keys[0]]))[::-1]:
                angle = agent_arrow["angle"][arrow_i]
                color = agent_arrow["color"][arrow_i]
                length_factor = agent_arrow["length_factor"][arrow_i]
                plt.arrow(
                    after_position_i["x"],
                    after_position_i["z"],
                    angle_to_dx[angle] / (length_factor + 0.05),
                    angle_to_dz[angle] / (length_factor + 0.05),
                    color=color,
                    alpha=0.4,
                )
                if sr_i % sampling_rate == 0:
                    plt.annotate(
                        str(sr_i),
                        (
                            after_position_i["x"]
                            + angle_to_dx[angle] * 2 / (length_factor + 0.05),
                            after_position_i["z"]
                            + angle_to_dz[angle] * 2 / (length_factor + 0.05),
                        ),
                    )
            for arrow_i in range(len(object_arrow[arrow_keys[0]]))[::-1]:
                angle = object_arrow["angle"][arrow_i]
                color = object_arrow["color"][arrow_i]
                length_factor = object_arrow["length_factor"][arrow_i]

                plt.arrow(
                    object_after_position_i["x"],
                    object_after_position_i["z"],
                    angle_to_dx[angle] / (length_factor + 0.05),
                    angle_to_dz[angle] / (length_factor + 0.05),
                    color=color,
                    alpha=0.4,
                )
                if sr_i % sampling_rate == 0:
                    plt.annotate(
                        str(sr_i),
                        (
                            object_after_position_i["x"]
                            + angle_to_dx[angle] * 2 / (length_factor + 0.05),
                            object_after_position_i["z"]
                            + angle_to_dz[angle] * 2 / (length_factor + 0.05),
                        ),
                    )
    plt.arrow(
        object_after_positions[-1]["x"],
        object_after_positions[-1]["z"],
        angle_to_dx[round(object_after_positions[-1]["rotation"] / 90.0) * 90] / 0.5,
        angle_to_dz[round(object_after_positions[-1]["rotation"] / 90.0) * 90] / 0.5,
        color="blue",
        head_width=0.08,
        linestyle=":",
    )
    # Mark all reachable points
    for p in agent.episode.environment.initially_reachable_points:
        plt.plot(p["x"], p["z"], "k.", markersize=5)
    # Mark all unreachable points
    for (px, pz) in list(agent.episode.environment.get_initially_unreachable_points()):
        plt.plot(px, pz, "k,", markersize=5)
    plt.savefig(
        os.path.join(
            ABS_PATH_TO_ANALYSIS_RESULTS_DIR, "episode_run_paths/{}_{}.jpg"
        ).format(kwargs["args"].task, k),
        bbox_inches="tight",
    )
    plt.close("all")
    time.sleep(0.1)


def save_agents_path_png(
    agent: MultiAgent, k=0, start_resolution: int = 84, save_resolution: int = 1000,
):
    agent.environment.step(
        {
            "action": "ChangeResolution",
            "x": save_resolution,
            "y": save_resolution,
            "agentId": 0,
        }
    )
    time.sleep(2)
    step_results = agent.step_results

    joint_position_mark_colors = []
    pickup_index = agent.episode.available_actions.index("Pickup")
    for sr in step_results:
        all_picked = all(
            sr[i]["action"] == pickup_index for i in range(agent.environment.num_agents)
        )
        all_success = all(
            sr[i]["action_success"] for i in range(agent.environment.num_agents)
        )
        if all_picked and all_success:
            joint_position_mark_colors.append("white")
        elif all_picked:
            joint_position_mark_colors.append("grey")
        else:
            joint_position_mark_colors.append(None)

    map_data = get_agent_map_data(agent.environment)
    frame = map_data["frame"]
    for i in range(agent.environment.num_agents):
        positions = [sr[i]["before_location"] for sr in step_results]
        positions.append(step_results[-1][i]["after_location"])
        position_mark_colors = []
        for j, sr in enumerate(step_results):
            if joint_position_mark_colors[j] is not None:
                position_mark_colors.append(joint_position_mark_colors[j])
            elif sr[i]["action"] == agent.episode.available_actions.index("Pickup"):
                position_mark_colors.append("black")
            else:
                position_mark_colors.append(None)
        position_mark_colors.append(None)
        frame = visualize_agent_path(
            positions,
            frame,
            map_data["pos_translator"],
            color_pair_ind=i,
            position_mark_colors=position_mark_colors,
            only_show_last_visibility_cone=True,
        )

    agent.environment.step(
        {
            "action": "ChangeResolution",
            "x": start_resolution,
            "y": start_resolution,
            "agentId": 0,
        }
    )
    time.sleep(2)


def vector3s_near_equal(p0: Dict[str, float], p1: Dict[str, float], ep: float = 1e-3):
    return abs(p0["x"] - p1["x"]) + abs(p0["y"] - p1["y"]) + abs(p0["z"] - p1["z"]) < ep


def at_same_location(m0, m1, ignore_camera: bool = False):
    the_same = vector3s_near_equal(
        m0 if "position" not in m0 else m0["position"],
        m1 if "position" not in m1 else m1["position"],
    ) and vector3s_near_equal(m0["rotation"], m1["rotation"])
    if "cameraHorizon" in m0:
        the_same = the_same and (
            ignore_camera
            or abs(
                (m0.get("horizon") if "horizon" in m0 else m0["cameraHorizon"])
                - (m1.get("horizon") if "horizon" in m1 else m1["cameraHorizon"])
            )
            < 0.001
        )
    return the_same
