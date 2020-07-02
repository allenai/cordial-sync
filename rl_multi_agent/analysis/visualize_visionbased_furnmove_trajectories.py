"""
This script parses the saved data in `furnmove_evaluations__test/METHOD` to create
trajectory visualizations included in the supplementary of the paper and the videos included in the
qualitative result video. Particularly, the marginal and SYNC 2-agent experiments are analysed and the
trajectory summaries (4 top down views, evenly spread over the episode) and selected video
trajectories are saved in `ABS_PATH_TO_ANALYSIS_RESULTS_DIR/furnmove_traj_visualizations_vision/{METHOD}` directory.
Moreover, if the `METHOD` is communicative, an audio rendering of the agent's communication is also saved as a MIDI file.

Set the appropriate options/flags at the start of the script for the desired outputs. Guiding comments are included.

Run using command:
`python rl_multi_agent/analysis/visualize_visionbased_furnmove_trajectories.py`

See `visualize_gridbased_furnmove_trajectories.py` for an analogous grid-based script.
"""
import copy
import glob
import json
import math
import os
from functools import lru_cache
from typing import Tuple, Optional, List

import PIL
import colour as col
import imageio
import numpy as np
from PIL import Image
from midiutil import MIDIFile

from constants import (
    ABS_PATH_TO_LOCAL_THOR_BUILD,
    ABS_PATH_TO_ANALYSIS_RESULTS_DIR,
    ABS_PATH_TO_DATA_DIR,
    PROJECT_TOP_DIR,
)
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_multi_agent.furnmove_episodes import FurnMoveEgocentricEpisode
from utils.visualization_utils import (
    visualize_agent_path,
    save_frames_to_mp4,
    get_agent_map_data,
)


def interleave(l0, l1):
    return [v for p in zip(l0, l1) for v in p]


IND_TO_ACTION_STR = FurnMoveEgocentricEpisode.class_available_actions(
    include_move_obj_actions=True
)


@lru_cache(128)
def get_icon(ind, size):
    im = Image.open(
        os.path.join(
            PROJECT_TOP_DIR,
            "images/action_icons/furnmove/{}.png".format(IND_TO_ACTION_STR[int(ind)]),
        ),
        "r",
    ).convert("RGBA")
    white_frame = Image.new("RGBA", im.size, color=(255, 255, 255, 255))
    return np.array(
        Image.alpha_composite(white_frame, im)
        .convert("RGB")
        .resize((size, size), resample=PIL.Image.LANCZOS)
    )


def add_outline(frame, color, size=2):
    color = np.array(color).reshape((1, 1, 3))
    frame[:, :size, :] = color
    frame[:, -size:, :] = color
    frame[:size, :, :] = color
    frame[-size:, :, :] = color


def add_progress_bar(frame, p, color=(100, 100, 255)):
    height, width, _ = frame.shape

    bar = np.full((max(20, height // 30), width, 3), fill_value=255, dtype=np.uint8)
    spacer = np.copy(bar[: bar.shape[0] // 3, :, :])

    bar[:, : round(width * p), :] = np.array(color).reshape((1, 1, 3))
    add_outline(bar, (200, 200, 200), size=max(1, height // 200))

    return np.concatenate((frame, spacer, bar), axis=0)


def add_actions_to_frame(
    frame,
    ba: Optional[int],
    aa: Optional[int],
    outline_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None,
):
    import copy

    frame = copy.deepcopy(frame)
    h = frame.shape[0]
    icon_size = h // 6

    ba_icon, aa_icon = [
        get_icon(i, size=icon_size) if i is not None else None for i in [ba, aa]
    ]
    if outline_colors is not None:
        for icon, color in zip([ba_icon, aa_icon], outline_colors):
            if icon is not None and color is not None:
                add_outline(icon, color=color)

    margin = icon_size // 8
    if ba_icon is not None:
        frame[
            -(margin + icon_size) : (-margin), margin : (margin + icon_size)
        ] = ba_icon
    if aa_icon is not None:
        frame[
            -(margin + icon_size) : (-margin), -(margin + icon_size) : (-margin)
        ] = aa_icon
    return frame


def unit_vector(vector):
    """ Returns the unit vector of the vector.

     Taken from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249.
     """
    return vector / np.linalg.norm(vector)


def linear_increase_decrease(p):
    if p <= 1 / 2:
        return 2 * (p ** 2)
    else:
        return -1 + 4 * p - 2 * (p ** 2)


def signed_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    Taken from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if round(angle, 8) == 0:
        return angle

    if minor == 0:
        raise NotImplementedError("Too odd vectors =(")
    return np.sign(minor) * angle


def interpolate_step_results(step_results, multiplier):
    if multiplier == 1:
        return step_results

    @lru_cache(100)
    def signed_angle(rota, rotb):
        avec = np.array(
            [math.cos(-2 * math.pi * rota / 360), math.sin(-2 * math.pi * rota / 360)]
        )
        bvec = np.array(
            [math.cos(-2 * math.pi * rotb / 360), math.sin(-2 * math.pi * rotb / 360)]
        )
        return 360 * (-signed_angle_between(avec, bvec)) / (2 * math.pi)

    def recursive_dict_interpolator(a, b, p, is_angle=False):
        if is_angle:
            is_angle_dict = type(a) == dict

            if is_angle_dict:
                rota = a["y"]
                rotb = b["y"]
            else:
                rota = a
                rotb = b

            rota = 90 * round(rota / 90) % 360
            rotb = 90 * round(rotb / 90) % 360

            angle_from_a_to_b = signed_angle(rota, rotb)

            if is_angle_dict:
                return {
                    "x": a["x"],
                    "y": (a["y"] + p * angle_from_a_to_b) % 360,
                    "z": a["z"],
                }
            else:
                return (a + p * angle_from_a_to_b) % 360

        t = type(a)
        if t == dict:
            return {
                k: (
                    (1 - p) * a[k] + p * b[k]
                    if k in ["x", "y", "z", "horizon"]
                    else (recursive_dict_interpolator(a[k], b[k], p, "rotation" in k))
                )
                for k in a
                if k in b
            }
        else:
            if p == 1:
                return b
            return a

    new_step_results = [step_results[0]]

    for next_sr in step_results[1:]:
        last_sr = new_step_results[-1]

        new_step_results.extend(
            [
                recursive_dict_interpolator(
                    last_sr[j], next_sr[j], (i + 1) / multiplier, False,
                )
                for j in range(2)
            ]
            for i in range(multiplier)
        )
    return new_step_results


if __name__ == "__main__":
    # See `visualize_gridbased_furnmove_trajectories.py` for a guide on how to set the options for
    # desired visualizations.
    # The additional flag of `MULTIPLIER` helps generate a smoother video, by dividing one step into `MULTIPLIER` steps.
    SAVE_VIDEO = False
    ADD_PROGRESS_BAR = True
    FPS = 5
    MULTIPLIER = 5
    mode = "mixture"
    # mode = "marginal"

    if mode == "mixture":
        METHOD = "vision_mixture_cl_rot"
    elif mode == "marginal":
        METHOD = "vision_marginal_nocl_rot"
    else:
        raise NotImplementedError

    load_from_dir = os.path.join(
        ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__test", METHOD
    )
    top_save_dir = os.path.join(
        ABS_PATH_TO_ANALYSIS_RESULTS_DIR, "furnmove_traj_visualizations_vision", METHOD
    )
    os.makedirs(top_save_dir, exist_ok=True)

    video_save_path_ids = {
        # "FloorPlan229_physics__0",
        "FloorPlan229_physics__15",
        # "FloorPlan229_physics__193",
        # "FloorPlan230_physics__47",
        # "FloorPlan230_physics__70",
        # "FloorPlan230_physics__72",
    }

    screen_size = 500
    env = AI2ThorEnvironment(
        local_thor_build=ABS_PATH_TO_LOCAL_THOR_BUILD, num_agents=2
    )
    env.start(
        "FloorPlan1_physics",
        player_screen_height=screen_size,
        player_screen_width=screen_size,
        quality="Ultra",
        move_mag=0.25 / MULTIPLIER,
    )

    paths = sorted(glob.glob(os.path.join(load_from_dir, "*.json")))

    for path in paths:
        path_id = str(os.path.basename(path).replace(".json", ""))
        if path_id not in video_save_path_ids:
            continue

        with open(path, "r") as f:
            traj_info = json.load(f)

        map_row_slice = None
        map_col_slice = None
        step_results = traj_info["step_results"]
        last_sr = copy.deepcopy(step_results[-1])
        step_results.append(last_sr)
        for i in range(len(last_sr)):
            last_sr[i]["before_location"] = last_sr[i]["after_location"]
            last_sr[i]["object_location"]["Television|1"]["before_location"] = last_sr[
                i
            ]["object_location"]["Television|1"]["after_location"]

        episode_init_data = traj_info["episode_init_data"]
        tv_loc_start = episode_init_data["lifted_object_location"]
        tv_loc_end = step_results[-1][0]["object_location"]["Television|1"][
            "after_location"
        ]
        if (not SAVE_VIDEO) and AI2ThorEnvironment.position_dist(
            tv_loc_start, tv_loc_end, use_l1=True
        ) < 3.0:
            print("Skipping {}".format(path_id))
            continue

        if SAVE_VIDEO:
            step_results = interpolate_step_results(step_results, multiplier=MULTIPLIER)

        print("Visualizing {}".format(path_id))

        env.reset(path_id.split("__")[0], move_mag=0.25)
        init_reachable = env.initially_reachable_points_set
        env.reset(path_id.split("__")[0], move_mag=0.25 / MULTIPLIER)

        env.step(
            {
                "action": "CreateObjectAtLocation",
                "objectType": "Television",
                "forceAction": True,
                "agentId": 0,
                "renderImage": False,
                "position": {
                    k: v for k, v in tv_loc_start.items() if k in ["x", "y", "z"]
                },
                "rotation": {"x": 0, "y": tv_loc_start["rotation"], "z": 0},
            }
        )
        assert env.last_event.metadata["lastActionSuccess"]
        tv_id = "Television|1"

        to_object_loc = episode_init_data["to_object_location"]
        env.step(
            {
                "action": "CreateObjectAtLocation",
                "objectType": "Dresser",
                "forceAction": True,
                "agentId": 0,
                "renderImage": False,
                "position": {
                    k: v for k, v in to_object_loc.items() if k in ["x", "y", "z"]
                },
                "rotation": {"x": 0, "y": to_object_loc["rotation"], "z": 0},
            }
        )
        assert env.last_event.metadata["lastActionSuccess"]
        dresser_id = "Dresser|2"

        video_inds = list(range(len(step_results)))

        png_inds = list(
            set(int(x) for x in np.linspace(0, len(step_results) - 1, num=4).round())
        )
        png_inds.sort()

        save_path_png = os.path.join(
            top_save_dir,
            "{}__{}.png".format(
                os.path.basename(path).replace(".json", ""),
                "_".join(
                    [str(ind // MULTIPLIER if SAVE_VIDEO else ind) for ind in png_inds]
                ),
            ),
        )
        save_path_mp4 = os.path.join(
            top_save_dir,
            "{}__video.mp4".format(os.path.basename(path).replace(".json", "")),
        )

        if os.path.exists(save_path_png) and (
            os.path.exists(save_path_png) or not SAVE_VIDEO
        ):
            print("{} already exists, skipping...".format(save_path_png))
            continue

        map_frames = []
        a0_frames = []
        a1_frames = []
        colors_list = [
            list(col.Color("red").range_to(col.Color("#ffc8c8"), len(step_results))),
            list(col.Color("green").range_to(col.Color("#c8ffc8"), len(step_results))),
            list(col.Color("blue").range_to(col.Color("#c8c8ff"), len(step_results))),
        ]

        agent_0_full_traj = [x["before_location"] for x, _ in step_results]
        agent_1_full_traj = [y["before_location"] for _, y in step_results]

        full_tv_traj = [
            x["object_location"][tv_id]["before_location"] for x, _ in step_results
        ]

        for ind in png_inds if not SAVE_VIDEO else video_inds:
            if SAVE_VIDEO and ind % (50 * MULTIPLIER) == 0:
                print("Processed {} steps".format(ind // MULTIPLIER))

            sr0, sr1 = step_results[ind]
            loc0 = sr0["before_location"]
            loc1 = sr1["before_location"]
            tv_loc = sr0["object_location"][tv_id]["before_location"]

            traj0 = agent_0_full_traj[: (ind + 1)]
            traj1 = agent_1_full_traj[: (ind + 1)]
            tv_traj = full_tv_traj[: (ind + 1)]
            if ind == len(step_results) - 1:
                traj0.append(loc0)
                traj1.append(loc1)
                tv_traj.append(tv_loc)

            env.teleport_agent_to(
                **loc0, force_action=True, agent_id=0, render_image=False
            )
            assert env.last_event.metadata["lastActionSuccess"]
            env.teleport_agent_to(
                **loc1, force_action=True, agent_id=1, render_image=False
            )
            assert env.last_event.metadata["lastActionSuccess"]

            env.step(
                {
                    "action": "TeleportObject",
                    "objectId": tv_id,
                    **tv_loc,
                    "rotation": tv_loc["rotation"],
                    "forceAction": True,
                    "agentId": 0,
                    "renderImage": False,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            map_data = get_agent_map_data(env)

            frame = map_data["frame"]

            for i, (traj, show_vis_cone, opacity) in enumerate(
                zip([traj0, traj1, tv_traj], [True, True, False], [0.4, 0.4, 0.8])
            ):
                frame = visualize_agent_path(
                    traj,
                    frame,
                    map_data["pos_translator"],
                    color_pair_ind=i,
                    show_vis_cone=show_vis_cone,
                    only_show_last_visibility_cone=True,
                    opacity=opacity,
                    colors=colors_list[i],
                )

            if map_row_slice is None:
                xs = [p[0] for p in init_reachable]
                zs = [p[1] for p in init_reachable]
                minx, maxx = min(xs) - 0.5, max(xs) + 0.5
                minz, maxz = min(zs) - 0.5, max(zs) + 0.5

                minrow, mincol = map_data["pos_translator"]((minx, maxz))
                maxrow, maxcol = map_data["pos_translator"]((maxx, minz))

                map_row_slice = slice(max(minrow, 0), min(maxrow, screen_size))
                map_col_slice = slice(max(mincol, 0), min(maxcol, screen_size))
            frame = frame[
                map_row_slice, map_col_slice,
            ]
            map_frames.append(frame)

            if SAVE_VIDEO:

                def to_outline_color(success):
                    if success is None:
                        return None
                    elif success:
                        return (0, 255, 0)
                    else:
                        return (255, 0, 0)

                action_success0 = [None, None]
                action_success1 = [None, None]
                if ind == 0:
                    ba0, ba1 = None, None
                    aa0, aa1 = sr0["action"], sr1["action"]
                    action_success0[1] = sr0["action_success"]
                    action_success1[1] = sr1["action_success"]
                elif ind == len(video_inds):
                    oldsr0, oldsr1 = step_results[ind - 1]
                    ba0, ba1 = oldsr0["action"], step_results["action"]
                    aa0, aa1 = None, None
                    action_success0[0] = sr0["action_success"]
                    action_success1[0] = sr1["action_success"]
                else:
                    oldsr0, oldsr1 = step_results[ind - 1]
                    ba0, ba1 = oldsr0["action"], oldsr1["action"]
                    aa0, aa1 = sr0["action"], sr1["action"]

                    action_success0[0] = oldsr0["action_success"]
                    action_success1[0] = oldsr1["action_success"]
                    action_success0[1] = sr0["action_success"]
                    action_success1[1] = sr1["action_success"]

                a0_frames.append(
                    add_actions_to_frame(
                        env.current_frames[0],
                        ba=ba0,
                        aa=None,
                        outline_colors=list(map(to_outline_color, action_success0)),
                    )
                )
                a1_frames.append(
                    add_actions_to_frame(
                        env.current_frames[1],
                        ba=ba1,
                        aa=None,
                        outline_colors=list(map(to_outline_color, action_success1)),
                    )
                )

        map_height, map_width, _ = map_frames[0].shape
        fp_frame_height = a0_frames[0].shape[0] if len(a0_frames) != 0 else map_height
        spacer = np.full(
            (fp_frame_height, screen_size // 20, 3), fill_value=255, dtype=np.uint8
        )

        if SAVE_VIDEO and not os.path.exists(save_path_mp4):
            new_map_width = round(fp_frame_height * map_width / map_height)
            f = lambda x: np.array(
                Image.fromarray(x.astype("uint8"), "RGB").resize(
                    (new_map_width, fp_frame_height), resample=PIL.Image.LANCZOS
                )
            )

            joined_frames = []
            for ind, (map_frame, a0_frame, a1_frame) in enumerate(
                zip(map_frames, a0_frames, a1_frames)
            ):
                joined_frames.append(
                    np.concatenate(
                        interleave(
                            [a0_frame, a1_frame, f(map_frame)],
                            [spacer] * len(map_frames),
                        )[:-1],
                        axis=1,
                    )
                )
                if ADD_PROGRESS_BAR:
                    color = (100, 100, 255)
                    if ind == len(map_frames) - 1:
                        color = (
                            (0, 255, 0)
                            if len(step_results) // MULTIPLIER < 250
                            else (255, 0, 0)
                        )

                    joined_frames[-1] = add_progress_bar(
                        joined_frames[-1], ind / (len(map_frames) - 1), color=color
                    )

            final_frame = copy.deepcopy(joined_frames[-1])

            if MULTIPLIER == 1:
                save_frames_to_mp4(
                    joined_frames + [final_frame] * FPS, save_path_mp4, fps=FPS,
                )
            else:
                save_frames_to_mp4(
                    joined_frames + [final_frame] * FPS * MULTIPLIER,
                    save_path_mp4,
                    fps=FPS * MULTIPLIER,
                )

        for agent_id in range(2):
            MyMIDI = MIDIFile(1)
            MyMIDI.addTempo(0, 0, FPS * 60)

            current_time = 0
            for probs in traj_info["a{}_reply_probs".format(agent_id)]:
                MyMIDI.addNote(0, 0, round(127 * probs[0]), current_time, 1, volume=100)
                current_time += 1

            for i in range(FPS):
                MyMIDI.addNote(0, 0, 0, current_time, 1, volume=0)
                current_time += 1

            midi_save_path = os.path.join(
                top_save_dir,
                "{}_{}.mid".format(
                    os.path.basename(path).replace(".json", ""), agent_id
                ),
            )

            with open(midi_save_path, "wb") as output_file:
                MyMIDI.writeFile(output_file)

        if not os.path.exists(save_path_png):
            map_frames = [
                map_frames[ind]
                for ind in (png_inds if SAVE_VIDEO else range(len(map_frames)))
            ]
            spacer = np.full(
                (map_height, screen_size // 20, 3), fill_value=255, dtype=np.uint8
            )

            big_frame = np.concatenate(
                interleave(map_frames, [spacer] * len(map_frames))[:-1], axis=1
            )
            imageio.imwrite(save_path_png, big_frame)
