"""
This script parses the saved data in `furnmove_evaluations__test/METHOD` to create
trajectory visualizations included in the supplementary of the paper and the videos included in the
qualitative result video. Particularly, the gridworld 2-agent experiments are analysed and the
trajectory summaries (4 top down views, evenly spread over the episode) and selected video
trajectories are saved in `ABS_PATH_TO_ANALYSIS_RESULTS_DIR/furnmove_traj_visualizations_gridworld/{METHOD}` directory.
Moreover, if the `METHOD` is communicative, an audio rendering of the agent's communication is also saved as a MIDI file.

Set the appropriate options/flags at the start of the script for the desired outputs. Guiding comments are included.

Run using command:
`python rl_multi_agent/analysis/visualize_gridbased_furnmove_trajectories.py`

See `visualize_visionbased_furnmove_trajectories.py` for an analogous vision-based script.
"""
import copy
import glob
import json
import os
import random
from typing import Tuple, Optional, List

import PIL
import colour as col
import imageio
import numpy as np
import torch
from PIL import Image
from midiutil import MIDIFile

import constants
from constants import ABS_PATH_TO_ANALYSIS_RESULTS_DIR, ABS_PATH_TO_DATA_DIR
from constants import ABS_PATH_TO_LOCAL_THOR_BUILD
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.ai2thor_gridworld_environment import AI2ThorLiftedObjectGridEnvironment
from rl_multi_agent.analysis.visualize_visionbased_furnmove_trajectories import (
    get_icon,
    add_outline,
    interleave,
    add_progress_bar,
)
from utils.visualization_utils import save_frames_to_mp4


def add_actions_to_frame(
    frame,
    ba: Optional[int],
    aa: Optional[int],
    outline_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None,
    position_top=False,
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
    row_slice = (
        slice(-(margin + icon_size), -margin)
        if not position_top
        else slice(margin, margin + icon_size)
    )
    if ba_icon is not None:
        frame[row_slice, margin : (margin + icon_size)] = ba_icon
    if aa_icon is not None:
        frame[row_slice, -(margin + icon_size) : (-margin)] = aa_icon
    return frame


if __name__ == "__main__":
    # Set some options for the desired output:
    # ---------

    # Note that a snapshot/summary of all trajectories will be saved by this script.

    # You could decide to save the video or not
    SAVE_VIDEO = True

    # Which episode do you wish to save the video for.
    # Scheme: FloorPlan{SCENE_NUMBER}_physics_{TEST_EPISODE}, where SCENE_NUMBER goes from 226 to 230
    # and TEST_EPISODE goes from 0 to 199.
    video_save_path_ids = {
        # "FloorPlan229_physics__0",
        "FloorPlan229_physics__15",
        # "FloorPlan229_physics__193",
        # "FloorPlan230_physics__47",
        # "FloorPlan230_physics__70",
        # "FloorPlan230_physics__72",
    }

    # To add a progress bar to indicate how much of the episode has been completed
    ADD_PROGRESS_BAR = True

    # FPS for the video, faster/slower viewing.
    FPS = 5

    # Method to test
    METHOD = "grid_mixture_cl_rot"
    # ---------

    # Additional path variables are set
    load_from_dir = os.path.join(
        ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__test", METHOD
    )
    top_save_dir = os.path.join(
        ABS_PATH_TO_ANALYSIS_RESULTS_DIR,
        "furnmove_traj_visualizations_gridworld",
        METHOD,
    )
    os.makedirs(top_save_dir, exist_ok=True)

    env = AI2ThorLiftedObjectGridEnvironment(
        local_thor_build=ABS_PATH_TO_LOCAL_THOR_BUILD,
        num_agents=2,
        lifted_object_height=1.3,
        object_type="Television",
        max_dist_to_lifted_object=0.76,
        min_steps_between_agents=2,
        remove_unconnected_positions=True,
    )
    env.start("FloorPlan1_physics")

    paths = sorted(glob.glob(os.path.join(load_from_dir, "*.json")))

    if not torch.cuda.is_available():
        random.shuffle(paths)

    for path in paths:
        path_id = str(os.path.basename(path).replace(".json", ""))
        if SAVE_VIDEO and path_id not in video_save_path_ids:
            continue

        with open(path, "r") as f:
            traj_info = json.load(f)

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

        print("Visualizing {}".format(path_id))

        env.reset(path_id.split("__")[0], move_mag=0.25)

        for agent_id, loc in enumerate(episode_init_data["agent_locations"]):
            env.step(
                {
                    "action": "TeleportFull",
                    "agentId": agent_id,
                    "x": loc["x"],
                    "z": loc["z"],
                    "rotation": {"y": loc["rotation"]},
                    "allowAgentIntersection": True,
                    "makeReachable": True,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

        env.step(
            {
                "action": "CreateLiftedFurnitureAtLocation",
                "objectType": "Television",
                "x": tv_loc_start["x"],
                "z": tv_loc_start["z"],
                "rotation": {"y": tv_loc_start["rotation"]},
                "agentId": 0,
                "forceAction": True,
            }
        )
        assert env.last_event.metadata["lastActionSuccess"]
        tv_id = "Television|1"

        to_object_loc = episode_init_data["to_object_location"]
        env.step(
            {
                "action": "CreateAndPlaceObjectOnFloorAtLocation",
                "objectType": "Dresser",
                "object_mask": env.controller.parse_template_to_mask(
                    constants.DRESSER_SILHOUETTE_STRING
                ),
                **to_object_loc,
                "rotation": {"y": to_object_loc["rotation"]},
                "agentId": 0,
                "forceAction": True,
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
                "_".join([str(ind) for ind in png_inds]),
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

        for ind in png_inds if not SAVE_VIDEO else video_inds:
            if SAVE_VIDEO and ind % 50 == 0:
                print("Processed {} steps".format(ind))

            sr0, sr1 = step_results[ind]
            loc0 = sr0["before_location"]
            loc1 = sr1["before_location"]
            tv_loc = sr0["object_location"][tv_id]["before_location"]

            traj0 = [x["before_location"] for x, _ in step_results[: (ind + 1)]]
            traj1 = [y["before_location"] for _, y in step_results[: (ind + 1)]]
            tv_traj = [
                x["object_location"][tv_id]["before_location"]
                for x, _ in step_results[: (ind + 1)]
            ]
            if ind == len(step_results) - 1:
                traj0.append(loc0)
                traj1.append(loc1)
                tv_traj.append(tv_loc)

            for agent_id, loc in enumerate([loc0, loc1]):
                env.step(
                    {
                        "action": "TeleportFull",
                        "agentId": agent_id,
                        "x": loc["x"],
                        "z": loc["z"],
                        "rotation": {"y": loc["rotation"]},
                        "allowAgentIntersection": True,
                        "makeReachable": True,
                        "forceAction": True,
                    }
                )
                assert env.last_event.metadata[
                    "lastActionSuccess"
                ], env.last_event.metadata["errorMessage"]

            env.step(
                {
                    "action": "TeleportObject",
                    "objectId": tv_id,
                    **tv_loc,
                    "rotation": {"y": tv_loc["rotation"]},
                    "forceAction": True,
                    "agentId": 0,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            map_frames.append(env.controller.viz_world(20, False, 0, True))

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
                # ba{i} aa{i} are before and after actions of agent i
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

                ego_frames = env.controller.viz_ego_agent_views(40, array_only=True)
                ego_frames = [
                    np.pad(
                        ef,
                        ((4, 4), (4, 4), (0, 0)),
                        mode="constant",
                        constant_values=200,
                    )
                    for ef in ego_frames
                ]
                a0_frames.append(
                    add_actions_to_frame(
                        ego_frames[0],
                        ba=ba0,
                        aa=None,  # aa0,
                        outline_colors=list(map(to_outline_color, action_success0)),
                        position_top=True,
                    )
                )
                a1_frames.append(
                    add_actions_to_frame(
                        ego_frames[1],
                        ba=ba1,
                        aa=None,  # aa1,
                        outline_colors=list(map(to_outline_color, action_success1)),
                        position_top=True,
                    )
                )

        map_height, map_width, _ = map_frames[0].shape
        fp_frame_height = a0_frames[0].shape[0] if len(a0_frames) != 0 else map_height
        spacer = np.full(
            (fp_frame_height, fp_frame_height // 20, 3), fill_value=255, dtype=np.uint8
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
                            # [f(a0_frame), f(a1_frame), map_frame],
                            [spacer] * len(map_frames),
                        )[:-1],
                        axis=1,
                    )
                )
                if ADD_PROGRESS_BAR:
                    color = (100, 100, 255)
                    if ind == len(map_frames) - 1:
                        color = (0, 255, 0) if len(step_results) < 250 else (255, 0, 0)

                    joined_frames[-1] = add_progress_bar(
                        joined_frames[-1], ind / (len(map_frames) - 1), color=color
                    )

            final_frame = copy.deepcopy(joined_frames[-1])

            save_frames_to_mp4(
                joined_frames + [final_frame] * FPS, save_path_mp4, fps=FPS,
            )

            for agent_id in range(2):
                MyMIDI = MIDIFile(1)
                MyMIDI.addTempo(0, 0, FPS * 60)

                current_time = 0
                for probs in traj_info["a{}_reply_probs".format(agent_id)]:
                    MyMIDI.addNote(
                        0, 0, round(127 * probs[0]), current_time, 1, volume=100
                    )
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
            if SAVE_VIDEO:
                # Select the frames from the video. No selection is needed if `SAVE_VIDEO` is `False`.
                map_frames = [map_frames[ind] for ind in png_inds]

            spacer = np.full(
                (map_height, fp_frame_height // 20, 3), fill_value=255, dtype=np.uint8
            )

            big_frame = np.concatenate(
                interleave(map_frames, [spacer] * len(map_frames))[:-1], axis=1
            )
            imageio.imwrite(save_path_png, big_frame)
