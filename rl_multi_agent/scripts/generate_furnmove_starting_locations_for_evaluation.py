"""
A script used to generate a fixed dataset of initial starting locations for agents,
the TV, and the TV stand. This dataset can then be used to provide a fair comparison
between different models trained to complete the FurnMove task.
"""
import hashlib
import itertools
import json
import os
import random
import sys
import time
import traceback
import warnings
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle

import constants
from constants import ABS_PATH_TO_LOCAL_THOR_BUILD, ABS_PATH_TO_DATA_DIR
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment

mp = mp.get_context("spawn")

LIFTED_OBJECT_INITIAL_HEIGHT = 1.3
MAX_DISTANCE_FROM_OBJECT = 0.76
CURRENTLY_ON_SERVER = torch.cuda.is_available()


def create_environment(num_agents):
    return AI2ThorEnvironment(
        x_display="0.0" if CURRENTLY_ON_SERVER else None,
        local_thor_build=ABS_PATH_TO_LOCAL_THOR_BUILD,
        restrict_to_initially_reachable_points=True,
        num_agents=num_agents,
    )


def try_setting_up_environment(env, scene, lifted_object_type, to_object_type):
    env.reset(scene)

    for i in range(env.num_agents):
        env.teleport_agent_to(
            **env.last_event.metadata["agent"]["position"],
            rotation=env.last_event.metadata["agent"]["rotation"]["y"],
            horizon=30.0,
            standing=True,
            force_action=True,
            agent_id=i,
        )

    failure_reasons = []
    scene_successfully_setup = False
    lifted_object_id = None
    to_object_id = None

    for _ in range(20):
        # Randomly generate agents looking at the TV
        env.step(
            {
                "action": "DisableAllObjectsOfType",
                "objectId": lifted_object_type,
                "agentId": 0,
            }
        )
        sr = env.step(
            {
                "action": "RandomlyCreateLiftedFurniture",
                "objectType": lifted_object_type,
                "y": LIFTED_OBJECT_INITIAL_HEIGHT,
                "z": MAX_DISTANCE_FROM_OBJECT,
            }
        )
        if (
            env.last_event.metadata["lastAction"] != "RandomlyCreateLiftedFurniture"
            or not env.last_event.metadata["lastActionSuccess"]
        ):
            failure_reasons.append(
                "Could not randomize location of {} in {}. Error message {}.".format(
                    lifted_object_type, scene, env.last_event.metadata["errorMessage"]
                )
            )
            continue

        # Refreshing reachable here should not be done as it means the
        # space underneath the TV would be considered unreachable even though,
        # once the TV moves, it is.

        objects_of_type = env.all_objects_with_properties(
            {"objectType": lifted_object_type}, agent_id=0
        )
        if len(objects_of_type) != 1:
            print("len(objects_of_type): {}".format(len(objects_of_type)))
            raise (Exception("len(objects_of_type) != 1"))

        object = objects_of_type[0]
        lifted_object_id = object["objectId"]
        assert lifted_object_id == sr.metadata["actionReturn"]

        # Create the object we should navigate to
        env.step(
            {
                "action": "RandomlyCreateAndPlaceObjectOnFloor",
                "objectType": to_object_type,
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
                    to_object_type, scene, env.last_event.metadata["errorMessage"]
                )
            )
            continue

        to_object_id = env.last_event.metadata["actionReturn"]
        assert to_object_id is not None and len(to_object_id) != 0

        scene_successfully_setup = True
        break

    return scene_successfully_setup, lifted_object_id, to_object_id


def generate_data(
    worker_num: int,
    num_agents: int,
    lifted_object_type: str,
    to_object_type: str,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    ptitle("Training Agent: {}".format(worker_num))

    try:
        env = None
        while not in_queue.empty():
            scene, index = in_queue.get(timeout=3)

            if env is None:
                env = create_environment(num_agents=num_agents)
                env.start(scene, player_screen_width=84, player_screen_height=84)

            seed = (
                int(hashlib.md5((scene + "|{}".format(index)).encode()).hexdigest(), 16)
                % 2 ** 31
            )
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            success, lifted_object_id, to_object_id = try_setting_up_environment(
                env,
                scene,
                lifted_object_type=lifted_object_type,
                to_object_type=to_object_type,
            )

            if not success:
                warnings.warn(
                    "Failed to successfully set up environment in scene {} with index {}".format(
                        scene, index
                    )
                )
                continue

            lifted_object = env.get_object_by_id(object_id=lifted_object_id, agent_id=0)
            lifted_object_location = {
                **lifted_object["position"],
                "rotation": (round(lifted_object["rotation"]["y"] / 90) % 4) * 90,
            }

            to_object = env.get_object_by_id(object_id=to_object_id, agent_id=0)
            to_object_location = {
                **to_object["position"],
                "rotation": (round(to_object["rotation"]["y"] / 90) % 4) * 90,
            }

            out_queue.put(
                {
                    "scene": scene,
                    "index": index,
                    "agent_locations": env.get_agent_locations(),
                    "lifted_object_location": lifted_object_location,
                    "to_object_location": to_object_location,
                }
            )

    except Exception as _:
        print("ERROR:")
        traceback.print_exc(file=sys.stdout)
        raise e
    finally:
        if env is not None:
            env.stop()
        print("Process ending.")
        sys.exit()


if __name__ == "__main__":
    num_agents = 2
    num_processes = 16 if CURRENTLY_ON_SERVER else 4

    split = "test"

    if split == "train":
        scenes = constants.TRAIN_SCENE_NAMES[20:40]
        repeats_per_scene = 50
    elif split == "valid":
        scenes = constants.VALID_SCENE_NAMES[5:10]
        repeats_per_scene = 200
    elif split == "test":
        scenes = constants.TEST_SCENE_NAMES[5:10]
        repeats_per_scene = 200
    else:
        raise NotImplementedError()

    assert os.path.exists(ABS_PATH_TO_DATA_DIR)

    save_path = os.path.join(
        ABS_PATH_TO_DATA_DIR,
        "furnmove_episode_start_positions_for_eval{}__{}.json".format(
            "" if num_agents == 2 else "__{}agents".format(num_agents), split
        ),
    )

    start_positions_data = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            start_positions_data = json.load(f)

    def save_data_to_disk():
        with open(save_path, "w") as f:
            json.dump(start_positions_data, f)

    processes = []
    try:
        remaining_keys = set()
        send_queue = mp.Queue()
        recieve_queue = mp.Queue()
        k = 0
        for scene, index in itertools.product(scenes, list(range(repeats_per_scene))):

            if scene not in start_positions_data:
                start_positions_data[scene] = {}

            if str(index) in start_positions_data[scene]:
                continue

            remaining_keys.add((scene, index))
            send_queue.put((scene, index))
            k += 1
        if k == 0:
            print("Already generated all positions for evaluation? Quitting...")
            sys.exit()

        print("Starting data generation with {} unique configurations.".format(k))

        for worker_num in range(num_processes):
            p = mp.Process(
                target=generate_data,
                kwargs=dict(
                    worker_num=worker_num,
                    num_agents=num_agents,
                    lifted_object_type="Television",
                    to_object_type="Dresser",
                    in_queue=send_queue,
                    out_queue=recieve_queue,
                ),
            )
            p.start()
            time.sleep(0.2)
            processes.append(p)

        time.sleep(3)

        while len(remaining_keys) != 0 and (
            any([p.is_alive() for p in processes]) or not recieve_queue.empty()
        ):
            print("Gathering incoming data...")
            last_time = time.time()
            while time.time() - last_time < 60 and len(remaining_keys) != 0:
                data = recieve_queue.get(timeout=120)
                scene, index = (data["scene"], data["index"])
                del data["scene"], data["index"]
                start_positions_data[scene][str(index)] = data

                remaining_keys.remove((scene, index))

            print("Saving to disk...")
            save_data_to_disk()

    except Empty as _:
        print(
            "Outqueue empty for 120 seconds. Assuming everything is done. Writing to disk..."
        )
        save_data_to_disk()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    finally:
        print("Cleaning up main process.\n")

        print("Joining processes.")
        for p in processes:
            p.join(0.1)
        print("Joined.\n")

        print("All done.")
        sys.exit(1)
