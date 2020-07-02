"""
A script used to generate a fixed dataset of initial starting locations for agents
and the TV. This dataset can then be used to provide a fair comparison between
different models trained to complete the FurnLift task.
"""
import hashlib
import itertools
import json
import os
import random
import sys
import time
import traceback
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle

import constants
from constants import ABS_PATH_TO_LOCAL_THOR_BUILD, ABS_PATH_TO_DATA_DIR
from rl_multi_agent.furnlift_episode_samplers import FurnLiftEpisodeSamplers
from rl_multi_agent.furnlift_episodes import FurnLiftNApartStateEpisode
from rl_multi_agent.scripts.generate_furnmove_starting_locations_for_evaluation import (
    CURRENTLY_ON_SERVER,
)


class TmpObject(object):
    def __init__(self):
        self.env = None
        self.task_data = None
        self.environment = None
        self.episode = None
        self.docker_enabled = False
        self.local_thor_build = ABS_PATH_TO_LOCAL_THOR_BUILD
        self.x_display = None


def fake_episode_class(env, task_data, **kwargs):
    o = TmpObject()
    o.task_data = task_data
    o.env = env
    return o


def try_setting_up_environment(
    env,
    fake_agent,
    scene,
    target_object_type,
    num_agents=2,
    min_dist_between_agents_to_pickup=8,
):
    init_params = {
        "scenes": [scene],
        "num_agents": num_agents,
        "object_type": target_object_type,
        "episode_class": FurnLiftNApartStateEpisode,
        "player_screen_height": 84,
        "player_screen_width": 84,
        "min_dist_between_agents_to_pickup": min_dist_between_agents_to_pickup,
        "max_ep_using_expert_actions": 1000000,
        "visible_agents": True,
        "max_visible_positions_push": 0,
        "min_max_visible_positions_to_consider": (1000, 1000),
    }
    episode_sampler = FurnLiftEpisodeSamplers(**init_params)
    episode_sampler.env_args = TmpObject()

    next(episode_sampler(agent=fake_agent, env=env))

    target_object_id = fake_agent.episode.task_data["goal_obj_id"]

    return target_object_id


def generate_data(worker_num, target_object_type, in_queue, out_queue):
    ptitle("Training Agent: {}".format(worker_num))

    env = None
    fake_agent = TmpObject()
    try:
        while not in_queue.empty():
            scene, index = in_queue.get(timeout=3)

            seed = (
                int(hashlib.md5((scene + "|{}".format(index)).encode()).hexdigest(), 16)
                % 2 ** 31
            )
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            target_object_id = try_setting_up_environment(
                env=env,
                fake_agent=fake_agent,
                scene=scene,
                target_object_type=target_object_type,
            )
            env = fake_agent.episode.environment

            target_object = env.get_object_by_id(object_id=target_object_id, agent_id=0)
            object_location = {
                **target_object["position"],
                "rotation": (round(target_object["rotation"]["y"] / 90) % 4) * 90,
            }

            out_queue.put(
                {
                    "scene": scene,
                    "index": index,
                    "agent_locations": env.get_agent_locations(),
                    "object_location": object_location,
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
    split = "test"
    num_processes = 16 if CURRENTLY_ON_SERVER else 4

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
        "furnlift_episode_start_positions_for_eval__{}.json".format(split),
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
                args=(worker_num, "Television", send_queue, recieve_queue),
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
