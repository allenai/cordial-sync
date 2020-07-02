"""
A script used to (randomly) test that our gridworld variant of AI2-THOR
(Grid-AI2-THOR) aligns properly with the standard version of AI2-THOR. In particular, this
script will randomly initialize agents in Grid-AI2-THOR and AI2-THOR in exactly
the same (living-room) scenes and positions and then will take the same (randomly chosen) actions
in both environments. This script then verifies that actions in both environments
succeed and fail at the same time steps.

Currently the number of agents used during this test is fixed at 2. If you wish to
change this number, simply edit `num_agents = 2` below.
"""

import random

import numpy as np
import torch.multiprocessing as mp

from constants import (
    ABS_PATH_TO_LOCAL_THOR_BUILD,
    DRESSER_SILHOUETTE_STRING,
    SCENES_NAMES_SPLIT_BY_TYPE,
)
from rl_ai2thor.ai2thor_gridworld_environment import AI2ThorLiftedObjectGridEnvironment
from rl_multi_agent.furnmove_episodes import egocentric_action_groups
from rl_multi_agent.scripts.generate_furnmove_starting_locations_for_evaluation import (
    create_environment,
    MAX_DISTANCE_FROM_OBJECT,
    LIFTED_OBJECT_INITIAL_HEIGHT,
    try_setting_up_environment,
)

mp = mp.get_context("spawn")

if __name__ == "__main__":
    num_agents = 2
    vision_env = create_environment(num_agents=num_agents)
    vision_env.start("FloorPlan1_physics")

    grid_env = AI2ThorLiftedObjectGridEnvironment(
        local_thor_build=ABS_PATH_TO_LOCAL_THOR_BUILD,
        object_type="Television",
        num_agents=num_agents,
        max_dist_to_lifted_object=MAX_DISTANCE_FROM_OBJECT,
        lifted_object_height=LIFTED_OBJECT_INITIAL_HEIGHT,
        min_steps_between_agents=2,
    )
    grid_env.start("FloorPlan1_physics")

    dresser_mask = grid_env.controller.parse_template_to_mask(DRESSER_SILHOUETTE_STRING)

    def to_xzr(loc):
        if hasattr(loc, "rot"):
            return np.array([loc.x, loc.z, loc.rot])
        elif "position" in loc:
            return np.array(
                [loc["position"]["x"], loc["position"]["z"], loc["rotation"]["y"]]
            )
        else:
            return np.array([loc["x"], loc["z"], loc["rotation"]])

    for i in range(1000):
        print(i)
        random.seed(i)
        scene = random.choice(SCENES_NAMES_SPLIT_BY_TYPE[1])

        (
            success,
            vision_lifted_object_id,
            vision_to_object_id,
        ) = try_setting_up_environment(vision_env, scene, "Television", "Dresser")

        agent_locations = vision_env.get_agent_locations()
        lifted_object = vision_env.get_object_by_id(vision_lifted_object_id, 0)
        to_object = vision_env.get_object_by_id(vision_to_object_id, 0)

        grid_env.reset(scene)

        for j, al in enumerate(agent_locations):
            grid_env.step(
                {
                    "action": "TeleportFull",
                    "agentId": j,
                    "x": al["x"],
                    "z": al["z"],
                    "rotation": {"y": al["rotation"]},
                    "allowAgentIntersection": True,
                }
            )
            assert grid_env.last_event.metadata["lastActionSuccess"]

        grid_env.step(
            {
                "action": "CreateLiftedFurnitureAtLocation",
                "objectType": "Television",
                **lifted_object["position"],
                "rotation": lifted_object["rotation"],
                "agentId": 0,
            }
        )
        assert grid_env.last_event.metadata["lastActionSuccess"]

        grid_env.step(
            {
                "action": "CreateAndPlaceObjectOnFloorAtLocation",
                "objectType": "Dresser",
                "object_mask": dresser_mask,
                **to_object["position"],
                "rotation": to_object["rotation"],
                "agentId": 0,
                "forceAction": True,
            }
        )
        if grid_env.last_event.metadata["lastActionSuccess"]:
            for _ in range(10):
                grid_env.controller.viz_world(do_wait=False)
                action = random.choice(sum(egocentric_action_groups(True), tuple()))
                agent_id = random.choice([0, 1])

                print()
                vision_object_xzr = to_xzr(
                    vision_env.get_object_by_id(vision_lifted_object_id, agent_id=0)
                )
                gridworld_object_xzr = to_xzr(grid_env.controller.lifted_object)
                print("TV positions before")
                print("V", vision_object_xzr.round(2))
                print("G", gridworld_object_xzr.round(2))
                print("Before agent locs")
                print("V", [to_xzr(l) for l in vision_env.get_agent_locations()])
                print("G", [to_xzr(l) for l in grid_env.get_agent_locations()])

                print(action, agent_id)

                vision_env.step(
                    {
                        "action": action,
                        "objectId": vision_lifted_object_id,
                        "maxAgentsDistance": 0.76,
                        "agentId": agent_id,
                    }
                )
                grid_env.step(
                    {
                        "action": action,
                        "objectId": "Television|1",
                        "maxAgentsDistance": 0.76,
                        "agentId": agent_id,
                    }
                )

                vision_object_xzr = to_xzr(
                    vision_env.get_object_by_id(vision_lifted_object_id, agent_id=0)
                )
                gridworld_object_xzr = to_xzr(grid_env.controller.lifted_object)

                a = vision_env.last_event.metadata["lastActionSuccess"]
                b = grid_env.last_event.metadata["lastActionSuccess"]

                print("action success", a, b)

                if a != b:
                    print("Different" + "\n" * 3)
                    break

                if np.abs(vision_object_xzr - gridworld_object_xzr).sum() > 0.01:
                    print("Different tv xzr's")
                    print("V", vision_object_xzr)
                    print("G", gridworld_object_xzr)
                    print("\n" * 5)
                    break

                different_locs = False
                for vloc, gloc in zip(
                    vision_env.get_agent_locations(), grid_env.get_agent_locations()
                ):
                    vloc = to_xzr(vloc)
                    gloc = to_xzr(gloc)

                    if np.abs(vloc - gloc).sum() > 0.01:
                        different_locs = True
                        print("Different agent locs")
                        print(vloc)
                        print(gloc)
                        break
                if different_locs:
                    continue

        else:
            print("Could not place dresser on floor")
