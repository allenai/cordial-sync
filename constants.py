"""Manages constants required by multiple files."""
import getpass
import os
import platform
import socket
from pathlib import Path


def make_scene_name(type, num):
    if type == "":
        return "FloorPlan" + str(num) + "_physics"
    elif num < 10:
        return "FloorPlan" + type + "0" + str(num) + "_physics"
    else:
        return "FloorPlan" + type + str(num) + "_physics"


SCENES_NAMES_SPLIT_BY_TYPE = [
    [make_scene_name(j, i) for i in range(1, 31)] for j in ["", "2", "3", "4"]
]
SCENE_NAMES = sum(SCENES_NAMES_SPLIT_BY_TYPE, [])
TRAIN_SCENE_NAMES = [
    SCENES_NAMES_SPLIT_BY_TYPE[j][i] for j in range(3) for i in range(20)
]
VALID_SCENE_NAMES = [
    SCENES_NAMES_SPLIT_BY_TYPE[j][i] for j in range(3) for i in range(20, 25)
]
TEST_SCENE_NAMES = [
    SCENES_NAMES_SPLIT_BY_TYPE[j][i] for j in range(3) for i in range(25, 30)
] + SCENES_NAMES_SPLIT_BY_TYPE[3]


VISIBILITY_DISTANCE = 1.25
FOV = 90.0

PICK_UP_HEIGHT = 1.0

UNREACHABLE_SYM = 0
REACHABLE_SYM = 1
AGENT_SYM = 2
NO_INFO_SYM = 3
GOAL_OBJ_SYM = 4
VISITED_SYM = 5

AGENT_SELF_0 = 6
AGENT_SELF_90 = 7
AGENT_SELF_180 = 8
AGENT_SELF_270 = 9

AGENT_OTHER_0 = 10
AGENT_OTHER_90 = 11
AGENT_OTHER_180 = 12
AGENT_OTHER_270 = 13

# Distance that qualifies an episode as easy or medium
# For more details see `create_table` method in
# `rl_multi_agent/analysis/summarize_furnlift_eval_results.py`.
# array([ 2., 11., 17., 45.]) == [0, 1/3, 2/3, 1] percentiles of initial manhat distances
EASY_MAX = 10
MED_MAX = 17


def rotate_tuple_90_clockwise(t):
    return (t[1], -t[0])


TELEVISION_ROT_0_POINTS = (
    [(0.25 * i, 0.25) for i in [-1, 0, 1]]
    + [(0.25 * i, 0) for i in [-2, -1, 0, 1, 2]]
    + [(0.25 * i, -0.25) for i in [-2, -1, 0, 1, 2]]
)
TELEVISION_ROT_90_POINTS = [
    rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_0_POINTS
]
TELEVISION_ROT_180_POINTS = [
    rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_90_POINTS
]
TELEVISION_ROT_270_POINTS = [
    rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_180_POINTS
]
TELEVISION_ROTATION_TO_OCCUPATIONS = {
    0: TELEVISION_ROT_0_POINTS,
    90: TELEVISION_ROT_90_POINTS,
    180: TELEVISION_ROT_180_POINTS,
    270: TELEVISION_ROT_270_POINTS,
}

TV_STAND_ROT_0_POINTS = [
    (0.25 * i, 0.25 * j) for i in range(-2, 3) for j in range(-1, 2)
]
TV_STAND_ROT_90_POINTS = [
    (0.25 * i, 0.25 * j) for i in range(-1, 2) for j in range(-2, 3)
]
TV_STAND_ROT_180_POINTS = TV_STAND_ROT_0_POINTS
TV_STAND_ROT_270_POINTS = TV_STAND_ROT_90_POINTS
TV_STAND_ROTATION_TO_OCCUPATIONS = {
    0: TV_STAND_ROT_0_POINTS,
    90: TV_STAND_ROT_90_POINTS,
    180: TV_STAND_ROT_180_POINTS,
    270: TV_STAND_ROT_270_POINTS,
}


DRESSER_SILHOUETTE_STRING = """1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
"""

SPLIT_TO_USE_FOR_EVALUATION = "test"

HOST_NAME = socket.gethostname()
USERNAME = getpass.getuser()
PLATFORM = platform.system()

PROJECT_TOP_DIR = os.path.dirname(Path(__file__))
if PLATFORM == "Linux":
    ABS_PATH_TO_LOCAL_THOR_BUILD = os.path.join(
        PROJECT_TOP_DIR, "ai2thor_builds/thor-2-3-2020-Linux64"
    )
elif PLATFORM == "Darwin":
    ABS_PATH_TO_LOCAL_THOR_BUILD = os.path.join(
        PROJECT_TOP_DIR,
        "ai2thor_builds/thor-2-3-2020-OSXIntel64.app/Contents/MacOS/thor-2-3-2020-OSXIntel64",
    )
else:
    raise NotImplementedError(
        (
            "Your platform, {}, is not currently supported. "
            "Supported platforms include Linux and MacOSX."
        ).format(PLATFORM)
    )

ABS_PATH_TO_FINAL_FURNMOVE_CKPTS = os.path.join(
    PROJECT_TOP_DIR, "trained_models", "final_furnmove_ckpts"
)
ABS_PATH_TO_FINAL_FURNLIFT_CKPTS = os.path.join(
    PROJECT_TOP_DIR, "trained_models", "final_furnlift_ckpts"
)
ABS_PATH_TO_ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_TOP_DIR, "analysis_output")
ABS_PATH_TO_DATA_DIR = os.path.join(PROJECT_TOP_DIR, "data")
