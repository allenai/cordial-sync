"""
This script generates joint policy summaries presented in the paper assuming you
have followed the directions in our README.md to download the `data` directory. The plots will
be saved in the `PATH/TO/THIS/PROJECT/analysis_output/plots/ATTEMPT_OR_SUCCESS` directory.
The `ATTEMPT_OR_SUCCESS` flag helps visualize the actions attempted and actions successful (note
that in the paper we show `attempt` joint policy summaries.
"""
import glob
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from constants import ABS_PATH_TO_ANALYSIS_RESULTS_DIR, ABS_PATH_TO_DATA_DIR


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


if __name__ == "__main__":

    split = "test"

    id_to_time_averaged_action_mat = {}
    id_to_time_averaged_success_action_mat = {}

    for dir in glob.glob(
        os.path.join(ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__{}/*").format(split)
    ):
        dir_name = os.path.basename(dir)
        id = dir_name.split("__")[0]

        if "3agents" in id:
            continue

        print()
        print(id)

        time_averaged_success_action_mats = []
        time_averaged_action_mats = []
        for i, p in enumerate(glob.glob(os.path.join(dir, "*.json"))):
            if i % 100 == 0:
                print(i)

            with open(p, "r") as f:
                result_dict = json.load(f)

            time_averaged_success_action_mats.append(
                np.array(result_dict["action_taken_success_matrix"])
            )
            time_averaged_success_action_mats[-1] = time_averaged_success_action_mats[
                -1
            ] / (time_averaged_success_action_mats[-1].sum() + 1e-6)

            time_averaged_action_mats.append(
                np.array(result_dict["action_taken_matrix"])
            )
            time_averaged_action_mats[-1] = time_averaged_action_mats[-1] / (
                time_averaged_action_mats[-1].sum() + 1e-6
            )

        id_to_time_averaged_success_action_mat[id] = np.stack(
            time_averaged_success_action_mats, axis=0
        ).mean(0)
        id_to_time_averaged_action_mat[id] = np.stack(
            time_averaged_action_mats, axis=0
        ).mean(0)

    c = mcolors.ColorConverter().to_rgb
    wr = make_colormap([c("white"), c("red")])

    success_or_attempt = "attempt"

    if success_or_attempt == "attempt":
        id_to_mat = id_to_time_averaged_action_mat
    else:
        id_to_mat = id_to_time_averaged_success_action_mat

    dir = os.path.join(
        ABS_PATH_TO_ANALYSIS_RESULTS_DIR, "plots/average_policy_mats/{}"
    ).format(success_or_attempt)
    os.makedirs(dir, exist_ok=True)
    for id in id_to_mat:
        if "3agents" in id:
            continue
        a = id_to_mat[id]
        size = a.shape[0]

        plt.figure(figsize=(3, 3))
        plt.matshow(a, cmap=wr)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # True,
            labelbottom=False,
            length=0,
        )
        ax.set_ylim(size + 0.1 - 0.5, -0.5 - 0.1)
        plt.tick_params(axis="y", which="both", left=False, length=0)  # left=True,

        for i in range(size):
            for j in range(size):
                rect = patches.Rectangle(
                    (i - 0.5, j - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="lightgrey",
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)

        for i in range(size):
            for j in range(size):
                if (
                    size == 13
                    and (
                        (i < 4 and j < 4)
                        or (3 < i <= 7 and 3 < j <= 7)
                        or (i == j == 8)
                        or (8 < i and 8 < j)
                    )
                    or (
                        size == 14
                        and (
                            (i < 5 and j < 5)
                            or (4 < i <= 8 and 4 < j <= 8)
                            or (i == j == 9)
                            or (9 < i and 9 < j)
                        )
                    )
                ):
                    rect = patches.Rectangle(
                        (i - 0.5, j - 0.5),
                        1,
                        1,
                        linewidth=1,
                        edgecolor="black",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

        plt.savefig(os.path.join(dir, "{}.pdf".format(id)), bbox_inches="tight")
