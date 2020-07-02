"""
This script can be used to reproduce the training curves presented in our paper assuming you
have followed the directions in our README.md to download the `data` directory. The plots will
be saved in the `PATH/TO/THIS/PROJECT/analysis_output/plots` directory.
"""

import json
import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
from skmisc.loess import loess

from constants import ABS_PATH_TO_ANALYSIS_RESULTS_DIR, ABS_PATH_TO_DATA_DIR
from utils.misc_util import unzip

DIR_TO_SAVE_PROCESSED_OUTPUT = os.path.join(
    ABS_PATH_TO_DATA_DIR, "furnmove_and_furnlift_logs_processed_to_jsons/"
)
os.makedirs(DIR_TO_SAVE_PROCESSED_OUTPUT, exist_ok=True)

# Vision
VISION_ID_TO_NAME = {
    "vision_bigcentral_cl_rot": "Central",
    "vision_mixture_cl_rot": "SYNC",
    "vision_marginal_nocl_rot": "Marginal",
    "vision_marginalnocomm_nocl_rot": "Marginal (w/o comm)",
}

# Grid
GRID_ID_TO_NAME = {
    "grid_bigcentral_cl_rot": "Central",
    "grid_mixture_cl_rot": "SYNC",
    "grid_marginal_nocl_rot": "Marginal",
    "grid_marginalnocomm_nocl_rot": "Marginal (w/o comm)",
}

# FurnLift vision noimplicit
FURNLIFT_VISION_TO_NAME = {
    "furnlift__vision_noimplicit_central_cl": "Central",
    "furnlift__vision_noimplicit_mixture_cl": "SYNC",
    "furnlift__vision_noimplicit_marginal_nocl": "Marginal",
    "furnlift__vision_noimplicit_marginalnocomm_nocl": "Marginal (w/o comm)",
}

METRIC_ID_TO_NAME = {
    "reached_target": "Reached Target Successfully",
    "ep_length": "Episode Length",
    "accuracy": "Find and Lift Success",
    "spl_manhattan": "MD-SPL",
    "pickupable_but_not_picked": "Pickupable But Not Picked",
    "picked_but_not_pickupable": "Picked But Not Pickupable",
    "invalid_prob_mass": "Invalid Probability Mass",
    "dist_to_low_rank": "TVD",
    "reward": "Reward",
}

HEAD_NAME_TO_COLOR = {
    "Central": "red",
    "SYNC": "blue",
    "Marginal": "green",
    "Marginal (w/o comm)": "black",
}


def plot_metric_for_saved_logs(id_to_name, metric_id, max_step, max_points=50000):
    for id in id_to_name:
        file_name = "furnmove_{}_info.json".format(id)
        try:
            with open(os.path.join(DIR_TO_SAVE_PROCESSED_OUTPUT, file_name), "r") as f:
                tb_data = json.load(f)
            span = 0.1
        except Exception as _:
            file_name = "{}_info.json".format(id)
            span = 0.1
            try:
                with open(
                    os.path.join(DIR_TO_SAVE_PROCESSED_OUTPUT, file_name), "r"
                ) as f:
                    tb_data = json.load(f)
            except Exception as _:
                raise RuntimeError("Could not find json log file with id {}".format(id))

        name = id_to_name[id]
        color = HEAD_NAME_TO_COLOR[name]

        for split in ["train", "valid"]:
            print(split)
            is_train = split == "train"

            tb_split_data = tb_data[split]

            if metric_id not in tb_split_data:
                warnings.warn(
                    "{} does not exist for metric {} and method {}".format(
                        split, METRIC_ID_TO_NAME[metric_id], name
                    )
                )
                continue
            x, y = unzip(tb_split_data[metric_id])
            x = np.array(x)

            inds = np.arange(0, len(x) - 1, max(len(x) // max_points, 1))
            inds = inds[x[inds] <= max_step]
            x, y = np.array(x)[inds], np.array(y)[inds]
            x = x / 1000

            print("Fitting loess")
            l = loess(x, y, span=span, degree=1)
            l.fit()

            pred = l.predict(x, stderror=True)
            conf = pred.confidence(alpha=0.05)

            y_pred = (
                pred.values if metric_id == "reward" else np.maximum(pred.values, 0)
            )
            ll = conf.lower if metric_id == "reward" else np.maximum(conf.lower, 0)
            ul = conf.upper

            print("Plotting line")
            plt.plot(
                x,
                y_pred,
                color=color,
                dashes=[] if is_train else [10, 5],  # [5, 5, 5, 5],
                linewidth=0.75 if is_train else 0.5,
                alpha=1,
                label=name if is_train else None,
            )
            print("Plotting fill")
            plt.fill_between(
                x,
                ll,
                ul,
                alpha=0.10 * (2 if is_train else 1),
                color=color,
                linewidth=0.0,
            )

    plt.ylabel(METRIC_ID_TO_NAME[metric_id])
    plt.xlabel("Training Episodes (Thousands)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin, ymax = ax.get_ylim()
    if ymax < 1 and "spl" not in metric_id:
        plt.scatter([1], [1], alpha=0)

    # ax.spines["left"].set_smart_bounds(True)
    # ax.spines["bottom"].set_smart_bounds(True)
    return ax


if __name__ == "__main__":
    os.makedirs(os.path.join(ABS_PATH_TO_ANALYSIS_RESULTS_DIR, "plots"), exist_ok=True)
    figsize = (4, 3)
    # Do you wish to overwrite existing plots?
    overwrite = True
    # Select tasks for which metrics are to be plotted.
    task_interested_in = ["furnlift_vision", "vision", "grid"]
    for type in task_interested_in:
        for metric_id in (
            [
                "reached_target",
                "ep_length",
                "spl_manhattan",
                "invalid_prob_mass",
                "dist_to_low_rank",
                "reward",
            ]
            if type != "furnlift_vision"
            else [
                "pickupable_but_not_picked",
                "picked_but_not_pickupable",
                "reward",
                "ep_length",
                "accuracy",
                "invalid_prob_mass",
                "dist_to_low_rank",
            ]
        ):
            plot_save_path = os.path.join(
                ABS_PATH_TO_ANALYSIS_RESULTS_DIR, "plots/{}__{}.pdf"
            ).format(type, metric_id)

            if (not overwrite) and os.path.exists(plot_save_path):
                continue
            print("\n" * 3)
            print(metric_id, type)
            print("\n")

            if type == "vision":
                id_to_name = VISION_ID_TO_NAME
                max_step = 500000
            elif type == "grid":
                id_to_name = GRID_ID_TO_NAME
                max_step = 1000000
            elif type == "furnlift_vision":
                id_to_name = FURNLIFT_VISION_TO_NAME
                max_step = 100000
            else:
                raise NotImplementedError

            plt.figure(figsize=figsize)
            try:
                plot_metric_for_saved_logs(
                    id_to_name=id_to_name,
                    metric_id=metric_id,
                    max_step=max_step,
                    max_points=20000,
                )
                if any(k in metric_id for k in ["reached", "accuracy"]):
                    plt.legend(loc="best")
                plt.savefig(plot_save_path, bbox_inches="tight")
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Continuing")
            finally:
                plt.close()
