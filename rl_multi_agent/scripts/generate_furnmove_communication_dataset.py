"""
A script which extracts the communication information from FurnMove evaluation episodes
(using agents trained with SYNC policies and the CORDIAL loss) and saves this information
into a TSV file. This TSV file can then be further analyzed (as we have done in our paper)
to study _what_ the agents have learned to communicate with one another.

By default, this script will also save visualizations of the communication between
agents for episodes that are possibly interesting (e.g. not too short and not too long).
If you wish turn off this behavior, change `create_communication_visualizations = True`
to `create_communication_visualizations = False` below.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import ABS_PATH_TO_ANALYSIS_RESULTS_DIR, ABS_PATH_TO_DATA_DIR


def visualize_talk_reply_probs(df: pd.DataFrame, save_path: Optional[str] = None):
    steps = list(range(df.shape[0]))

    a0_tv_visible = np.array(df["a0_tv_visible"])
    a1_tv_visible = np.array(df["a1_tv_visible"])

    a0_action_taken = np.array(df["a0_next_action"])
    a1_action_taken = np.array(df["a1_next_action"])

    a0_took_pass_action = np.logical_and(a0_action_taken == 3, a1_action_taken < 3)
    a1_took_pass_action = np.logical_and(a1_action_taken == 3, a0_action_taken < 3)

    a0_took_mwo_action = np.logical_and(4 <= a0_action_taken, a0_action_taken <= 7)
    a1_took_mwo_action = np.logical_and(4 <= a1_action_taken, a1_action_taken <= 7)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(3, 3))

    fig.subplots_adjust(hspace=0.1)

    # Plot 1
    axs[0].set_ylim((-0.15, 1.15))
    axs[0].set_ylabel("Talk Weight")
    axs[0].plot(steps, df["a00_talk_probs"], color="r", linewidth=0.75)
    axs[0].plot(steps, df["a10_talk_probs"], color="g", linewidth=0.75)

    a0_tv_visible_inds = np.argwhere(a0_tv_visible)
    if len(a0_tv_visible_inds) != 0:
        axs[0].scatter(
            a0_tv_visible_inds,
            [-0.05] * len(a0_tv_visible_inds),
            color="r",
            s=4,
            marker="|",
        )

    a1_tv_visible_inds = np.argwhere(a1_tv_visible)
    if len(a1_tv_visible_inds) != 0:
        axs[0].scatter(
            a1_tv_visible_inds,
            [-0.1] * len(a1_tv_visible_inds),
            color="green",
            s=4,
            marker="|",
        )

    # Plot 2
    axs[1].set_ylim((-0.15, 1.15))
    axs[1].set_ylabel("Reply Weight")
    axs[1].set_xlabel("Steps in Episode")
    axs[1].plot(steps, df["a00_reply_probs"], color="r", linewidth=0.75)
    axs[1].plot(steps, df["a10_reply_probs"], color="g", linewidth=0.75)

    a0_pass_action_steps = np.argwhere(a0_took_pass_action)
    if len(a0_pass_action_steps) != 0:
        axs[1].scatter(
            a0_pass_action_steps,
            [1.1] * len(a0_pass_action_steps),
            color="r",
            s=4,
            marker="|",
        )

    a1_pass_action_steps = np.argwhere(a1_took_pass_action)
    if len(a1_pass_action_steps) != 0:
        axs[1].scatter(
            a1_pass_action_steps,
            [1.05] * len(a1_pass_action_steps),
            color="g",
            s=4,
            marker="|",
        )

    a0_mwo_action = np.argwhere(a0_took_mwo_action)
    if len(a0_mwo_action) != 0:
        axs[1].scatter(
            a0_mwo_action, [-0.05] * len(a0_mwo_action), color="r", s=4, marker="|",
        )

    a1_mwo_action = np.argwhere(a1_took_mwo_action)
    if len(a1_mwo_action) != 0:
        axs[1].scatter(
            a1_mwo_action, [-0.1] * len(a1_mwo_action), color="g", s=4, marker="|",
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    create_communication_visualizations = True

    dir = os.path.join(
        ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__test/vision_mixture_cl_rot"
    )
    id = dir.split("__")[-2]
    print()
    print(id)

    recorded = defaultdict(lambda: [])
    for i, p in enumerate(sorted(glob.glob(os.path.join(dir, "*.json")))):
        if i % 100 == 0:
            print(i)

        with open(p, "r") as f:
            result_dict = json.load(f)

        recorded["a00_talk_probs"].extend(
            [probs[0] for probs in result_dict["a0_talk_probs"]]
        )
        recorded["a10_talk_probs"].extend(
            [probs[0] for probs in result_dict["a1_talk_probs"]]
        )

        recorded["a00_reply_probs"].extend(
            [probs[0] for probs in result_dict["a0_reply_probs"]]
        )
        recorded["a10_reply_probs"].extend(
            [probs[0] for probs in result_dict["a1_reply_probs"]]
        )

        ar0, ar1 = result_dict["agent_action_info"]

        for j, srs in enumerate(result_dict["step_results"]):
            sr0, sr1 = srs
            before_loc0 = sr0["before_location"]
            before_loc1 = sr1["before_location"]
            recorded["from_a0_to_a1"].append(
                round((before_loc0["rotation"] - before_loc1["rotation"]) / 90) % 4
            )
            recorded["from_a1_to_a0"].append((-recorded["from_a0_to_a1"][-1]) % 4)

            recorded["a0_next_action"].append(sr0["action"])
            recorded["a1_next_action"].append(sr1["action"])

            recorded["a0_action_success"].append(1 * sr0["action_success"])
            recorded["a1_action_success"].append(1 * sr1["action_success"])

            if j == 0:
                recorded["a0_last_action_success"].append(1)
                recorded["a1_last_action_success"].append(1)
            else:
                old0, old1 = result_dict["step_results"][j - 1]
                recorded["a0_last_action_success"].append(1 * old0["action_success"])
                recorded["a1_last_action_success"].append(1 * old1["action_success"])

            e0 = sr0["extra_before_info"]
            e1 = sr1["extra_before_info"]

            recorded["a0_tv_visible"].append(1 * e0["tv_visible"])
            recorded["a1_tv_visible"].append(1 * e1["tv_visible"])

            recorded["a0_dresser_visible"].append(1 * e0["dresser_visible"])
            recorded["a1_dresser_visible"].append(1 * e1["dresser_visible"])

        recorded["index"].extend([i] * len(result_dict["a0_talk_probs"]))

    recorded = dict(recorded)
    df = pd.DataFrame(recorded)

    df.to_csv(
        os.path.join(
            ABS_PATH_TO_DATA_DIR,
            "furnmove_communication_analysis/furnmove_talk_reply_dataset.tsv",
        ),
        sep="\t",
    )

    if create_communication_visualizations:
        print("Creating communication visualizations")
        k = 0
        while True:
            print(k)
            k += 1
            subdf = df.query("index == {}".format(k))
            if 60 < subdf.shape[0] < 80 and subdf.shape[0] != 250:

                visualize_talk_reply_probs(
                    subdf,
                    save_path=os.path.join(
                        ABS_PATH_TO_ANALYSIS_RESULTS_DIR,
                        "plots/furnmove_communication/{}.pdf",
                    ).format(k),
                )
