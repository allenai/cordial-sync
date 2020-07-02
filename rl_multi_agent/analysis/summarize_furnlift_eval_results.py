"""
A script to summarize data saved in the `data/furnlift_evaluations__test/` directory,
as a result of downloading data or running evaluation using the script:
`rl_multi_agent/scripts/run_furnmove_or_furnlift_evaluations.py`

Set the metrics, methods, dataset split and generate a csv-styled table of metrics and confidence
intervals.

Run using command:
`python rl_multi_agent/analysis/summarize_furnlift_eval_results.py`

Also, see `rl_multi_agent/analysis/summarize_furnmove_eval_results.py` for a similar script for
FurnMove task.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from constants import ABS_PATH_TO_DATA_DIR, EASY_MAX, MED_MAX

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def create_table(id_to_df, metrics_to_record, split: str = "all"):
    if split == "easy":
        min_dist = -1
        max_dist = EASY_MAX
    elif split == "medium":
        min_dist = EASY_MAX + 1
        max_dist = MED_MAX
    elif split == "hard":
        min_dist = MED_MAX + 1
        max_dist = 1000
    elif split == "all":
        min_dist = -1
        max_dist = 10000
    else:
        raise NotImplementedError()

    sorted_ids = sorted(list(id_to_df.keys()))

    result_means = defaultdict(lambda: [])
    result_95_confs = defaultdict(lambda: [])

    for id in sorted_ids:
        df = id_to_df[id].query(
            "initial_manhattan_steps <= {max_dist} and initial_manhattan_steps > {min_dist}".format(
                max_dist=max_dist, min_dist=min_dist
            )
        )

        for metric_name in metrics_to_record:
            values = np.array(df[metric_name])
            result_means[metric_name].append(np.mean(values))
            result_95_confs[metric_name].append(
                1.96 * np.std(values) / math.sqrt(len(values))
            )

    means_df = pd.DataFrame(dict(result_means), index=sorted_ids)
    confs_df = pd.DataFrame(dict(result_95_confs), index=sorted_ids)
    return means_df.round(6), confs_df.round(6)


if __name__ == "__main__":
    # Set constants for summarizing results.
    # Select metrics to summarize
    metrics_to_record = [
        "spl_manhattan",
        "accuracy",
        "ep_length",
        "final_distance",
        "invalid_prob_mass",
        "tvd",
        "picked_but_not_pickupable",
        "pickupable_but_not_picked",
    ]
    # Listing all methods
    vision_multi_nav = [
        "vision_noimplicit_central_cl",
        "vision_noimplicit_marginal_nocl",
        "vision_noimplicit_marginalnocomm_nocl",
        "vision_noimplicit_mixture_cl",
    ]
    # Split of the dataset. The relevant directory would be searched for saved evaluation information
    split = "test"
    # Which methods to summarize
    subset_interested_in = vision_multi_nav
    # Summarize absolute values, 95% confidence intervals, and show them with a (val \pm conf) schema
    # The last is useful to input into a latex table.
    display_values = True
    display_95conf = True
    display_combined = False
    id_to_df = {}

    for dir in glob.glob(
        os.path.join(ABS_PATH_TO_DATA_DIR, "furnlift_evaluations__{}/*__*").format(
            split
        )
    ):
        dir_name = os.path.basename(dir)
        id = dir_name.split("__")[1]
        if id not in subset_interested_in:
            continue
        print()
        print("Processing method: {}".format(id))

        recorded = defaultdict(lambda: [])
        for i, p in enumerate(glob.glob(os.path.join(dir, "*.json"))):
            if i % 100 == 0:
                print("Completed {} episodes".format(i))

            with open(p, "r") as f:
                result_dict = json.load(f)

            for k in metrics_to_record + ["initial_manhattan_steps"]:
                if k == "tvd":
                    assert "dist_to_low_rank" in result_dict
                    recorded[k].append(result_dict["dist_to_low_rank"] / 2)
                elif k != "action_success_percent":
                    recorded[k].append(result_dict[k])

        id_to_df[id] = pd.DataFrame(dict(recorded))
        if id_to_df[id].shape[0] == 0:
            del id_to_df[id]

    a, b = create_table(id_to_df, metrics_to_record)
    if display_values:
        # Metrics only
        print(a.round(3).to_csv())
    if display_95conf:
        # Corresponding 95% confidence intervals
        print(b.round(3).to_csv())
    if display_combined:
        # Combining the above
        a_str = a.round(3).to_numpy().astype("str").astype(np.object)
        b_str = b.round(3).to_numpy().astype("str").astype(np.object)
        combined_str = a_str + " pm " + b_str
        print(
            pd.DataFrame(data=combined_str, index=a.index, columns=a.columns).to_csv()
        )
