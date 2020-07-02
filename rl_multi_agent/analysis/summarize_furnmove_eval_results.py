"""
A script to summarize data saved in the `data/furnmove_evaluations__test/` directory,
as a result of downloading data or running evaluation using the script:
`rl_multi_agent/scripts/run_furnmove_or_furnlift_evaluations.py`

Set the metrics, methods, dataset split and generate a csv-styled table of metrics and confidence
intervals. Particularly, which of the methods:
* grid_vision_furnmove: Four prominent methods (no comm, marginal, SYNC, central) for gridworld and
visual environments.
* grid_3agents: Marginal, SYNC, and central methods for three agent gridworld-FurnMove setting.
* vision_mixtures: Effect of number of mixture components m on SYNC’s performance (in FurnMove)
* vision_cl_ablation: Effect of CORDIAL (`cl`) on marginal, SYNC, and central methods.
* vision_3agents: Marginal, SYNC, and central methods for three agent FurnMove setting.

Run using command:
`python rl_multi_agent/analysis/summarize_furnmove_eval_results.py`

Also, see `rl_multi_agent/analysis/summarize_furnlift_eval_results.py` for a similar script for
FurnLift task.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from constants import ABS_PATH_TO_DATA_DIR
from rl_multi_agent.analysis.summarize_furnlift_eval_results import create_table

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

if __name__ == "__main__":
    # Set constants for summarizing results.
    # Select metrics to summarize
    metrics_to_record = [
        "spl_manhattan",
        "reached_target",
        "ep_length",
        "final_distance",
        "invalid_prob_mass",
        "tvd",
        # "reward",
    ]
    # Split of the dataset. The relevant directory would be searched for saved evaluation information
    split = "test"

    id_to_df = {}
    # Listing all methods
    grid_vision_furnmove = [
        "grid_bigcentral_cl_rot",
        "grid_marginal_nocl_rot",
        "grid_marginalnocomm_nocl_rot",
        "grid_mixture_cl_rot",
        "vision_bigcentral_cl_rot",
        "vision_marginal_nocl_rot",
        "vision_marginalnocomm_nocl_rot",
        "vision_mixture_cl_rot",
    ]
    grid_3agents = [
        "grid_central_3agents",
        "grid_marginal_3agents",
        "grid_mixture_3agents",
    ]
    vision_mixtures = [
        "vision_marginal_cl_rot",
        "vision_mixture2mix_cl_rot",
        "vision_mixture4mix_cl_rot",
        "vision_mixture_cl_rot",
    ]
    vision_cl_ablation = [
        "vision_marginal_nocl_rot",
        "vision_marginal_cl_rot",
        "vision_bigcentral_nocl_rot",
        "vision_bigcentral_cl_rot",
        "vision_mixture_nocl_rot",
        "vision_mixture_cl_rot",
    ]
    vision_3agents = [
        "vision_central_3agents",
        "vision_marginal_3agents",
        "vision_mixture_3agents",
    ]
    # Which methods to summarize
    subset_interested_in = grid_vision_furnmove
    # Summarize absolute values, 95% confidence intervals, and show them with a (val \pm conf) schema
    # The last is useful to input into a latex table.
    display_values = True
    display_95conf = True
    display_combined = False

    for dir in glob.glob(
        os.path.join(ABS_PATH_TO_DATA_DIR, "furnmove_evaluations__{}/*").format(split)
    ):
        id = os.path.basename(dir)
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

            recorded["action_success_percent"].append(
                np.array(result_dict["action_taken_success_matrix"]).sum()
                / (result_dict["ep_length"] // 2)
            )

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
