"""
This script can be used to evaluate the model checkpoints saved in

* `PATH/TO/THIS/PROJECT/trained_models/final_furnlift_ckpts` or
* `PATH/TO/THIS/PROJECT/trained_models/final_furnmove_ckpts`

using the fixed evaluation datasets saved in

* `PATH/TO/THIS/PROJECT/data/furnlift_episode_start_positions_for_eval__test.json`,
* `PATH/TO/THIS/PROJECT/data/furnmove_episode_start_positions_for_eval__3agents__test.json`,
* `PATH/TO/THIS/PROJECT/data/furnmove_episode_start_positions_for_eval__test.json`, or
* `PATH/TO/THIS/PROJECT/data/furnmove_episode_start_positions_for_eval__train.json`.

The above checkpoints can either be trained and created by you or downloaded by following
the directions in the README.md. Likewise, you can download the above json files by following
the directions in the README.md or generate new starting evaluation datasets by running
`generate_furnlift_starting_locations_for_evaluation.py` and
`generate_furnmove_starting_locations_for_evaluation.py` appropriately. Note, do to randomness
in generating these evaluation datasets, we DO NOT GUARANTEE that running the above scripts
will result in the same `*.json` that we used in our experiments. If you wish to directly
compare to our results, please download the above json files. To run this script, run

```
python rl_multi_agent/scripts/generate_furnmove_starting_locations_for_evaluation.py furnmove
```
where you will want to change `furnmove` to `furnlift` if you wish to run the FurnLift evaluations.
"""
import glob
import os
import subprocess
import sys
import time

import torch

from constants import ABS_PATH_TO_LOCAL_THOR_BUILD, PROJECT_TOP_DIR
from utils.misc_util import NonBlockingStreamReader


def run(args):
    thor_build = ABS_PATH_TO_LOCAL_THOR_BUILD
    thor_build = os.path.abspath(thor_build)
    split_command = (
        ["pipenv", "run", "python"]
        + args
        + ["--local_thor_build", thor_build, "--x_display", "0.0"]
    )
    print("Command:\n" + " ".join(split_command))
    p = subprocess.Popen(
        split_command,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        stdout=subprocess.PIPE,
    )
    nbsr = NonBlockingStreamReader(p.stdout)
    last_time = time.time()
    while p.poll() is None:
        l = nbsr.readline(1)
        if l:
            print(l.rstrip(), flush=True)
            if "All done." in l:
                p.kill()
                return
        time.sleep(0.02)

        if time.time() - last_time > 60:
            last_time = time.time()
            print("PID of process currently running: {}".format(p.pid), flush=True)
    while True:
        l = nbsr.readline(1)
        if l:
            print(l.rstrip(), flush=True)
        else:
            break
    print(p.stdout.read())


eval_args_string = (
    "main.py"
    " --experiment_dir {eval_experiments_dir}"
    " --task {task}"
    " --num_steps 50"
    " --val_timeout 0"
    " --enable_val_agent f"
    " --enable_logging f"
    " --save_freq 100000"
    " --seed 1"
    " --gpu_ids {gpu_ids_string}"
    " --amsgrad t"
    " --use_episode_init_queue t"
    " --save_extra_data t"
    " --skip_backprop t"
    " --workers {workers}"
    " --verbose t"
)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        eval_experiments_dir = "rl_multi_agent/furnmove_eval_experiments"
    else:
        eval_experiments_dir = "rl_multi_agent/{}_eval_experiments".format(
            sys.argv[1].strip().lower()
        )

    os.chdir(PROJECT_TOP_DIR)
    gpu_ids_string = (
        "-1"
        if not torch.cuda.is_available()
        else " ".join([str(i) for i in range(1, 8)])
    )
    workers = 4 if not torch.cuda.is_available() else 40

    results = []
    for config_path in sorted(
        glob.glob(os.path.join(eval_experiments_dir, "/*_config.py"))
    ):
        task = os.path.basename(config_path).replace("_config.py", "")

        try:
            run(
                eval_args_string.format(
                    eval_experiments_dir=eval_experiments_dir,
                    task=task,
                    gpu_ids_string=gpu_ids_string,
                    workers=workers,
                ).split(" ")
            )
            results.append("{} likely success".format(task))
        except subprocess.CalledProcessError as cpe:
            results.append("{} failure".format(task))

    print("Summary:")
    print("\n".join(results))
