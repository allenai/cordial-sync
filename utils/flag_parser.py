import argparse

from constants import ABS_PATH_TO_LOCAL_THOR_BUILD


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="A3C")
    parser.add_argument(
        "--task",
        type=str,
        help="The experiment config to run (e.g. furnmove_grid_bigcentral_cl_rot, furnmove_grid_mixture_3agents, etc.)."
        "`--experiment_dir` should be used to specify the directory where this config resides.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="rl_multi_agent/experiments",
        help="The directory to look in to find the config file for this run. (default `rl_multi_agent/experiments`)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="A tag for the run (e.g. lstm-not-gru, trying-new-thing). If not-empty, this tag is used as a subdirectory"
        "along the tensorboard path. (default: '')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="Random seed. As A3C is asynchronous, setting this seed has "
        "does not guarantee any exact reproducibility from run to run. (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        metavar="W",
        help="How many training processes to use. (default: 32)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        metavar="NS",
        help="Number of forward steps in A3C before computing the loss and backproping. (default: 50)",
    )
    parser.add_argument(
        "--shared_optimizer",
        default=True,
        metavar="SO",
        type=str2bool,
        help="use an optimizer with shared statistics. (default: True)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1e6,
        help="Save model after this # of training episodes. (default: 1e+6)",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        metavar="OPT",
        help="Optimizer choice (must be Adam or RMSprop). (default: Adam)",
    )
    parser.add_argument(
        "--save_model_dir",
        default="trained_models/",
        metavar="SMD",
        help="Folder to save trained model checkpoints. (default: trained_models)",
    )
    parser.add_argument(
        "--log_dir",
        default="logs/",
        metavar="LG",
        help="Folder in which to save (tensorboard) logs. (default: logs)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        default=-1,
        nargs="+",
        help="GPUs to use [-1 CPU only] (default: -1)",
    )
    parser.add_argument(
        "--amsgrad",
        type=str2bool,
        default=True,
        metavar="AM",
        help="Adam optimizer amsgrad parameter. (default: True)",
    )
    parser.add_argument(
        "--docker_enabled",
        action="store_true",
        help="Whether or not to use docker."
        " This flag should not be used unless you know what you're doing.",
    )
    parser.add_argument(
        "--x_display",
        type=str,
        default=None,
        help=(
            "The X display to target, if any. If targeting a multiple displays"
            "please use the x_displays argument."
        ),
    )
    parser.add_argument(
        "--x_displays",
        type=str,
        default=None,
        nargs="+",
        help="The x-displays to target, if any.",
    )
    parser.add_argument(
        "--val_timeout",
        type=float,
        default=10,
        help="The length of time to wait in between validation episodes. (default: 10)",
    )
    parser.add_argument(
        "--enable_val_agent",
        type=str2bool,
        default=True,
        help="Whether or not to use an agent to validate results while training. (default: True)",
    )
    parser.add_argument(
        "--enable_logging",
        type=str2bool,
        default=True,
        help="Whether or not to record logging information (e.g. tensorboard logs). (default: True)",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="If true, validation agent will print more information. (default: False)",
    )
    parser.add_argument(
        "--skip_backprop",
        type=str2bool,
        default=False,
        help="If true, will not backprop during training. Useful when debugging. (default: False)",
    )
    parser.add_argument(
        "--max_ep",
        type=float,
        default=float("inf"),
        help="Maximum # of episodes to run when training. (default: 'inf')",
    )
    parser.add_argument(
        "--local_thor_build",
        type=str,
        default=ABS_PATH_TO_LOCAL_THOR_BUILD,
        help="A path to a local thor build to use if desired. (default: {})".format(
            ABS_PATH_TO_LOCAL_THOR_BUILD
        ),
    )
    parser.add_argument(
        "--visualize_test_agent",
        type=str2bool,
        default=False,
        help="Whether or not to create plots and graphics for test agent runs. (default: False)",
    )
    parser.add_argument(
        "--test_gpu_ids",
        type=int,
        default=None,
        nargs="+",
        help="GPUs to use for test agents [-1 CPU only]. (default: -1)",
    )
    parser.add_argument(
        "--use_episode_init_queue",
        type=str2bool,
        default=False,
        help="If True, attempts to use the episode init queue. This is necessary when evaluating models on fixed"
        "datasets. Search this codebase for the `create_episode_init_queue` for more information."
        " (default: False)",
    )
    parser.add_argument(
        "--save_extra_data",
        type=str2bool,
        default=False,
        help="If true, attempt to save extra data from train processes. (default: False)",
    )
    return parser.parse_args()
