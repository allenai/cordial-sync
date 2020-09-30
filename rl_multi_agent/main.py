from __future__ import print_function, division

import collections
import copy
import ctypes
import importlib
import os
import queue
import random
import signal
import sys
import time
import warnings
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle
from tensorboardX import SummaryWriter
from torch import nn

from rl_base.shared_optim import SharedRMSprop, SharedAdam
from rl_multi_agent.test_process_runner import test
from rl_multi_agent.train_process_runner import train
from utils import flag_parser
from utils.misc_util import save_project_state_in_log
from utils.net_util import (
    ScalarMeanTracker,
    load_model_from_state_dict,
    TensorConcatTracker,
)

np.set_printoptions(threshold=10000000, suppress=True, linewidth=100000)

os.environ["OMP_NUM_THREADS"] = "1"

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == "__main__":
    ptitle("Train/Test Manager")
    args = flag_parser.parse_arguments()
    task = args.task
    task = task.replace("-", "_")
    if task[-7:] == "_config":
        task = task[:-7]
    if task[-10:] == "_config.py":
        task = task[:-10]

    module = importlib.import_module(
        "{}.{}_config".format(
            args.experiment_dir.replace("/", "."), task.replace("-", "_")
        )
    )

    experiment = module.get_experiment()
    experiment.init_train_agent.env_args = args
    experiment.init_test_agent.env_args = args

    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H-%M-%S", time.localtime(start_time)
    )

    if args.enable_logging:
        # Caching current state of the project
        log_file_path = save_project_state_in_log(
            sys.argv,
            task + ("" if args.tag == "" else "/{}".format(args.tag)),
            local_start_time_str,
            None
            if experiment.saved_model_path is None
            else (experiment.saved_model_path,),
            args.log_dir,
        )
        # Create a tensorboard logger
        log_writer = SummaryWriter(log_file_path)

        for arg in vars(args):
            log_writer.add_text("call/" + arg, str(getattr(args, arg)), 0)
        s = ""
        for arg in vars(args):
            s += "--" + arg + " "
            s += str(getattr(args, arg)) + " "
        log_writer.add_text("call/full", s, 0)
        log_writer.add_text("log-path/", log_file_path, 0)
        if experiment.saved_model_path is not None:
            log_writer.add_text("model-load-path", experiment.saved_model_path, 0)

    # Seed (hopefully) all sources of randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
    mp = mp.get_context("spawn")

    if any(gpu_id >= 0 for gpu_id in args.gpu_ids):
        assert torch.cuda.is_available(), (
            f"You have specified gpu_ids=={args.gpu_ids} but no GPUs are available."
            " Please check that your machine has GPUs installed and that the correct (GPU compatible)"
            " version of torch has been installed."
        )

    shared_model: nn.Module = experiment.create_model()

    optimizer_state = None
    restarted_from_episode = None
    if experiment.saved_model_path is not None:
        path = experiment.saved_model_path
        saved_state = torch.load(path, map_location=lambda storage, loc: storage)
        print("\nLoading pretrained weights from {}...".format(path))
        if "model_state" in saved_state:
            load_model_from_state_dict(shared_model, saved_state["model_state"])
            optimizer_state = saved_state["optimizer_state"]
            restarted_from_episode = saved_state["episodes"]
        else:
            load_model_from_state_dict(shared_model, saved_state)
        print("Done.")

    shared_model.share_memory()
    pytorch_total_params = sum(p.numel() for p in shared_model.parameters())
    pytorch_total_trainable_params = sum(
        p.numel() for p in shared_model.parameters() if p.requires_grad
    )
    print("pytorch_total_params:" + str(pytorch_total_params))
    print("pytorch_total_trainable_params:" + str(pytorch_total_trainable_params))

    if not args.shared_optimizer:
        raise NotImplementedError("Must use shared optimizer.")

    optimizer: Optional[torch.optim.Optimizer] = None
    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(
                filter(lambda param: param.requires_grad, shared_model.parameters()),
                lr=args.lr,
                saved_state=optimizer_state,
            )
        elif args.optimizer == "Adam":
            optimizer = SharedAdam(
                filter(lambda param: param.requires_grad, shared_model.parameters()),
                lr=args.lr,
                amsgrad=args.amsgrad,
                saved_state=optimizer_state,
            )
        else:
            raise NotImplementedError(
                "Must choose a shared optimizer from 'RMSprop' or 'Adam'."
            )

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)
    train_scalars = ScalarMeanTracker()
    train_tensors = TensorConcatTracker()
    train_total_ep = mp.Value(
        ctypes.c_int32, restarted_from_episode if restarted_from_episode else 0
    )

    train_res_queue = mp.Queue()
    assert (
        args.x_display is None or args.x_displays is None
    ), "One of x_display ({}) and x_displays ({}) must not have a value.".format(
        args.x_display, args.x_displays
    )

    save_data_queue = None if not args.save_extra_data else mp.Queue()
    episode_init_queue = (
        None
        if not args.use_episode_init_queue
        else experiment.create_episode_init_queue(mp_module=mp)
    )

    if experiment.stopping_criteria_reached():
        warnings.warn("Stopping criteria reached before any computations started!")
        print("All done.")
        sys.exit()

    for rank in range(0, args.workers):
        train_experiment = copy.deepcopy(experiment)
        train_experiment.init_train_agent.seed = random.randint(0, 10 ** 10)
        if args.x_displays is not None:
            args = copy.deepcopy(args)
            args.x_display = args.x_displays[rank % len(args.x_displays)]
            train_experiment.init_train_agent.env_args = args
            train_experiment.init_test_agent.env_args = args
        p = mp.Process(
            target=train,
            args=(
                rank,
                args,
                shared_model,
                train_experiment,
                optimizer,
                train_res_queue,
                end_flag,
                train_total_ep,
                None,  # Update lock
                save_data_queue,
                episode_init_queue,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.2)

    time.sleep(5)

    valid_res_queue = mp.Queue()
    valid_total_ep = mp.Value(
        ctypes.c_int32, restarted_from_episode if restarted_from_episode else 0
    )
    if args.enable_val_agent:
        test_experiment = copy.deepcopy(experiment)
        if args.x_displays is not None:
            args = copy.deepcopy(args)
            args.x_display = args.x_displays[-1]
            test_experiment.init_train_agent.env_args = args
            test_experiment.init_test_agent.env_args = args
        p = mp.Process(
            target=test,
            args=(args, shared_model, test_experiment, valid_res_queue, end_flag, 0, 1),
        )
        p.start()
        processes.append(p)
        time.sleep(0.2)

    time.sleep(1)

    train_thin = 500
    valid_thin = 1
    n_frames = 0
    try:
        while (
            (not experiment.stopping_criteria_reached())
            and any(p.is_alive() for p in processes)
            and train_total_ep.value < args.max_ep
        ):
            try:
                train_result = train_res_queue.get(timeout=10)
                if len(train_result) != 0:
                    train_scalars.add_scalars(train_result)
                    train_tensors.add_tensors(train_result)
                    ep_length = sum(
                        train_result[k] for k in train_result if "ep_length" in k
                    )
                    train_total_ep.value += 1
                    n_frames += ep_length

                    if args.enable_logging and train_total_ep.value % train_thin == 0:
                        tracked_means = train_scalars.pop_and_reset()
                        for k in tracked_means:
                            log_writer.add_scalar(
                                k + "/train", tracked_means[k], train_total_ep.value
                            )
                        if train_total_ep.value % (20 * train_thin) == 0:
                            tracked_tensors = train_tensors.pop_and_reset()
                            for k in tracked_tensors:
                                log_writer.add_histogram(
                                    k + "/train",
                                    tracked_tensors[k],
                                    train_total_ep.value,
                                )

                    if args.enable_logging and train_total_ep.value % (10 * train_thin):
                        log_writer.add_scalar(
                            "n_frames", n_frames, train_total_ep.value
                        )

                if args.enable_logging and args.enable_val_agent:
                    while not valid_res_queue.empty():
                        valid_result = valid_res_queue.get()
                        if len(valid_result) == 0:
                            continue
                        key = list(valid_result.keys())[0].split("/")[0]
                        valid_total_ep.value += 1
                        if valid_total_ep.value % valid_thin == 0:
                            for k in valid_result:
                                if np.isscalar(valid_result[k]):
                                    log_writer.add_scalar(
                                        k + "/valid",
                                        valid_result[k],
                                        train_total_ep.value,
                                    )
                                elif isinstance(valid_result[k], collections.Iterable):
                                    log_writer.add_histogram(
                                        k + "/train",
                                        valid_result[k],
                                        train_total_ep.value,
                                    )

                # Saving extra data (if any)
                if save_data_queue is not None:
                    while True:
                        try:
                            experiment.save_episode_summary(
                                data_to_save=save_data_queue.get(timeout=0.2)
                            )
                        except queue.Empty as _:
                            break

                # Checkpoints
                if (
                    train_total_ep.value == args.max_ep
                    or (train_total_ep.value % args.save_freq) == 0
                ):
                    if not os.path.exists(args.save_model_dir):
                        os.makedirs(args.save_model_dir)

                    state_to_save = shared_model.state_dict()
                    save_path = os.path.join(
                        args.save_model_dir,
                        "{}_{}_{}.dat".format(
                            task, train_total_ep.value, local_start_time_str
                        ),
                    )
                    torch.save(
                        {
                            "model_state": shared_model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "episodes": train_total_ep.value,
                        },
                        save_path,
                    )
            except queue.Empty as _:
                pass
    finally:
        end_flag.value = True
        print(
            "Stopping criteria reached: {}".format(
                experiment.stopping_criteria_reached()
            ),
            flush=True,
        )
        print(
            "Any workers still alive: {}".format(any(p.is_alive() for p in processes)),
            flush=True,
        )
        print(
            "Reached max episodes: {}".format(train_total_ep.value >= args.max_ep),
            flush=True,
        )

        if args.enable_logging:
            log_writer.close()
        for p in processes:
            p.join(0.1)
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)

        print("All done.", flush=True)
