from __future__ import division

import sys
import traceback
import warnings
from typing import Optional, Dict

import networkx
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from setproctitle import setproctitle as ptitle
from torch import nn

from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.experiment import ExperimentConfig
from rl_multi_agent.multi_agent_utils import (
    TrainingCompleteException,
    EndProcessException,
    compute_losses_and_backprop,
)
from utils.net_util import recursively_detach

warnings.simplefilter("always", UserWarning)


def train(
    worker_number: int,
    args,
    shared_model: nn.Module,
    experiment: ExperimentConfig,
    optimizer: optim.Optimizer,
    res_queue: mp.Queue,
    end_flag: mp.Value,
    train_total_ep: mp.Value,
    update_lock: Optional[mp.Lock] = None,
    save_data_queue: Optional[mp.Queue] = None,
    episode_init_queue: Optional[mp.Queue] = None,
) -> None:
    ptitle("Training Agent: {}".format(worker_number))

    experiment.init_train_agent.worker_number = worker_number
    gpu_id = args.gpu_ids[worker_number % len(args.gpu_ids)]
    torch.manual_seed(args.seed + worker_number)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            torch.cuda.manual_seed(args.seed + worker_number)

    model: nn.Module = experiment.create_model()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model.load_state_dict(shared_model.state_dict())
    else:
        model.load_state_dict(shared_model.state_dict())

    agent: MultiAgent = experiment.create_agent(model=model, gpu_id=gpu_id)

    num_sequential_errors = 0
    while not end_flag.value:
        last_losses: Dict[str, float] = {}
        additional_metrics = {}

        try:
            experiment.init_train_agent.current_train_episode = train_total_ep.value
            agent.global_episode_count = train_total_ep.value
            agent.sync_with_shared(shared_model)

            while True:
                agent_iterator = experiment.init_train_agent(
                    agent=agent, episode_init_queue=episode_init_queue
                )
                try:
                    next(agent_iterator)
                    break
                except StopIteration as e:
                    warnings.warn("Agent iterator was empty.")

            additional_metrics = {}
            step_results = []
            while not agent.episode.is_complete():
                agent.global_episode_count = train_total_ep.value

                for step in range(args.num_steps):
                    agent.act(train=True)
                    if args.skip_backprop:
                        agent.repackage_hidden()
                        if len(agent.eval_results) != 0:
                            agent.eval_results[-1] = recursively_detach(
                                agent.eval_results[-1]
                            )
                    if agent.episode.is_complete():
                        break

                if end_flag.value:
                    raise EndProcessException()

                last_losses = compute_losses_and_backprop(
                    agent=agent,
                    shared_model=shared_model,
                    optimizer=optimizer,
                    update_lock=update_lock,
                    gpu=gpu_id >= 0,
                    retain_graph=False,
                    skip_backprop=args.skip_backprop,
                )

                if (
                    len(agent.eval_results) > 0
                    and "additional_metrics" in agent.eval_results[0]
                ):
                    for a_metric in agent.eval_results[0]["additional_metrics"]:
                        if a_metric not in additional_metrics:
                            additional_metrics[a_metric] = []
                        additional_metrics[a_metric].extend(
                            er["additional_metrics"][a_metric]
                            for er in agent.eval_results
                        )

                if save_data_queue is not None:
                    step_results.extend(agent.step_results)

                if (not agent.episode.is_complete()) or (not args.skip_backprop):
                    agent.clear_history()
                agent.repackage_hidden()
                agent.sync_with_shared(shared_model)

            additional_metrics = {
                k: float(sum(additional_metrics[k]) / len(additional_metrics[k]))
                for k in additional_metrics
                if len(additional_metrics[k]) != 0
            }

            try:
                next(agent_iterator)
                raise Exception("Agent iterator should only yield once.")
            except StopIteration as _:
                pass

            info = agent.episode.info()

            if save_data_queue is not None:
                data_to_save = experiment.create_episode_summary(
                    agent,
                    additional_metrics=additional_metrics,
                    step_results=step_results,
                )
                if data_to_save is not None:
                    save_data_queue.put(data_to_save)

            if last_losses is not None:
                info = {**info, **last_losses}
            if additional_metrics:
                info = {**info, **additional_metrics}
            res_queue.put({k: info[k] for k in info})

            agent.clear_history()
            agent.repackage_hidden()
            num_sequential_errors = 0

        except EndProcessException as _:
            print(
                "End process signal received in worker {}. Quitting...".format(
                    worker_number
                ),
                flush=True,
            )
            sys.exit()

        except TrainingCompleteException as _:
            print(
                (
                    "Training complete signal received in worker {}. "
                    "Closing open files/subprocesses exiting."
                ).format(worker_number),
                flush=True,
            )
            try:
                agent.environment.stop()
            except Exception as _:
                pass

            sys.exit()

        except (RuntimeError, ValueError, networkx.exception.NetworkXNoPath) as e:
            num_sequential_errors += 1
            if num_sequential_errors == 10:
                print("Too many sequential errors in training.")
                raise e

            if "out of memory at" in str(e):
                raise e

            print(
                "RuntimeError, ValueError, or networkx exception when"
                " training, attempting to print traceback, print metadata, and reset."
            )
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)

            try:
                print(agent.environment.last_event.metadata)
            except NameError:
                print("Unable to print environment metadata.")
            print("-" * 60)

            agent.clear_history()
            agent.repackage_hidden()
