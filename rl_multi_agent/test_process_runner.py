from __future__ import division

import random
import sys
import time
import traceback
import warnings

from rl_multi_agent.furnmove_utils import save_agents_path_without_frame_png
from rl_multi_agent.multi_agent_utils import (
    compute_losses_no_backprop,
    EndProcessException,
)

warnings.simplefilter("always", UserWarning)

from setproctitle import setproctitle as ptitle
from typing import Optional

import networkx
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import utils.debug_util as debug_util
import matplotlib as mpl

mpl.use("Agg")
from rl_multi_agent import MultiAgent
from rl_multi_agent.experiments.experiment import ExperimentConfig

from utils.net_util import recursively_detach

np.set_printoptions(linewidth=1000)


def test(
    args,
    shared_model: nn.Module,
    experiment: ExperimentConfig,
    res_queue: mp.Queue,
    end_flag: mp.Value,
    worker_num: Optional[int] = None,
    num_test_workers: Optional[int] = None,
) -> None:
    if worker_num is not None:
        ptitle("Test Agent {}".format(worker_num))
        if args.test_gpu_ids is not None:
            gpu_id = args.test_gpu_ids[worker_num % len(args.test_gpu_ids)]
        else:
            gpu_id = args.gpu_ids[worker_num % len(args.gpu_ids)]
    else:
        ptitle("Test Agent")
        if args.test_gpu_ids is not None:
            gpu_id = args.test_gpu_ids[-1]
        else:
            gpu_id = args.gpu_ids[-1]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    if "set_worker_num" in dir(experiment.init_test_agent):
        experiment.init_test_agent.set_worker_num(
            worker_num=worker_num, total_workers=num_test_workers
        )

    model: nn.Module = experiment.create_model()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model.load_state_dict(shared_model.state_dict())
    else:
        model.load_state_dict(shared_model.state_dict())
    agent: MultiAgent = experiment.create_agent(model=model, gpu_id=gpu_id)

    res_sampler = debug_util.ReservoirSampler(3)
    k = 0
    while not end_flag.value:
        try:
            last_episode_data = None
            agent.sync_with_shared(shared_model)

            while True:
                agent_iterator = experiment.init_test_agent(agent=agent)
                try:
                    next(agent_iterator)
                    break
                except StopIteration as _:
                    warnings.warn("Agent iterator was empty.")

            while not agent.episode.is_complete():
                action_result = agent.act(train=False)
                if len(agent.eval_results) != 0:
                    agent.eval_results[-1] = recursively_detach(agent.eval_results[-1])

                if end_flag.value:
                    raise EndProcessException()

                last_episode_data = {
                    "iter": agent.episode.num_steps_taken_in_episode(),
                    "probs_per_agent": tuple(
                        probs.detach().cpu()[0, :].numpy()
                        for probs in action_result["probs_per_agent"]
                    ),
                    "actions": action_result["actions"],
                    "action_success": [
                        action_result["step_result"][i]["action_success"]
                        for i in range(len(action_result["step_result"]))
                    ],
                    "reward": [
                        action_result["step_result"][i]["reward"]
                        for i in range(len(action_result["step_result"]))
                    ],
                }

                if not agent.episode.is_complete():
                    res_sampler.add(last_episode_data)
            k += 1

            if args.visualize_test_agent:
                save_agents_path_without_frame_png(agent, k=k, args=args)

            try:
                next(agent_iterator)
                raise Exception("Agent iterator should only yield once.")
            except StopIteration as e:
                pass

            # Saving losses:
            if agent.record_all_in_test:
                last_losses = compute_losses_no_backprop(agent=agent)
            else:
                last_losses = None
            info = agent.episode.info()
            if last_losses is not None:
                info = {**info, **last_losses}
            if (
                len(agent.eval_results) > 0
                and "additional_metrics" in agent.eval_results[0]
            ):
                for a_metric in agent.eval_results[0]["additional_metrics"]:
                    info[a_metric] = (
                        1.0
                        / len(agent.eval_results)
                        * sum(
                            er["additional_metrics"][a_metric]
                            for er in agent.eval_results
                        )
                    )
            res_queue.put({k: info[k] for k in info})

            if args.verbose and last_episode_data is not None:
                res_sample = res_sampler.get_sample()
                print(
                    "\nRandom sample of testing episode in scene {}:".format(
                        agent.environment.scene_name
                    )
                )

                num_agents = agent.environment.num_agents
                using_central_agent = num_agents > len(
                    last_episode_data["probs_per_agent"]
                )
                for i in range(num_agents if not using_central_agent else 1):
                    verbose_string = "\nAGENT {}".format(i)
                    for x in sorted(
                        res_sample + [last_episode_data], key=lambda y: y["iter"]
                    ):
                        # TODO: Currently marginalizing to only show the first two agents.
                        #   As things become quickly messy. Is there a nice way to visualize
                        #   the probabilities for >2 agents?
                        probs_str = np.array2string(
                            x["probs_per_agent"][i]
                            if not using_central_agent
                            else np.reshape(
                                x["probs_per_agent"][i],
                                [
                                    round(
                                        len(x["probs_per_agent"][i]) ** (1 / num_agents)
                                    )
                                ]
                                * 2
                                + [-1],
                            ).sum(-1),
                            precision=2,
                            floatmode="fixed",
                            suppress_small=True,
                        )
                        if (
                            len(x["probs_per_agent"][i].squeeze().shape) == 1
                            and not using_central_agent
                        ):
                            probs_split = probs_str.split(" ")
                            if len(probs_split) > 10:
                                probs_split = (
                                    probs_split[0:5]
                                    + ["(..{}..)".format(len(probs_split) - 10)]
                                    + probs_split[-5:]
                                )
                            probs_str = " ".join(probs_split)
                        else:
                            probs_str = "\n" + probs_str + "\n"
                        verbose_string += "\n\tIter {0:>3}: probs {1}, action {2}, success: {3}, reward: {4}".format(
                            x["iter"],
                            probs_str,
                            x["actions"][i],
                            x["action_success"][i],
                            x["reward"][i],
                        )
                    print(verbose_string)
                print(info)
                res_sampler = debug_util.ReservoirSampler(3)

        except EndProcessException as _:
            print(
                "End process signal received in test worker. Quitting...", flush=True,
            )

            sys.exit()
        except (RuntimeError, ValueError, networkx.exception.NetworkXNoPath) as e:
            print(
                "RuntimeError, ValueError, or networkx exception when training, attempting to print traceback, print metadata, and reset."
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

        time.sleep(args.val_timeout)
