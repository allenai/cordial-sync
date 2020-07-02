import random
from typing import Tuple, Dict, Union, Optional, List, Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_base import Episode
from rl_base.agent import EnvType, RLAgent, A3CAgent
from utils import net_util
from utils.misc_util import (
    log_sum_exp,
    outer_product,
    joint_probability_tensor_from_mixture,
    joint_log_probability_tensor_from_mixture,
    outer_sum,
)


class MultiAgent(RLAgent):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_expert: Optional[nn.Module] = None,
        episode: Optional[Episode[EnvType]] = None,
        expert_class_available_actions: Optional[Sequence[str]] = None,
        gpu_id: int = -1,
        include_test_eval_results: bool = False,
        record_all_in_test: bool = False,
        include_depth_frame: bool = False,
        huber_delta: Optional[float] = None,
        discourage_failed_coordination: Union[bool, int, float] = False,
        use_smooth_ce_loss_for_expert: bool = False,
        resize_image_as: Optional[int] = None,
        dagger_mode: Optional[bool] = None,
        coordination_use_marginal_entropy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model=model, episode=episode, gpu_id=gpu_id, **kwargs)

        self.last_reward_per_agent: Optional[Tuple[float, ...]] = None
        self.include_test_eval_results = include_test_eval_results or record_all_in_test
        self.record_all_in_test = record_all_in_test
        self.include_depth_frame = include_depth_frame
        self.huber_delta = huber_delta
        self.discourage_failed_coordination = discourage_failed_coordination
        self.resize_image_as = resize_image_as

        self.log_prob_of_actions: List[Tuple[torch.FloatTensor]] = []
        self.entropy_per_agent: List[Optional[Tuple[torch.FloatTensor, ...]]] = []
        self.rewards_per_agent: List[Tuple[float, ...]] = []
        self.values_per_agent: List[Optional[Tuple[torch.FloatTensor, ...]]] = []
        self.actions: List[int] = []
        self.step_results: List[Dict[str, Any]] = []
        self.eval_results: List[Dict[str, Any]] = []

        self.log_prob_of_expert_action = []
        self.log_prob_of_unsafe_action = []
        self.expert_actions = []
        self.took_expert_action = []

        # For model based expert
        self._model_expert: Optional[nn.Module] = None
        if model_expert is not None:
            self.model_expert = model_expert
        self.hidden_expert: Optional[Any] = None
        self.expert_class_available_actions = expert_class_available_actions
        self.use_smooth_ce_loss_for_expert = use_smooth_ce_loss_for_expert

        # For rule based expert:
        assert not (dagger_mode and model_expert)
        self.dagger_mode = dagger_mode
        self.coordination_use_marginal_entropy = coordination_use_marginal_entropy

        self.take_expert_action_prob = (
            0
            if "take_expert_action_prob" not in kwargs
            else kwargs["take_expert_action_prob"]
        )

        self._unsafe_action_ind = kwargs.get("unsafe_action_ind")
        self._safety_loss_lambda = (
            1.0 if "safety_loss_lambda" not in kwargs else kwargs["safety_loss_lambda"]
        )
        self._use_a3c_loss = (
            False
            if "use_a3c_loss_when_not_expert_forcing" not in kwargs
            else kwargs["use_a3c_loss_when_not_expert_forcing"]
        )

        self.a3c_gamma = 0.99
        self.a3c_tau = 1.0
        self.a3c_beta = 1e-2

    @property
    def model_expert(self) -> nn.Module:
        assert self._model_expert is not None
        return self._model_expert

    @model_expert.setter
    def model_expert(self, model_expert_to_set: nn.Module):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self._model_expert = model_expert_to_set.cuda()
        else:
            self._model_expert = model_expert_to_set
        self._model_expert.eval()

    def eval_at_state(
        self,
        state: Dict[str, Union[Tuple, torch.FloatTensor, torch.cuda.FloatTensor]],
        hidden: Optional[Tuple[Tuple[torch.FloatTensor, ...]]],
        **kwargs,
    ) -> Dict[str, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]:
        assert type(state) == dict

        extra = {}
        if "object_ind" in state:
            extra["object_ind"] = self.preprocess_action(state["object_ind"])

        if "extra_embedding" in state:
            extra["extra_embedding"] = state["extra_embedding"]

        return self.model(
            inputs=torch.cat(tuple(s.unsqueeze(0) for s in state["frames"]), dim=0),
            hidden=hidden,
            agent_rotations=state["agent_rotations"],
            **extra,
        )

    def eval_at_state_expert(
        self,
        state: Dict[str, Union[Tuple, torch.FloatTensor, torch.cuda.FloatTensor]],
        hidden: Optional[Tuple[Tuple[torch.FloatTensor, ...]]],
        **kwargs,
    ) -> Dict[str, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]:
        assert type(state) == dict

        extra = {}
        if "object_ind" in state:
            extra["object_ind"] = self.preprocess_action(state["object_ind"])

        if "extra_embedding" in state:
            extra["extra_embedding"] = state["extra_embedding"]

        return net_util.recursively_detach(
            self.model_expert(
                inputs=torch.cat(tuple(s.unsqueeze(0) for s in state["frames"]), dim=0),
                hidden=hidden,
                agent_rotations=state["agent_rotations"],
                **extra,
            )
        )

    def _get_state(self, expert: bool):
        if not expert:
            states_for_agents = self.episode.states_for_agents()
        else:
            states_for_agents = self.episode.expert_states_for_agents()

        if self.include_depth_frame:
            assert (
                self.resize_image_as is not None
                and states_for_agents[0]["frame"].shape[-1] == 3
            )
            frames = tuple(
                torch.cat(
                    (
                        self.preprocess_frame(s["frame"], self.resize_image_as),
                        self.preprocess_depth_frame(
                            s["depth_frame"], self.resize_image_as
                        ),
                    ),
                    dim=0,
                )
                for s in states_for_agents
            )
        elif self.resize_image_as is not None:
            assert states_for_agents[0]["frame"].shape[-1] == 3
            frames = tuple(
                self.preprocess_frame(s["frame"], self.resize_image_as)
                for s in states_for_agents
            )
        else:
            assert self.resize_image_as is None
            frames = tuple(
                self.preprocess_long_tensor_frame(s["frame"]) for s in states_for_agents
            )

        extra = {}
        if "object_ind" in states_for_agents[0]:
            extra["object_ind"] = states_for_agents[0]["object_ind"]

        state = {
            "frames": frames,
            "agent_rotations": [
                90
                * int(self.environment.get_agent_location(agent_id=i)["rotation"] / 90)
                for i in range(self.environment.num_agents)
            ],
            **extra,
        }

        if "would_coordinated_action_succeed" in states_for_agents[0]:
            state["extra_embedding"] = self.gpuify(
                torch.FloatTensor(
                    states_for_agents[0]["would_coordinated_action_succeed"]
                )
                .view(1, -1)
                .repeat(self.environment.num_agents, 1)
            )

        return state

    @property
    def state(
        self,
    ) -> Dict[str, Union[Tuple, torch.FloatTensor, torch.cuda.FloatTensor]]:
        return self._get_state(expert=False)

    @property
    def state_expert(
        self,
    ) -> Dict[str, Union[Tuple, torch.FloatTensor, torch.cuda.FloatTensor]]:
        return self._get_state(expert=True)

    def next_expert_agent_action(self) -> Dict[str, Tuple[Optional[int], ...]]:
        # Currently not supporting model expert
        assert hasattr(self.episode, "next_expert_action")
        return {"expert_actions": self.episode.next_expert_action()}

    def act(
        self,
        train: bool = True,
        overriding_actions: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Dict:
        if self.dagger_mode and overriding_actions:
            # Dagger mode and overriding actions
            raise Exception(
                "Overriding action and dagger flags, both set to true, check!"
            )
        if not self.model.training and train:
            self.model.train()
        if self.model.training and not train:
            self.model.eval()

        assert self.episode is not None
        assert not self.episode.is_complete()

        if not train and self.hidden is not None:
            self.repackage_hidden()

        eval_result = self.eval_at_current_state()

        eval_result["coordination_type_tensor"] = self.gpuify(
            torch.LongTensor(
                np.array(self.episode.coordination_type_tensor(), dtype=int)
            )
        )
        m = eval_result["coordination_type_tensor"].squeeze()
        m_individual = (m == 0).float()
        m_coord = (m > 0).float()
        valid_action_tensor = m_individual + m_coord

        num_agents = self.environment.num_agents
        num_actions = len(self.episode.available_actions)

        coordinated_actions = False
        coordination_ind = None
        if "logit_all" in eval_result:
            # Either marginal or central agent
            logit_per_agent = tuple(
                eval_result["logit_all"][i].unsqueeze(0)
                for i in range(len(eval_result["logit_all"]))
            )
            probs_per_agent = tuple(
                F.softmax(logit, dim=1) for logit in logit_per_agent
            )
            log_probs_per_agent = tuple(
                F.log_softmax(logit, dim=1) for logit in logit_per_agent
            )
            joint_logit_all = eval_result.get("joint_logit_all")
            if joint_logit_all is not None and joint_logit_all:
                # Central agent
                assert len(probs_per_agent) == 1
                joint_prob_tensor = probs_per_agent[0].view(
                    *(num_agents * [num_actions])
                )
            else:
                joint_prob_tensor = outer_product(probs_per_agent)

            eval_result["log_probs_per_agent"] = log_probs_per_agent

        elif "coordinated_logits" in eval_result:
            # Mixture
            coordinated_actions = True

            randomization_logits = eval_result["randomization_logits"]
            coordinated_logits = eval_result["coordinated_logits"].view(
                num_agents, -1, num_actions
            )

            coordinated_probs = F.softmax(coordinated_logits, dim=2)
            randomization_probs = F.softmax(randomization_logits, dim=1)

            coordinated_log_probs = F.log_softmax(coordinated_logits, dim=2)
            randomization_log_probs = F.log_softmax(randomization_logits, dim=1)

            eval_result["coordinated_log_probs"] = coordinated_log_probs
            eval_result["randomization_log_probs"] = randomization_log_probs

            coordination_ind = randomization_probs.multinomial(num_samples=1).item()

            probs_per_agent = tuple(
                coordinated_probs[agent_id, :, :].unsqueeze(0)
                * randomization_probs.view(1, -1, 1)
                for agent_id in range(num_agents)
            )
            log_probs_per_agent = tuple(
                coordinated_log_probs[agent_id, :, :].unsqueeze(0)
                + randomization_log_probs.view(1, -1, 1)
                for agent_id in range(num_agents)
            )

            joint_prob_tensor = joint_probability_tensor_from_mixture(
                mixture_weights=randomization_probs.view(-1),
                marginal_prob_matrices=coordinated_probs,
            )
        else:
            raise NotImplementedError()

        low_rank_approx = outer_product(
            tuple(
                joint_prob_tensor.sum(dim=[j for j in range(num_agents) if j != i])
                for i in range(num_agents)
            )
        )
        eval_result["additional_metrics"] = {}
        eval_result["additional_metrics"]["dist_to_low_rank"] = torch.dist(
            joint_prob_tensor, low_rank_approx, p=1
        ).item()
        eval_result["additional_metrics"]["invalid_prob_mass"] = (
            torch.mul(joint_prob_tensor, 1 - valid_action_tensor).sum().item()
        )

        self.hidden = eval_result.get("hidden_all")

        joint_logit_all = eval_result.get("joint_logit_all")

        # If expert action needs to be taken or not
        if self.dagger_mode:
            take_expert_action = (
                train
                and self.take_expert_action_prob != 0
                and random.random() < self.take_expert_action_prob
            )
        else:
            take_expert_action = False

        if not take_expert_action:
            # Operations when not acting as per expert actions
            if self._use_a3c_loss:
                if self.coordination_use_marginal_entropy and coordinated_actions:
                    assert len(log_probs_per_agent[0].shape) == 3
                    self.entropy_per_agent.append(
                        tuple(
                            -(log_sum_exp(log_probs, 1) * probs.sum(1))
                            .sum()
                            .unsqueeze(0)
                            for log_probs, probs in zip(
                                log_probs_per_agent, probs_per_agent
                            )
                        )
                    )
                else:
                    self.entropy_per_agent.append(
                        tuple(
                            -(log_probs * probs).sum().unsqueeze(0)
                            for log_probs, probs in zip(
                                log_probs_per_agent, probs_per_agent
                            )
                        )
                    )

        # Compute and update the expert actions
        # Ensures length of expert actions is always less than (i.e. with None's) or equal to agent action-steps
        expert_actions_dict = self.next_expert_agent_action()
        self.expert_actions.append(expert_actions_dict["expert_actions"])

        # deciding which action to take:
        # The flow currently expects "actions":
        #   joint: is tuple of len 1 (multi index)
        #   mixture: is tuple of len nagents
        #   marginal: is tuple of len nagents
        if take_expert_action:
            assert not overriding_actions
            # take actions as per the expert.
            if joint_logit_all:
                # tuple of length 1
                actions = tuple(
                    [
                        int(
                            np.ravel_multi_index(
                                self.expert_actions[-1], (num_actions,) * num_agents
                            )
                        )
                    ]
                )
            else:
                actions = tuple(self.expert_actions[-1])

        elif overriding_actions:
            # take actions as per the specified overriding actions.
            actions = tuple(overriding_actions)
        else:
            # take actions as per the agent policy
            if coordinated_actions:
                actions = tuple(
                    (
                        probs[:, coordination_ind, :]
                        / probs[:, coordination_ind, :].sum()
                    )
                    .multinomial(num_samples=1)
                    .item()
                    for probs in probs_per_agent
                )
            else:
                actions = tuple(
                    probs.multinomial(num_samples=1).item() for probs in probs_per_agent
                )
        self.took_expert_action.append(take_expert_action)

        if coordinated_actions:
            log_prob_of_action_per_agent = tuple(
                log_probs[:, coordination_ind, :].view(num_actions)[action]
                for log_probs, action in zip(log_probs_per_agent, actions)
            )
        else:
            log_prob_of_action_per_agent = tuple(
                log_probs.view(-1)[action]
                for log_probs, action in zip(log_probs_per_agent, actions)
            )

        log_prob_of_expert_action_per_agent = None
        if self.expert_actions[-1] is not None and all(
            x is not None for x in self.expert_actions[-1]
        ):
            if self.use_smooth_ce_loss_for_expert:
                raise NotImplementedError(
                    "Expert actions coming from an expert model, currently not supported"
                )
            elif coordinated_actions:
                assert "permute_agent_probs" not in expert_actions_dict

                log_probs_tensor = joint_log_probability_tensor_from_mixture(
                    log_mixture_weights=randomization_log_probs.view(-1),
                    marginal_log_prob_matrices=coordinated_log_probs,
                )
                ce = log_probs_tensor
                for expert_action in self.expert_actions[-1]:
                    ce = ce[expert_action]
                assert ce.dim() == 0
                log_prob_of_expert_action_per_agent = (
                    ce / self.environment.num_agents,
                ) * self.environment.num_agents
            else:
                # Multiagent navigation task update
                if joint_logit_all:
                    # Central
                    assert len(self.expert_actions[-1]) == num_agents
                    assert len(log_probs_per_agent) == 1
                    assert log_probs_per_agent[0].shape[1] == num_actions ** num_agents
                    log_prob_of_expert_action_per_agent = tuple(
                        [
                            log_probs_per_agent[0].view(num_actions ** num_agents)[
                                int(
                                    np.ravel_multi_index(
                                        self.expert_actions[-1],
                                        (num_actions,) * num_agents,
                                    )
                                )
                            ]
                        ]
                    )
                else:
                    # Marginal
                    log_prob_of_expert_action_per_agent = tuple(
                        log_probs.view(num_actions)[action]
                        for log_probs, action in zip(
                            log_probs_per_agent, self.expert_actions[-1]
                        )
                    )

        if train and (self._unsafe_action_ind is not None):
            raise NotImplementedError()

        before_locations = tuple(
            self.environment.get_agent_location(agent_id=i)
            for i in range(self.environment.num_agents)
        )
        # Making joint action single agent work with the current multi (marginal) agent setup
        if joint_logit_all is not None and joint_logit_all:
            assert len(probs_per_agent) == 1
            assert probs_per_agent[0].shape[1] == num_actions ** num_agents
            actions = tuple(
                int(a)
                for a in np.unravel_index(actions[0], (num_actions,) * num_agents)
            )

        step_result = self.episode.multi_step(actions)
        after_locations = tuple(
            self.environment.get_agent_location(agent_id=i)
            for i in range(self.environment.num_agents)
        )

        rewards = tuple(s.get("reward") for s in step_result)
        if any(r is None for r in rewards):
            self.last_reward_per_agent = None
        elif joint_logit_all is not None:
            self.last_reward_per_agent = tuple(
                sum(rewards) for _ in range(self.environment.num_agents)
            )
        else:
            self.last_reward_per_agent = tuple(float(r) for r in rewards)
        self.values_per_agent.append(eval_result.get("value_all"))

        for i in range(len(step_result)):
            step_result[i]["before_location"] = before_locations[i]
            step_result[i]["after_location"] = after_locations[i]
        eval_result["probs_per_agent"] = probs_per_agent
        eval_result["log_probs_per_agent"] = log_probs_per_agent
        eval_result["step_result"] = step_result
        eval_result["actions"] = actions
        eval_result["training"] = train
        self.step_results.append(step_result)
        self.actions.append(eval_result["actions"])
        self.rewards_per_agent.append(self.last_reward_per_agent)
        if train or self.include_test_eval_results or self.record_all_in_test:
            self.eval_results.append(eval_result)
        if train or self.record_all_in_test:
            self.log_prob_of_actions.append(log_prob_of_action_per_agent)
            self.log_prob_of_expert_action.append(log_prob_of_expert_action_per_agent)
        return eval_result

    def clear_history(self) -> None:
        self.clear_graph_data()
        self.last_reward_per_agent = None
        self.rewards_per_agent = []
        self.actions = []
        self.expert_actions = []
        self.step_results = []
        self.took_expert_action = []

    def clear_graph_data(self):
        self.log_prob_of_actions = []
        self.entropy_per_agent = []
        self.values_per_agent = []
        self.log_prob_of_expert_action = []
        self.log_prob_of_unsafe_action = []
        self.eval_results = []

    def safety_loss(self, loss_dict):
        assert self._unsafe_action_ind is not None

        divisor = 0
        numerator = 0
        for expert_actions, actions, log_probs in zip(
            self.expert_actions, self.actions, self.log_prob_of_unsafe_action
        ):
            unsafe_but_expert_safe = [
                expert_action != self._unsafe_action_ind
                and action == self._unsafe_action_ind
                for expert_action, action in zip(expert_actions, actions)
            ]
            numerator += sum(
                log_prob
                for include, log_prob in zip(unsafe_but_expert_safe, log_probs)
                if include
            )
            divisor += sum(unsafe_but_expert_safe)

        if divisor != 0:
            loss_dict["safety_loss"] = self._safety_loss_lambda * (numerator / divisor)

    def failed_coordination_loss(self, loss_dict):
        loss = 0.0
        nactions = len(self.episode.available_actions)
        for er in self.eval_results:
            m = er["coordination_type_tensor"].squeeze()
            m_individual = (m == 0).float()
            m_coord = (m > 0).float()

            if "coordinated_log_probs" in er:
                joint_log_probs = joint_log_probability_tensor_from_mixture(
                    log_mixture_weights=er["randomization_log_probs"].view(-1),
                    marginal_log_prob_matrices=er["coordinated_log_probs"],
                )
            elif "joint_logit_all" in er and er["joint_logit_all"]:
                assert len(er["logit_all"]) == 1
                joint_log_probs = er["log_probs_per_agent"][0].view(
                    *[self.environment.num_agents * [nactions]]
                )
            else:
                assert len(er["logit_all"]) > 1
                joint_log_probs = outer_sum(
                    [log_probs.view(-1) for log_probs in er["log_probs_per_agent"]]
                )

            loss -= 0.5 * (
                (joint_log_probs * m_individual).sum() / (m_individual.sum())
                + (joint_log_probs * m_coord).sum() / (m_coord.sum())
            )

        loss_dict["failed_coordination"] = loss / len(self.eval_results)

    def loss(
        self, train: bool = True, size_average: bool = True, **kwargs
    ) -> Union[Dict[str, torch.Tensor], None]:
        loss_dict = {}

        filtered_expert_log_prob_sums = [
            sum(x) for x in self.log_prob_of_expert_action if x is not None
        ]

        if len(filtered_expert_log_prob_sums) != 0:
            loss_dict["xe_loss"] = -sum(filtered_expert_log_prob_sums) / len(
                filtered_expert_log_prob_sums
            )

        if self._unsafe_action_ind is not None:
            self.safety_loss(loss_dict)

        if (
            self._use_a3c_loss
            and (not any(self.took_expert_action))
            and all(r is not None for r in self.rewards_per_agent)
            and all(e is not None for e in self.entropy_per_agent)
            and all(v is not None for v in self.values_per_agent)
        ):
            policy_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0
            eval = self.eval_at_current_state()
            for agent_id in range(len(eval["value_all"])):
                if self.episode.is_complete():
                    future_reward_est = 0.0
                else:
                    future_reward_est = eval["value_all"][agent_id].item()

                def extract(inputs: List[Tuple]):
                    return [input[agent_id] for input in inputs]

                a3c_losses = A3CAgent.a3c_loss(
                    values=extract(self.values_per_agent),
                    rewards=extract(self.rewards_per_agent),
                    log_prob_of_actions=extract(self.log_prob_of_actions),
                    entropies=extract(self.entropy_per_agent),
                    future_reward_est=future_reward_est,
                    gamma=self.a3c_gamma,
                    tau=self.a3c_tau,
                    beta=self.a3c_beta,
                    gpu_id=self.gpu_id,
                    huber_delta=self.huber_delta,
                )
                policy_loss += a3c_losses["policy"]
                value_loss += a3c_losses["value"]
                entropy_loss += a3c_losses["entropy"]

            policy_loss /= len(self.rewards_per_agent)
            value_loss /= len(self.rewards_per_agent)
            entropy_loss /= len(self.rewards_per_agent)
            loss_dict["a3c_policy_loss"] = policy_loss
            loss_dict["a3c_value_loss"] = value_loss
            loss_dict["a3c_entropy_loss"] = entropy_loss

        if self.discourage_failed_coordination:
            if "a3c_entropy_loss" in loss_dict:
                loss_dict["a3c_entropy_loss"] = net_util.recursively_detach(
                    loss_dict["a3c_entropy_loss"]
                )

            if type(self.discourage_failed_coordination) in [int, float]:
                self.failed_coordination_loss(loss_dict)
                alpha = max(
                    self.a3c_beta,
                    (self.discourage_failed_coordination - self.global_episode_count)
                    / self.discourage_failed_coordination,
                )
                loss_dict["failed_coordination"] *= alpha
            else:
                assert type(self.discourage_failed_coordination) is bool
                self.failed_coordination_loss(loss_dict)

        return loss_dict
