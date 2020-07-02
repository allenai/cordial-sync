from __future__ import division

from collections import OrderedDict
from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc_util import norm_col_init, weights_init, outer_product, outer_sum


def _unfold_communications(speech_per_agent: torch.FloatTensor):
    assert len(speech_per_agent.shape) >= 2
    num_agents = speech_per_agent.shape[0]

    unfolded_commns = [speech_per_agent[1:, :].view(1, -1)]
    for i in range(1, num_agents - 1):
        unfolded_commns.append(
            torch.cat(
                (
                    speech_per_agent[(i + 1) :,].view(1, -1),
                    speech_per_agent[:i,].view(1, -1),
                ),
                dim=1,
            )
        )
    unfolded_commns.append(speech_per_agent[:-1,].view(1, -1))

    return torch.cat(unfolded_commns, dim=0)


class A3CLSTMCentralEgoGridsEmbedCNN(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        occupancy_embed_length: int,
    ):
        super(A3CLSTMCentralEgoGridsEmbedCNN, self).__init__()

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents

        self.occupancy_embed_length = occupancy_embed_length
        self.occupancy_embeddings = nn.Embedding(
            num_inputs_per_agent, self.occupancy_embed_length
        )

        # input to conv is (num_agents * self.occupancy_embed_length, 15, 15)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.occupancy_embed_length * self.num_agents,
                            32,
                            3,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape = (15, 15)
                    ("conv2", nn.Conv2d(32, 64, 3, stride=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape = (7, 7)
                    ("conv3", nn.Conv2d(64, 128, 3, stride=2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape = (3, 3)
                    ("conv4", nn.Conv2d(128, 256, 3, stride=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (1, 1); Stride doesn't matter above
                ]
            )
        )

        # LSTM
        self.lstm_in_dim = 256
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        # Marginal Linear actor
        self.marginal_actor_linear_list = nn.ModuleList(
            [
                nn.Linear(state_repr_length, self.num_outputs)
                for _ in range(self.num_agents)
            ]
        )

        # Conditional actor
        self.joint_actor_linear = nn.Linear(
            state_repr_length, self.num_outputs ** self.num_agents
        )

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        for agent_id in range(self.num_agents):
            self.marginal_actor_linear_list[agent_id].weight.data = norm_col_init(
                self.marginal_actor_linear_list[agent_id].weight.data, 0.01
            )
            self.marginal_actor_linear_list[agent_id].bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 15, 15):
            raise Exception("input to model is not as expected, check!")

        inputs = inputs.float()

        # inputs.shape == (self.num_agents, self.num_inputs, 15, 15)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        original_shape = inputs.shape
        # original_shape == (self.num_agents, 15, 15, self.num_inputs)

        scaling_factor = torch.reciprocal(
            (inputs == 1).sum(dim=3).unsqueeze(3).float() + 1e-3
        )
        # scaling_factor.shape == (self.num_agents, 15, 15, 1)

        inputs = inputs.view(-1, self.num_inputs_per_agent)
        inputs = inputs.matmul(self.occupancy_embeddings.weight)
        inputs = inputs.view(
            original_shape[0],
            original_shape[1],
            original_shape[2],
            self.occupancy_embed_length,
        )
        x = torch.mul(inputs, scaling_factor.expand_as(inputs))
        # x.shape == (self.num_agents, 15, 15, self.occupancy_embed_length)
        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(
                1,
                original_shape[0] * self.occupancy_embed_length,
                original_shape[1],
                original_shape[2],
            )
        )
        # x.shape == (1, self.num_agents * self.occupancy_embed_length, 15, 15)
        x = self.cnn(x)
        # x.shape == (1, 256, 1, 1)
        x = x.view(1, -1)
        # x.shape == (1, self.lstm_in_dim)
        batch_dim = 1
        # x.shape = [batch_dim, self.lstm_in_dim]
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        marginal_actor_linear_output_list = [
            self.marginal_actor_linear_list[agent_id](final_state).view(-1)
            for agent_id in range(self.num_agents)
        ]
        # list with each element of size [batch_dim, self.num_outputs]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])
        joint_actor_logits = self.joint_actor_linear(final_state).view(
            batch_dim, *((self.num_outputs,) * self.num_agents)
        )
        if self.num_agents <= 2:
            marginal_logits = outer_product(
                marginal_actor_linear_output_list
            ).unsqueeze(0)
        else:
            marginal_logits = outer_sum(marginal_actor_linear_output_list).unsqueeze(0)

        assert marginal_logits.shape == joint_actor_logits.shape

        combined_logits = joint_actor_logits + marginal_logits
        # One length lists as this is a central agent
        return {
            "logit_all": combined_logits.view(
                batch_dim, self.num_outputs ** self.num_agents
            ),
            "value_all": self.critic_linear(final_state),  # shape: [batch_dim,1]
            "hidden_all": hidden,
            "joint_logit_all": True,
        }


class A3CLSTMBigCentralEgoGridsEmbedCNN(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        occupancy_embed_length: int,
    ):
        super(A3CLSTMBigCentralEgoGridsEmbedCNN, self).__init__()

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents

        self.occupancy_embed_length = occupancy_embed_length
        self.occupancy_embeddings = nn.Embedding(
            num_inputs_per_agent, self.occupancy_embed_length
        )

        # input to conv is (self.occupancy_embed_length, 15, 15)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.occupancy_embed_length, 32, 3, padding=1)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape = (15, 15)
                    ("conv2", nn.Conv2d(32, 64, 3, stride=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape = (7, 7)
                    ("conv3", nn.Conv2d(64, 128, 3, stride=2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape = (3, 3)
                    ("conv4", nn.Conv2d(128, 256, 3, stride=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (1, 1); Stride doesn't matter above
                ]
            )
        )

        self.lstm_in_dim = 256
        self.before_lstm_mlp = nn.Sequential(
            nn.Linear(
                self.num_agents * self.lstm_in_dim, self.num_agents * self.lstm_in_dim
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_agents * self.lstm_in_dim, self.lstm_in_dim),
            nn.ReLU(inplace=True),
        )

        # LSTM
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        # Marginal Linear actor
        self.marginal_actor_linear_list = nn.ModuleList(
            [
                nn.Linear(state_repr_length, self.num_outputs)
                for _ in range(self.num_agents)
            ]
        )

        # Conditional actor
        self.joint_actor_linear = nn.Linear(
            state_repr_length, self.num_outputs ** self.num_agents
        )

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        for agent_id in range(self.num_agents):
            self.marginal_actor_linear_list[agent_id].weight.data = norm_col_init(
                self.marginal_actor_linear_list[agent_id].weight.data, 0.01
            )
            self.marginal_actor_linear_list[agent_id].bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 15, 15):
            raise Exception("input to model is not as expected, check!")

        inputs = inputs.float()

        # inputs.shape == (self.num_agents, self.num_inputs, 15, 15)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        original_shape = inputs.shape
        # original_shape == (self.num_agents, 15, 15, self.num_inputs)

        scaling_factor = torch.reciprocal(
            (inputs == 1).sum(dim=3).unsqueeze(3).float() + 1e-3
        )
        # scaling_factor.shape == (self.num_agents, 15, 15, 1)

        inputs = inputs.view(-1, self.num_inputs_per_agent)
        inputs = inputs.matmul(self.occupancy_embeddings.weight)
        inputs = inputs.view(
            original_shape[0],
            original_shape[1],
            original_shape[2],
            self.occupancy_embed_length,
        )
        x = torch.mul(inputs, scaling_factor.expand_as(inputs))
        # x.shape == (self.num_agents, 15, 15, self.occupancy_embed_length)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x.shape == (self.num_agents, self.occupancy_embed_length, 15, 15)
        x = self.cnn(x)
        # x.shape == (self.num_agents, 256, 1, 1)
        x = x.view(1, -1)
        # x.shape == (1, self.num_agents * self.lstm_in_dim)
        x = self.before_lstm_mlp(x)
        batch_dim = 1
        # x.shape = [batch_dim, self.lstm_in_dim]
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        marginal_actor_linear_output_list = [
            self.marginal_actor_linear_list[agent_id](final_state).view(-1)
            for agent_id in range(self.num_agents)
        ]
        # list with each element of size [batch_dim, self.num_outputs]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])
        joint_actor_logits = self.joint_actor_linear(final_state).view(
            batch_dim, *((self.num_outputs,) * self.num_agents)
        )
        if self.num_agents <= 2:
            marginal_logits = outer_product(
                marginal_actor_linear_output_list
            ).unsqueeze(0)
        else:
            marginal_logits = outer_sum(marginal_actor_linear_output_list).unsqueeze(0)

        assert marginal_logits.shape == joint_actor_logits.shape

        combined_logits = joint_actor_logits + marginal_logits
        # One length lists as this is a central agent
        return {
            "logit_all": combined_logits.view(
                batch_dim, self.num_outputs ** self.num_agents
            ),
            "value_all": self.critic_linear(final_state),  # shape: [batch_dim,1]
            "hidden_all": hidden,
            "joint_logit_all": True,
        }


class A3CLSTMBigCentralBackboneAndCoordinatedActionsEgoGridsEmbedCNN(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        occupancy_embed_length: int,
        coordinate_actions: bool,
        coordinate_actions_dim: Optional[int] = None,
        central_critic: bool = False,
    ):
        super(
            A3CLSTMBigCentralBackboneAndCoordinatedActionsEgoGridsEmbedCNN, self
        ).__init__()

        # Either coordinate actions (mixture of marginals)
        # Otherwise i.e. single marginals then there shouldn't be any
        # coordinate_actions_dim
        assert coordinate_actions or not coordinate_actions_dim
        # Since the representation for both agents is the same, having
        # separate actor weights for mixture and also for single is
        # necessary. Explicitly captured below.
        separate_actor_weights = True
        # Also since this is a single central agent, there is no central critic
        # feature expected in this model
        assert not central_critic

        self.separate_actor_weights = separate_actor_weights
        self.coordinate_actions_dim = (
            self.num_outputs
            if coordinate_actions and coordinate_actions_dim is None
            else coordinate_actions_dim
        )

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents

        self.occupancy_embed_length = occupancy_embed_length
        self.occupancy_embeddings = nn.Embedding(
            num_inputs_per_agent, self.occupancy_embed_length
        )
        self.coordinate_actions = coordinate_actions
        # input to conv is (self.occupancy_embed_length, 15, 15)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.occupancy_embed_length, 32, 3, padding=1)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape = (15, 15)
                    ("conv2", nn.Conv2d(32, 64, 3, stride=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape = (7, 7)
                    ("conv3", nn.Conv2d(64, 128, 3, stride=2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape = (3, 3)
                    ("conv4", nn.Conv2d(128, 256, 3, stride=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (1, 1); Stride doesn't matter above
                ]
            )
        )

        self.lstm_in_dim = 256
        self.before_lstm_mlp = nn.Sequential(
            nn.Linear(2 * self.lstm_in_dim, 2 * self.lstm_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.lstm_in_dim, self.lstm_in_dim),
            nn.ReLU(inplace=True),
        )

        # LSTM
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        if coordinate_actions:
            # Randomization MLP operating on the central state representation
            self.to_randomization_logits = nn.Sequential(
                nn.Linear(1 * state_repr_length, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.coordinate_actions_dim),
            )

        # Linear actor
        self.actor_linear = None
        actor_linear_input_dim = state_repr_length
        # self.pre_actor_list = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(state_repr_length, actor_linear_input_dim),
        #             nn.ReLU(inplace=True),
        #         )
        #
        #         for _ in range(2)
        #     ]
        # )
        if coordinate_actions:
            self.actor_linear_list = nn.ModuleList(
                [
                    nn.Linear(
                        actor_linear_input_dim,
                        self.num_outputs * self.coordinate_actions_dim,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.actor_linear_list = nn.ModuleList(
                [nn.Linear(actor_linear_input_dim, self.num_outputs) for _ in range(2)]
            )

        for al in self.actor_linear_list:
            al.weight.data = norm_col_init(al.weight.data, 0.01)
            al.bias.data.fill_(0)

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 15, 15):
            raise Exception("input to model is not as expected, check!")

        inputs = inputs.float()

        # inputs.shape == (2, self.num_inputs, 15, 15)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        original_shape = inputs.shape
        # original_shape == (2, 15, 15, self.num_inputs)

        scaling_factor = torch.reciprocal(
            (inputs == 1).sum(dim=3).unsqueeze(3).float() + 1e-3
        )
        # scaling_factor.shape == (2, 15, 15, 1)

        inputs = inputs.view(-1, self.num_inputs_per_agent)
        inputs = inputs.matmul(self.occupancy_embeddings.weight)
        inputs = inputs.view(
            original_shape[0],
            original_shape[1],
            original_shape[2],
            self.occupancy_embed_length,
        )
        x = torch.mul(inputs, scaling_factor.expand_as(inputs))
        # x.shape == (2, 15, 15, self.occupancy_embed_length)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x.shape == (2, self.occupancy_embed_length, 15, 15)
        x = self.cnn(x)
        # x.shape == (2, 256, 1, 1)
        x = x.view(1, -1)
        # x.shape == (1, 2 * self.lstm_in_dim = 512)
        x = self.before_lstm_mlp(x)
        batch_dim = 1
        # x.shape = [batch_dim, self.lstm_in_dim]
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])

        # One central backbone agent, so no option for central critic, check asserts
        value_all = self.critic_linear(final_state)

        # Same value for both agents
        to_return = {
            "value_all": torch.cat([value_all, value_all], dim=0),
            "hidden_all": hidden,
        }

        logits = torch.cat(
            [
                linear(
                    # self.pre_actor_list[i](final_state)
                    final_state
                )
                for i, linear in enumerate(self.actor_linear_list)
            ],
            dim=0,
        )

        if self.coordinate_actions:
            # Mixture marginals, actor output = num_outputs * coordinate_action_dim
            to_return["randomization_logits"] = self.to_randomization_logits(
                final_state
            )
            to_return["coordinated_logits"] = logits
        else:
            # Single marginal, actor output = num_outputs
            to_return["logit_all"] = logits

        return to_return


class A3CLSTMCentralEgoVision(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
    ):
        super(A3CLSTMCentralEgoVision, self).__init__()

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents

        # input to conv is (num_agents * self.num_inputs_per_agent, 84, 84)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent * self.num_agents,
                            16,
                            5,
                            stride=1,
                            padding=2,
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(16, 16, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(16, 32, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(2, 2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv4", nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                    ("maxpool4", nn.MaxPool2d(2, 2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                    ("conv5", nn.Conv2d(64, 128, 3, stride=1, padding=1)),
                    ("maxpool5", nn.MaxPool2d(2, 2)),
                    ("relu5", nn.ReLU(inplace=True)),
                    # shape = (2, 2)
                ]
            )
        )

        # LSTM
        self.lstm_in_dim = 512
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        # Marginal Linear actor
        self.marginal_actor_linear_list = nn.ModuleList(
            [
                nn.Linear(state_repr_length, self.num_outputs)
                for _ in range(self.num_agents)
            ]
        )

        # Conditional actor
        self.joint_actor_linear = nn.Linear(
            state_repr_length, self.num_outputs * self.num_outputs
        )

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        for agent_id in range(self.num_agents):
            self.marginal_actor_linear_list[agent_id].weight.data = norm_col_init(
                self.marginal_actor_linear_list[agent_id].weight.data, 0.01
            )
            self.marginal_actor_linear_list[agent_id].bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):

        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        inputs = inputs.view(
            self.num_agents * self.num_inputs_per_agent, 84, 84
        ).unsqueeze(0)
        # inputs.shape == (1, self.num_agents * self.num_inputs, 84, 84)

        x = self.cnn(inputs)
        # x.shape == (1, 128, 4, 4)

        x = x.view(1, -1)
        # x.shape == (batch_dim, self.lstm_in_dim)

        batch_dim = 1
        # x.shape = [batch_dim, self.lstm_in_dim]
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        marginal_actor_linear_output_list = [
            self.marginal_actor_linear_list[agent_id](final_state)
            for agent_id in range(self.num_agents)
        ]
        # list with each element of size [batch_dim, self.num_outputs]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])
        joint_actor_logits = self.joint_actor_linear(final_state).view(
            batch_dim, self.num_outputs, self.num_outputs
        )
        outer_product_marginal_logits = marginal_actor_linear_output_list[0].unsqueeze(
            2
        ) * marginal_actor_linear_output_list[1].unsqueeze(1)
        assert outer_product_marginal_logits.shape == (
            batch_dim,
            self.num_outputs,
            self.num_outputs,
        )
        combined_logits = joint_actor_logits + outer_product_marginal_logits
        # One length lists as this is a central agent
        return {
            "logit_all": combined_logits.view(
                batch_dim, self.num_outputs * self.num_outputs
            ),
            "value_all": self.critic_linear(final_state),  # shape: [batch_dim,1]
            "hidden_all": hidden,
            "joint_logit_all": True,
        }


class A3CLSTMBigCentralEgoVision(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
    ):
        super(A3CLSTMBigCentralEgoVision, self).__init__()

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents

        # input to conv is (self.num_inputs_per_agent, 84, 84)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent, 16, 5, stride=1, padding=2
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(16, 16, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(16, 32, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(2, 2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv4", nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                    ("maxpool4", nn.MaxPool2d(2, 2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                    ("conv5", nn.Conv2d(64, 128, 3, stride=1, padding=1)),
                    ("maxpool5", nn.MaxPool2d(2, 2)),
                    ("relu5", nn.ReLU(inplace=True)),
                    # shape = (2, 2)
                ]
            )
        )

        # LSTM
        self.lstm_in_dim = 512

        self.before_lstm_mlp = nn.Sequential(
            nn.Linear(self.lstm_in_dim * self.num_agents, self.lstm_in_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        # Marginal Linear actor
        self.marginal_actor_linear_list = nn.ModuleList(
            [
                nn.Linear(state_repr_length, self.num_outputs)
                for _ in range(self.num_agents)
            ]
        )

        # Conditional actor
        self.joint_actor_linear = nn.Linear(
            state_repr_length, self.num_outputs ** self.num_agents
        )

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        for agent_id in range(self.num_agents):
            self.marginal_actor_linear_list[agent_id].weight.data = norm_col_init(
                self.marginal_actor_linear_list[agent_id].weight.data, 0.01
            )
            self.marginal_actor_linear_list[agent_id].bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):

        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        # inputs.shape == (1, self.num_agents * self.num_inputs, 84, 84)

        x = self.cnn(inputs)
        # x.shape == (self.num_agents, 128, 4, 4)

        x = x.view(1, -1)
        # x.shape == (batch_dim, self.lstm_in_dim * self.num_agents)

        x = self.before_lstm_mlp(x)
        # x.shape == (batch_dim, self.lstm_in_dim)

        batch_dim = 1
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        marginal_actor_linear_output_list = [
            self.marginal_actor_linear_list[agent_id](final_state).view(-1)
            for agent_id in range(self.num_agents)
        ]
        # list with each element of size [batch_dim, self.num_outputs]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])
        joint_actor_logits = self.joint_actor_linear(final_state).view(
            batch_dim, *((self.num_outputs,) * self.num_agents)
        )
        if self.num_agents <= 2:
            marginal_logits = outer_product(
                marginal_actor_linear_output_list
            ).unsqueeze(0)
        else:
            marginal_logits = outer_sum(marginal_actor_linear_output_list).unsqueeze(0)
        assert marginal_logits.shape == joint_actor_logits.shape
        assert marginal_logits.shape == (
            batch_dim,
            *((self.num_outputs,) * self.num_agents),
        )
        combined_logits = joint_actor_logits + marginal_logits
        # One length lists as this is a central agent
        return {
            "logit_all": combined_logits.view(
                batch_dim, self.num_outputs ** self.num_agents
            ),
            "value_all": self.critic_linear(final_state),  # shape: [batch_dim,1]
            "hidden_all": hidden,
            "joint_logit_all": True,
        }


class OldTaskA3CLSTMBigCentralEgoVision(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
    ):
        super(OldTaskA3CLSTMBigCentralEgoVision, self).__init__()

        self.num_outputs = sum(len(x) for x in action_groups)

        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_inputs = num_inputs_per_agent * num_agents
        self.num_agents = num_agents
        final_cnn_channels = 19

        # input to conv is (self.num_inputs_per_agent, 84, 84)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent, 32, 5, stride=1, padding=2
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(32, 32, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(32, 64, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(2, 2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    (
                        "conv4",
                        nn.Conv2d(64, final_cnn_channels, 3, stride=1, padding=1),
                    ),
                    ("maxpool4", nn.MaxPool2d(2, 2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                ]
            )
        )
        # CNN output:
        self.cnn_output_dim = final_cnn_channels * 4 * 4
        # LSTM
        self.lstm_in_dim = 64 * 4 * 4

        self.before_lstm_mlp = nn.Sequential(
            nn.Linear(self.cnn_output_dim * 2, self.lstm_in_dim), nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(self.lstm_in_dim, state_repr_length, batch_first=True)

        # Post LSTM fully connected layers
        self.after_lstm_mlp = nn.Sequential(
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
            nn.Linear(state_repr_length, state_repr_length),
            nn.ReLU(inplace=True),
        )

        # Marginal Linear actor
        self.marginal_actor_linear_list = nn.ModuleList(
            [
                nn.Linear(state_repr_length, self.num_outputs)
                for _ in range(self.num_agents)
            ]
        )

        # Conditional actor
        self.joint_actor_linear = nn.Linear(
            state_repr_length, self.num_outputs * self.num_outputs
        )

        # Linear critic
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        for agent_id in range(self.num_agents):
            self.marginal_actor_linear_list[agent_id].weight.data = norm_col_init(
                self.marginal_actor_linear_list[agent_id].weight.data, 0.01
            )
            self.marginal_actor_linear_list[agent_id].bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):

        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        # inputs.shape == (1, self.num_agents * self.num_inputs, 84, 84)

        x = self.cnn(inputs)
        # x.shape == (self.num_agents, 128, 4, 4)

        x = x.view(1, -1)
        # x.shape == (batch_dim, self.lstm_in_dim * 2)

        x = self.before_lstm_mlp(x)
        # x.shape == (batch_dim, self.lstm_in_dim)

        batch_dim = 1
        if x.shape != (1, self.lstm_in_dim):
            print("x.shape: {}".format(x.shape))
            raise Exception("output of model is not as expected, check!")

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [batch_dim, 1, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [batch_dim, state_repr_length]

        final_state = x + self.after_lstm_mlp(x)
        # state_talk_reply_repr.shape = [batch=1, state_repr_length]

        marginal_actor_linear_output_list = [
            self.marginal_actor_linear_list[agent_id](final_state)
            for agent_id in range(self.num_agents)
        ]
        # list with each element of size [batch_dim, self.num_outputs]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (final_state.unsqueeze(0), hidden[1])
        joint_actor_logits = self.joint_actor_linear(final_state).view(
            batch_dim, self.num_outputs, self.num_outputs
        )
        outer_product_marginal_logits = marginal_actor_linear_output_list[0].unsqueeze(
            2
        ) * marginal_actor_linear_output_list[1].unsqueeze(1)
        assert outer_product_marginal_logits.shape == (
            batch_dim,
            self.num_outputs,
            self.num_outputs,
        )
        combined_logits = joint_actor_logits + outer_product_marginal_logits
        # One length lists as this is a central agent
        return {
            "logit_all": combined_logits.view(
                batch_dim, self.num_outputs * self.num_outputs
            ),
            "value_all": self.critic_linear(final_state),  # shape: [batch_dim,1]
            "hidden_all": hidden,
            "joint_logit_all": True,
        }


class A3CLSTMNStepComCoordinatedActionsEgoVision(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        talk_embed_length: int,
        agent_num_embed_length: int,
        reply_embed_length: int,
        num_talk_symbols: int,
        num_reply_symbols: int,
        turn_off_communication: bool,
        coordinate_actions: bool,
        coordinate_actions_dim: Optional[int] = None,
        central_critic: bool = False,
        separate_actor_weights: bool = False,
        final_cnn_channels: int = 128,
    ):
        super(A3CLSTMNStepComCoordinatedActionsEgoVision, self).__init__()
        self.num_outputs = sum(len(x) for x in action_groups)

        self.turn_off_communication = turn_off_communication
        self.central_critic = central_critic
        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_agents = num_agents
        self.num_talk_symbols = num_talk_symbols
        self.num_reply_symbols = num_reply_symbols
        self.separate_actor_weights = separate_actor_weights
        self.coordinate_actions_dim = (
            self.num_outputs
            if coordinate_actions_dim is None
            else coordinate_actions_dim
        )

        self.coordinate_actions = coordinate_actions

        # input to conv is (num_agents, self.num_inputs_per_agent, 84, 84)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent, 16, 5, stride=1, padding=2
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(16, 16, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(16, 32, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(2, 2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv4", nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                    ("maxpool4", nn.MaxPool2d(2, 2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                    (
                        "conv5",
                        nn.Conv2d(64, final_cnn_channels, 3, stride=1, padding=1),
                    ),
                    ("maxpool5", nn.MaxPool2d(2, 2)),
                    ("relu5", nn.ReLU(inplace=True)),
                    # shape = (2, 2)
                ]
            )
        )

        # Vocab embed
        self.talk_embeddings = nn.Embedding(num_talk_symbols, talk_embed_length)
        self.reply_embeddings = nn.Embedding(num_reply_symbols, reply_embed_length)

        self.talk_symbol_classifier = nn.Linear(state_repr_length, num_talk_symbols)
        self.reply_symbol_classifier = nn.Linear(state_repr_length, num_reply_symbols)

        # Agent embed
        self.agent_num_embeddings = nn.Parameter(
            torch.rand(self.num_agents, agent_num_embed_length)
        )

        # LSTM
        self.lstm = nn.LSTM(
            final_cnn_channels * 4 + agent_num_embed_length,
            state_repr_length,
            batch_first=True,
        )

        # Belief update MLP
        state_and_talk_rep_len = state_repr_length + talk_embed_length * (
            num_agents - 1
        )
        self.after_talk_mlp = nn.Sequential(
            nn.Linear(state_and_talk_rep_len, state_and_talk_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_talk_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )
        state_and_reply_rep_len = state_repr_length + reply_embed_length * (
            num_agents - 1
        )
        self.after_reply_mlp = nn.Sequential(
            nn.Linear(state_and_reply_rep_len, state_and_reply_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_reply_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )

        if coordinate_actions:
            # Randomization MLP
            self.to_randomization_logits = nn.Sequential(
                nn.Linear(self.num_agents * reply_embed_length, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.coordinate_actions_dim),
            )

            # self.marginal_linear_actor = nn.Linear(state_repr_length, self.num_outputs)
            # self.marginal_linear_actor.weight.data = norm_col_init(
            #     self.marginal_linear_actor.weight.data, 0.01
            # )
            # self.marginal_linear_actor.bias.data.fill_(0)

        # Linear actor
        self.actor_linear = None
        if coordinate_actions:
            if separate_actor_weights:
                self.actor_linear_list = nn.ModuleList(
                    [
                        nn.Linear(
                            state_repr_length,
                            self.num_outputs * self.coordinate_actions_dim,
                        )
                        for _ in range(2)
                    ]
                )
            else:
                self.actor_linear = nn.Linear(
                    state_repr_length, self.num_outputs * self.coordinate_actions_dim
                )
        else:
            assert not separate_actor_weights
            self.actor_linear = nn.Linear(state_repr_length, self.num_outputs)

        if self.actor_linear is not None:
            self.actor_linear.weight.data = norm_col_init(
                self.actor_linear.weight.data, 0.01
            )
            self.actor_linear.bias.data.fill_(0)
        else:
            for al in self.actor_linear_list:
                al.weight.data = norm_col_init(al.weight.data, 0.01)
                al.bias.data.fill_(0)

        # Linear critic
        if self.central_critic:
            self.critic_linear = nn.Linear(state_repr_length * self.num_agents, 1)
        else:
            self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv5"].weight.data.mul_(relu_gain)

        self.talk_symbol_classifier.weight.data = norm_col_init(
            self.talk_symbol_classifier.weight.data, 0.01
        )
        self.talk_symbol_classifier.bias.data.fill_(0)
        self.reply_symbol_classifier.weight.data = norm_col_init(
            self.reply_symbol_classifier.weight.data, 0.01
        )
        self.reply_symbol_classifier.bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        x = self.cnn(inputs)
        # x.shape == (2, 128, 2, 2)

        x = x.view(x.size(0), -1)
        # x.shape = [num_agents, 512]

        x = torch.cat((x, self.agent_num_embeddings), dim=1)
        # x.shape = [num_agents, 512 + agent_num_embed_length]

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [num_agents, 1, state_repr_length]
        # hidden[0].shape == [1, num_agents, state_repr_length]
        # hidden[1].shape == [1, num_agents, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [num_agents, state_repr_length]

        talk_logits = self.talk_symbol_classifier(x)
        talk_probs = F.softmax(talk_logits, dim=1)
        # talk_probs.shape = [num_agents, num_talk_symbols]

        talk_outputs = torch.mm(talk_probs, self.talk_embeddings.weight)
        # talk_outputs.shape = [num_agents, talk_embed_length]

        if not self.turn_off_communication:
            talk_heard_per_agent = _unfold_communications(talk_outputs)
        else:
            talk_heard_per_agent = torch.cat(
                [talk_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_repr = x + self.after_talk_mlp(
            torch.cat((x, talk_heard_per_agent), dim=1)
        )
        # feature_talk_repr.shape = [num_agents, state_repr_length]

        reply_logits = self.reply_symbol_classifier(state_talk_repr)
        reply_probs = F.softmax(reply_logits, dim=1)
        # reply_probs.shape = [num_agents, num_reply_symbols]

        reply_outputs = torch.mm(reply_probs, self.reply_embeddings.weight)
        # reply_outputs.shape = [num_agents, reply_embed_length]

        if not self.turn_off_communication:
            reply_heard_per_agent = _unfold_communications(reply_outputs)
        else:
            reply_heard_per_agent = torch.cat(
                [reply_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_reply_repr = state_talk_repr + self.after_reply_mlp(
            torch.cat((state_talk_repr, reply_heard_per_agent), dim=1)
        )
        # state_talk_reply_repr.shape = [num_agents, state_repr_length]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (state_talk_reply_repr.unsqueeze(0), hidden[1])
        if self.central_critic:
            value_all = self.critic_linear(
                torch.cat(
                    [
                        state_talk_reply_repr,
                        _unfold_communications(state_talk_reply_repr),
                    ],
                    dim=1,
                )
            )
        else:
            value_all = self.critic_linear(state_talk_reply_repr)

        to_return = {
            "value_all": value_all,
            "hidden_all": hidden,
            "talk_probs": talk_probs,
            "reply_probs": reply_probs,
        }

        if self.coordinate_actions:
            if self.num_agents != 2:
                to_return["randomization_logits"] = self.to_randomization_logits(
                    reply_outputs.view(1, -1)
                )
            else:
                to_return["randomization_logits"] = self.to_randomization_logits(
                    reply_heard_per_agent.view(1, -1)
                )
            if not self.separate_actor_weights:
                logits = self.actor_linear(state_talk_reply_repr)
            else:
                logits = torch.cat(
                    [
                        linear(state_talk_reply_repr[i].unsqueeze(0))
                        for i, linear in enumerate(self.actor_linear_list)
                    ],
                    dim=0,
                )

            # logits = self.actor_linear(
            #     state_talk_reply_repr
            # ) + self.marginal_linear_actor(state_talk_reply_repr).unsqueeze(1).repeat(
            #     1, self.coordinate_actions_dim, 1
            # ).view(
            #     self.num_agents, self.num_outputs ** 2
            # )
            to_return["coordinated_logits"] = logits
        else:
            to_return["logit_all"] = self.actor_linear(state_talk_reply_repr)

        return to_return


class BigA3CLSTMNStepComCoordinatedActionsEgoVision(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        talk_embed_length: int,
        agent_num_embed_length: int,
        reply_embed_length: int,
        num_talk_symbols: int,
        num_reply_symbols: int,
        turn_off_communication: bool,
        coordinate_actions: bool,
        coordinate_actions_dim: Optional[int] = None,
        central_critic: bool = False,
        separate_actor_weights: bool = False,
        final_cnn_channels: int = 64,
    ):
        super(BigA3CLSTMNStepComCoordinatedActionsEgoVision, self).__init__()
        self.num_outputs = sum(len(x) for x in action_groups)

        self.turn_off_communication = turn_off_communication
        self.central_critic = central_critic
        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_agents = num_agents
        self.num_talk_symbols = num_talk_symbols
        self.num_reply_symbols = num_reply_symbols
        self.separate_actor_weights = separate_actor_weights
        self.coordinate_actions_dim = (
            self.num_outputs
            if coordinate_actions_dim is None
            else coordinate_actions_dim
        )

        self.coordinate_actions = coordinate_actions

        # input to conv is (num_agents, self.num_inputs_per_agent, 84, 84)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent, 32, 5, stride=1, padding=2
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(32, 32, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(32, 64, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(2, 2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    (
                        "conv4",
                        nn.Conv2d(64, final_cnn_channels, 3, stride=1, padding=1),
                    ),
                    ("maxpool4", nn.MaxPool2d(2, 2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                ]
            )
        )

        # Vocab embed
        self.talk_embeddings = nn.Embedding(num_talk_symbols, talk_embed_length)
        self.reply_embeddings = nn.Embedding(num_reply_symbols, reply_embed_length)

        self.talk_symbol_classifier = nn.Linear(state_repr_length, num_talk_symbols)
        self.reply_symbol_classifier = nn.Linear(state_repr_length, num_reply_symbols)

        # Agent embed
        self.agent_num_embeddings = nn.Parameter(
            torch.rand(self.num_agents, agent_num_embed_length)
        )

        # LSTM
        self.lstm = nn.LSTM(
            final_cnn_channels * 4 * 4 + agent_num_embed_length,
            state_repr_length,
            batch_first=True,
        )

        # Belief update MLP
        state_and_talk_rep_len = state_repr_length + talk_embed_length * (
            num_agents - 1
        )
        self.after_talk_mlp = nn.Sequential(
            nn.Linear(state_and_talk_rep_len, state_and_talk_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_talk_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )
        state_and_reply_rep_len = state_repr_length + reply_embed_length * (
            num_agents - 1
        )
        self.after_reply_mlp = nn.Sequential(
            nn.Linear(state_and_reply_rep_len, state_and_reply_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_reply_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )

        if coordinate_actions:
            # Randomization MLP
            self.to_randomization_logits = nn.Sequential(
                nn.Linear(self.num_agents * reply_embed_length, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.coordinate_actions_dim),
            )

            # self.marginal_linear_actor = nn.Linear(state_repr_length, self.num_outputs)
            # self.marginal_linear_actor.weight.data = norm_col_init(
            #     self.marginal_linear_actor.weight.data, 0.01
            # )
            # self.marginal_linear_actor.bias.data.fill_(0)

        # Linear actor
        self.actor_linear = None
        if coordinate_actions:
            if separate_actor_weights:
                self.actor_linear_list = nn.ModuleList(
                    [
                        nn.Linear(
                            state_repr_length,
                            self.num_outputs * self.coordinate_actions_dim,
                        )
                        for _ in range(2)
                    ]
                )
            else:
                self.actor_linear = nn.Linear(
                    state_repr_length, self.num_outputs * self.coordinate_actions_dim
                )
        else:
            assert not separate_actor_weights
            self.actor_linear = nn.Linear(state_repr_length, self.num_outputs)

        if self.actor_linear is not None:
            self.actor_linear.weight.data = norm_col_init(
                self.actor_linear.weight.data, 0.01
            )
            self.actor_linear.bias.data.fill_(0)
        else:
            for al in self.actor_linear_list:
                al.weight.data = norm_col_init(al.weight.data, 0.01)
                al.bias.data.fill_(0)

        # Linear critic
        if self.central_critic:
            self.critic_linear = nn.Linear(state_repr_length * self.num_agents, 1)
        else:
            self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        self.talk_symbol_classifier.weight.data = norm_col_init(
            self.talk_symbol_classifier.weight.data, 0.01
        )
        self.talk_symbol_classifier.bias.data.fill_(0)
        self.reply_symbol_classifier.weight.data = norm_col_init(
            self.reply_symbol_classifier.weight.data, 0.01
        )
        self.reply_symbol_classifier.bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        x = self.cnn(inputs)
        # x.shape == (2, 128, 2, 2)

        x = x.view(x.size(0), -1)
        # x.shape = [num_agents, 512]

        x = torch.cat((x, self.agent_num_embeddings), dim=1)
        # x.shape = [num_agents, 512 + agent_num_embed_length]

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [num_agents, 1, state_repr_length]
        # hidden[0].shape == [1, num_agents, state_repr_length]
        # hidden[1].shape == [1, num_agents, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [num_agents, state_repr_length]

        talk_logits = self.talk_symbol_classifier(x)
        talk_probs = F.softmax(talk_logits, dim=1)
        # talk_probs.shape = [num_agents, num_talk_symbols]

        talk_outputs = torch.mm(talk_probs, self.talk_embeddings.weight)
        # talk_outputs.shape = [num_agents, talk_embed_length]

        if not self.turn_off_communication:
            talk_heard_per_agent = _unfold_communications(talk_outputs)
        else:
            talk_heard_per_agent = torch.cat(
                [talk_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_repr = x + self.after_talk_mlp(
            torch.cat((x, talk_heard_per_agent), dim=1)
        )
        # feature_talk_repr.shape = [num_agents, state_repr_length]

        reply_logits = self.reply_symbol_classifier(state_talk_repr)
        reply_probs = F.softmax(reply_logits, dim=1)
        # reply_probs.shape = [num_agents, num_reply_symbols]

        reply_outputs = torch.mm(reply_probs, self.reply_embeddings.weight)
        # reply_outputs.shape = [num_agents, reply_embed_length]

        if not self.turn_off_communication:
            reply_heard_per_agent = _unfold_communications(reply_outputs)
        else:
            reply_heard_per_agent = torch.cat(
                [reply_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_reply_repr = state_talk_repr + self.after_reply_mlp(
            torch.cat((state_talk_repr, reply_heard_per_agent), dim=1)
        )
        # state_talk_reply_repr.shape = [num_agents, state_repr_length]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (state_talk_reply_repr.unsqueeze(0), hidden[1])
        if self.central_critic:
            value_all = self.critic_linear(
                torch.cat(
                    [
                        state_talk_reply_repr,
                        _unfold_communications(state_talk_reply_repr),
                    ],
                    dim=1,
                )
            )
        else:
            value_all = self.critic_linear(state_talk_reply_repr)

        to_return = {
            "value_all": value_all,
            "hidden_all": hidden,
            "talk_probs": talk_probs,
            "reply_probs": reply_probs,
        }

        if self.coordinate_actions:
            to_return["randomization_logits"] = self.to_randomization_logits(
                reply_heard_per_agent.view(1, -1)
            )
            if not self.separate_actor_weights:
                logits = self.actor_linear(state_talk_reply_repr)
            else:
                logits = torch.cat(
                    [
                        linear(state_talk_reply_repr[i].unsqueeze(0))
                        for i, linear in enumerate(self.actor_linear_list)
                    ],
                    dim=0,
                )

            # logits = self.actor_linear(
            #     state_talk_reply_repr
            # ) + self.marginal_linear_actor(state_talk_reply_repr).unsqueeze(1).repeat(
            #     1, self.coordinate_actions_dim, 1
            # ).view(
            #     self.num_agents, self.num_outputs ** 2
            # )
            to_return["coordinated_logits"] = logits
        else:
            to_return["logit_all"] = self.actor_linear(state_talk_reply_repr)

        return to_return


class A3CLSTMNStepComCoordinatedActionsEgoGridsEmbedCNN(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        talk_embed_length: int,
        agent_num_embed_length: int,
        reply_embed_length: int,
        num_talk_symbols: int,
        num_reply_symbols: int,
        occupancy_embed_length: int,
        turn_off_communication: bool,
        coordinate_actions: bool,
        coordinate_actions_dim: Optional[int] = None,
        central_critic: bool = False,
        separate_actor_weights: bool = False,
        final_cnn_channels: int = 256,
    ):
        super(A3CLSTMNStepComCoordinatedActionsEgoGridsEmbedCNN, self).__init__()
        self.num_outputs = sum(len(x) for x in action_groups)

        # assert not turn_off_communication
        self.turn_off_communication = turn_off_communication
        self.central_critic = central_critic
        self.num_inputs = num_inputs
        self.num_agents = num_agents
        self.num_talk_symbols = num_talk_symbols
        self.num_reply_symbols = num_reply_symbols
        self.separate_actor_weights = separate_actor_weights
        self.coordinate_actions_dim = (
            self.num_outputs
            if coordinate_actions_dim is None
            else coordinate_actions_dim
        )

        self.occupancy_embed_length = occupancy_embed_length
        self.occupancy_embeddings = nn.Embedding(
            num_inputs, self.occupancy_embed_length
        )
        self.coordinate_actions = coordinate_actions

        # input to conv is (num_agents, self.occupancy_embed_length, 15, 15)
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.occupancy_embed_length, 32, 3, padding=1)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape = (15, 15)
                    ("conv2", nn.Conv2d(32, 64, 3, stride=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape = (7, 7)
                    ("conv3", nn.Conv2d(64, 128, 3, stride=2)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape = (3, 3)
                    ("conv4", nn.Conv2d(128, final_cnn_channels, 3, stride=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (1, 1); Stride doesn't matter above
                ]
            )
        )

        # Vocab embed
        self.talk_embeddings = nn.Embedding(num_talk_symbols, talk_embed_length)
        self.reply_embeddings = nn.Embedding(num_reply_symbols, reply_embed_length)

        self.talk_symbol_classifier = nn.Linear(state_repr_length, num_talk_symbols)
        self.reply_symbol_classifier = nn.Linear(state_repr_length, num_reply_symbols)

        # Agent embed
        self.agent_num_embeddings = nn.Parameter(
            torch.rand(self.num_agents, agent_num_embed_length)
        )

        # LSTM
        self.lstm = nn.LSTM(
            final_cnn_channels * 1 + agent_num_embed_length,
            state_repr_length,
            batch_first=True,
        )

        # Belief update MLP
        state_and_talk_rep_len = state_repr_length + talk_embed_length * (
            num_agents - 1
        )
        self.after_talk_mlp = nn.Sequential(
            nn.Linear(state_and_talk_rep_len, state_and_talk_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_talk_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )
        state_and_reply_rep_len = state_repr_length + reply_embed_length * (
            num_agents - 1
        )
        self.after_reply_mlp = nn.Sequential(
            nn.Linear(state_and_reply_rep_len, state_and_reply_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_reply_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )

        if coordinate_actions:
            # Randomization MLP
            self.to_randomization_logits = nn.Sequential(
                nn.Linear(self.num_agents * reply_embed_length, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.coordinate_actions_dim),
            )

            # self.marginal_linear_actor = nn.Linear(state_repr_length, self.num_outputs)
            # self.marginal_linear_actor.weight.data = norm_col_init(
            #     self.marginal_linear_actor.weight.data, 0.01
            # )
            # self.marginal_linear_actor.bias.data.fill_(0)

        # Linear actor
        self.actor_linear = None
        if coordinate_actions:
            if separate_actor_weights:
                self.actor_linear_list = nn.ModuleList(
                    [
                        nn.Linear(
                            state_repr_length,
                            self.num_outputs * self.coordinate_actions_dim,
                        )
                        for _ in range(2)
                    ]
                )
            else:
                self.actor_linear = nn.Linear(
                    state_repr_length, self.num_outputs * self.coordinate_actions_dim
                )
        else:
            assert not separate_actor_weights
            self.actor_linear = nn.Linear(state_repr_length, self.num_outputs)

        if self.actor_linear is not None:
            self.actor_linear.weight.data = norm_col_init(
                self.actor_linear.weight.data, 0.01
            )
            self.actor_linear.bias.data.fill_(0)
        else:
            for al in self.actor_linear_list:
                al.weight.data = norm_col_init(al.weight.data, 0.01)
                al.bias.data.fill_(0)

        # Linear critic
        if self.central_critic:
            self.critic_linear = nn.Linear(state_repr_length * self.num_agents, 1)
        else:
            self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        self.talk_symbol_classifier.weight.data = norm_col_init(
            self.talk_symbol_classifier.weight.data, 0.01
        )
        self.talk_symbol_classifier.bias.data.fill_(0)
        self.reply_symbol_classifier.weight.data = norm_col_init(
            self.reply_symbol_classifier.weight.data, 0.01
        )
        self.reply_symbol_classifier.bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs, 15, 15):
            raise Exception("input to model is not as expected, check!")

        inputs = inputs.float()

        # inputs.shape == (self.num_agents, self.num_inputs, 15, 15)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        original_shape = inputs.shape
        # original_shape == (self.num_agents, 15, 15, self.num_inputs)

        scaling_factor = torch.reciprocal(
            (inputs == 1).sum(dim=3).unsqueeze(3).float() + 1e-3
        )
        # scaling_factor.shape == (2, 15, 15, 1)

        inputs = inputs.view(-1, self.num_inputs)
        inputs = inputs.matmul(self.occupancy_embeddings.weight)
        inputs = inputs.view(
            original_shape[0],
            original_shape[1],
            original_shape[2],
            self.occupancy_embed_length,
        )
        x = torch.mul(inputs, scaling_factor.expand_as(inputs))
        # x.shape == (self.num_agents, 15, 15, self.occupancy_embed_length)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x.shape == (self.num_agents, self.occupancy_embed_length, 15, 15)

        x = self.cnn(x)
        # x.shape == (2, 256, 1, 1)

        x = x.view(x.size(0), -1)
        # x.shape = [num_agents, 256]

        x = torch.cat((x, self.agent_num_embeddings), dim=1)
        # x.shape = [num_agents, 256 + agent_num_embed_length]

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [num_agents, 1, state_repr_length]
        # hidden[0].shape == [1, num_agents, state_repr_length]
        # hidden[1].shape == [1, num_agents, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [num_agents, state_repr_length]

        talk_logits = self.talk_symbol_classifier(x)
        talk_probs = F.softmax(talk_logits, dim=1)
        # talk_probs.shape = [num_agents, num_talk_symbols]

        talk_outputs = torch.mm(talk_probs, self.talk_embeddings.weight)
        # talk_outputs.shape = [num_agents, talk_embed_length]

        if not self.turn_off_communication:
            talk_heard_per_agent = _unfold_communications(talk_outputs)
        else:
            talk_heard_per_agent = torch.cat(
                [talk_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_repr = x + self.after_talk_mlp(
            torch.cat((x, talk_heard_per_agent), dim=1)
        )
        # feature_talk_repr.shape = [num_agents, state_repr_length]

        reply_logits = self.reply_symbol_classifier(state_talk_repr)
        reply_probs = F.softmax(reply_logits, dim=1)
        # reply_probs.shape = [num_agents, num_reply_symbols]

        reply_outputs = torch.mm(reply_probs, self.reply_embeddings.weight)
        # reply_outputs.shape = [num_agents, reply_embed_length]

        if not self.turn_off_communication:
            reply_heard_per_agent = _unfold_communications(reply_outputs)
        else:
            reply_heard_per_agent = torch.cat(
                [reply_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_reply_repr = state_talk_repr + self.after_reply_mlp(
            torch.cat((state_talk_repr, reply_heard_per_agent), dim=1)
        )
        # state_talk_reply_repr.shape = [num_agents, state_repr_length]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (state_talk_reply_repr.unsqueeze(0), hidden[1])
        if self.central_critic:
            value_all = self.critic_linear(
                torch.cat(
                    [
                        state_talk_reply_repr,
                        _unfold_communications(state_talk_reply_repr),
                    ],
                    dim=1,
                )
            )
        else:
            value_all = self.critic_linear(state_talk_reply_repr)

        to_return = {
            "value_all": value_all,
            "hidden_all": hidden,
            "talk_probs": talk_probs,
            "reply_probs": reply_probs,
        }

        if self.coordinate_actions:
            if self.num_agents != 2:
                to_return["randomization_logits"] = self.to_randomization_logits(
                    reply_outputs.view(1, -1)
                )
            else:
                to_return["randomization_logits"] = self.to_randomization_logits(
                    reply_heard_per_agent.view(1, -1)
                )
            if not self.separate_actor_weights:
                logits = self.actor_linear(state_talk_reply_repr)
            else:
                logits = torch.cat(
                    [
                        linear(state_talk_reply_repr[i].unsqueeze(0))
                        for i, linear in enumerate(self.actor_linear_list)
                    ],
                    dim=0,
                )

            # logits = self.actor_linear(
            #     state_talk_reply_repr
            # ) + self.marginal_linear_actor(state_talk_reply_repr).unsqueeze(1).repeat(
            #     1, self.coordinate_actions_dim, 1
            # ).view(
            #     self.num_agents, self.num_outputs ** 2
            # )
            to_return["coordinated_logits"] = logits
        else:
            to_return["logit_all"] = self.actor_linear(state_talk_reply_repr)

        return to_return
