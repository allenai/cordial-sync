from __future__ import division

import math
from collections import defaultdict
from typing import Tuple, Dict, Union, Optional, Iterable, Callable

import torch
import torch.optim as optim
from torch import nn


class SharedRMSprop(optim.Optimizer):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Dict],
        lr: float = 7e-4,
        alpha: float = 0.99,
        eps: float = 0.1,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        saved_state: Dict = None,
    ) -> None:
        defaults: defaultdict = defaultdict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super(SharedRMSprop, self).__init__(params, defaults)
        if saved_state:
            self.load_state_dict(saved_state)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = torch.zeros(1)
                    state["grad_avg"] = p.data.new().resize_as_(p.data).zero_()
                    state["square_avg"] = p.data.new().resize_as_(p.data).zero_()
                    state["momentum_buffer"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["square_avg"].share_memory_()
                state["step"].share_memory_()
                state["grad_avg"].share_memory_()
                state["momentum_buffer"].share_memory_()

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.FloatTensor]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = (
                        square_avg.addcmul(-1, grad_avg, grad_avg)
                        .sqrt()
                        .add_(group["eps"])
                    )
                else:
                    avg = square_avg.sqrt().add_(group["eps"])

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.data.add_(-group["lr"], buf)
                else:
                    p.data.addcdiv_(-group["lr"], grad, avg)

        return loss


class SharedAdam(optim.Optimizer):
    """Implements Adam algorithm with shared states.
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Dict],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0,
        amsgrad: bool = False,
        saved_state: Dict = None,
    ) -> None:
        defaults: defaultdict = defaultdict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(SharedAdam, self).__init__(params, defaults)
        if saved_state:
            self.load_state_dict(saved_state)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = torch.zeros(1)
                    state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                    state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()
                    state["max_exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["max_exp_avg_sq"].share_memory_()

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.FloatTensor]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"][0]
                bias_correction2 = 1 - beta2 ** state["step"][0]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
