from typing import Callable, Sequence, Union, Type

import torch
from hydra.utils import instantiate

from utils.conf_helpers import ModuleConf


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: torch.Tensor,
        base_optimizer_conf: ModuleConf,
        rho: float = 0.05,
        adaptive: bool = False,
        **base_optimizer_overrides
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        base_optimizer_conf.pop('params', None) # hotfix

        super().__init__(params, dict(rho=rho, adaptive=adaptive))

        self.base_optimizer = instantiate(base_optimizer_conf, self.param_groups, **base_optimizer_overrides)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    torch.pow(p, 2)
                    if group["adaptive"]
                    else 1.0
                ) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def lightning_step(
        self,
        model: torch.nn.Module,
        forward: Callable,
        internal_training_disable: bool = True,
        modules: Union[torch.nn.Module, Sequence[torch.nn.Module]] = torch.nn.Module
    ):

        out = forward()
        self.first_step(zero_grad=True)

        switch = model.training and internal_training_disable
        if switch:
            _switch_training(model, training=False, modules=modules)

        forward() # discard second returns
        self.second_step(zero_grad=True)

        if switch:
            _switch_training(model, training=True, modules=modules)

        return out

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def _switch_training(model, training: bool, modules: Union[torch.nn.Module, Sequence[torch.nn.Module]]):
    def _switch(module):
        if isinstance(module, modules):
            module.training = training

    model.apply(_switch)
