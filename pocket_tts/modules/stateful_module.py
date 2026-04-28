"""Streaming-state helpers.

State is represented as a `dict[str, dict[str, T]]` mapping module FQN →
state-tensor name → tensor (numpy arrays at runtime, MAX TensorValues during
graph construction). This mirrors the layout used by the legacy PyTorch
implementation so existing voice-prompt safetensors files remain loadable.
"""

from __future__ import annotations

import numpy as np
from max.nn.layer.layer import Module, recursive_named_layers


def init_states(
    model: Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, np.ndarray]]:
    """Build a nested numpy state dict for every stateful module in `model`."""
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, module in recursive_named_layers(model):
        if not isinstance(module, StatefulModule):
            continue
        result[name] = module.init_state(batch_size=batch_size, sequence_length=sequence_length)
    return result


def increment_steps(
    model: Module, model_state: dict[str, dict[str, np.ndarray]], increment: int = 1
) -> None:
    for name, module in recursive_named_layers(model):
        if not isinstance(module, StatefulModule):
            continue
        if name in model_state:
            module.increment_step(model_state[name], increment=increment)


def assign_module_absolute_names(model: Module) -> None:
    """Walk the model tree and stamp each StatefulModule with its FQN.

    The graph-building code uses this name to fetch the module's state slice
    out of a flat state dict — same pattern as the legacy implementation.
    """
    for name, module in recursive_named_layers(model):
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name


class StatefulModule(Module):
    """Marker base class for streaming-state owners."""

    _module_absolute_name: str | None = None

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def increment_step(self, state: dict[str, np.ndarray], increment: int = 1) -> None:
        pass

    def get_state(self, model_state):
        if self._module_absolute_name is None:
            raise RuntimeError(
                "StatefulModule has no absolute name; did you call assign_module_absolute_names()?"
            )
        return model_state[self._module_absolute_name]

    def __call__(self, *args, **kwargs):  # pragma: no cover - subclasses override
        raise NotImplementedError
