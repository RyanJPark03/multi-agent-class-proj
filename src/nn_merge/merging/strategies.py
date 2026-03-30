from typing import Protocol

import torch


class MergeStrategy(Protocol):
    """A merge strategy is any callable: list[state_dict] -> state_dict."""

    def __call__(
        self, state_dicts: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]: ...


def weight_average(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Element-wise mean of all corresponding parameters."""
    merged = {}
    for key in state_dicts[0]:
        merged[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
    return merged

MERGE_STRATEGIES: dict[str, MergeStrategy] = {
    "weight_average": weight_average,
}