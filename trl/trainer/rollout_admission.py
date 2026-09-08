"""Alignment and accumulation for externally admitted complete rollout groups."""

import torch


class NoAdmittedRollouts(RuntimeError):
    """An external candidate round contains no complete, usable groups."""


def retained_rollout_indices(indices, input_count, output_count, group_size):
    """Reject ambiguous identities and partial groups before rewards or tensors."""
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError("retained_input_indices must be a list of integer source-row indices")
    if len(indices) != output_count or indices != sorted(set(indices)):
        raise ValueError("retained_input_indices must be ordered, unique and align with rollout rows")
    if not indices:
        raise NoAdmittedRollouts("rollout admission retained no complete groups")
    if indices[0] < 0 or indices[-1] >= input_count:
        raise ValueError("rollout admission requires at least one valid retained group")
    selected = set(indices)
    if input_count % group_size or any(
        set(range(index // group_size * group_size, (index // group_size + 1) * group_size)) - selected
        for index in indices
    ):
        raise ValueError("rollout admission must retain complete prompt groups")
    return indices


def pad_admitted_rollout_batch(batch, scheduled_rows):
    """Keep scheduled accumulation slots; padded rows have no policy/KL credit.

    Padding is applied only after reward normalization and observation. These
    are masked tensor slots, never synthetic environment evidence or rewards.
    """
    retained_rows = batch["completion_ids"].shape[0]
    if not 0 < retained_rows <= scheduled_rows:
        raise ValueError("invalid admitted rollout batch size")
    if retained_rows == scheduled_rows:
        return batch
    padded = {}
    indices = torch.arange(scheduled_rows, device=batch["completion_ids"].device) % retained_rows
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != retained_rows:
                raise ValueError(f"rollout tensor {key} does not align with admitted rows")
            padded[key] = value[indices].clone()
        elif isinstance(value, list):
            padded[key] = [value[index % retained_rows] for index in range(scheduled_rows)]
        else:
            padded[key] = value
    padded["completion_mask"][retained_rows:] = 0
    if "tool_mask" in padded:
        padded["tool_mask"][retained_rows:] = 0
    padded["advantages"][retained_rows:] = 0
    padded["admission_loss_scale"] = torch.tensor(scheduled_rows / retained_rows, device=indices.device)
    return padded
