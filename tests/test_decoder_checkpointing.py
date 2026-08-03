# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch.utils.checkpoint import CheckpointError, checkpoint

from weathergen.model.checkpointing import checkpoint_full_precision


class _ForwardOnlyInputCast(torch.nn.Module):
    """Simulate FSDP2 casting inputs in forward but not in recomputation."""

    def __init__(self):
        super().__init__()
        self.num_calls = 0

    def forward(self, values, lengths):
        self.num_calls += 1
        if self.num_calls == 1:
            values = values.float()
        assert lengths.dtype == torch.int32
        return (values * values).sum()


def test_full_precision_cast_is_repeated_during_checkpoint_recomputation():
    mismatched_module = _ForwardOnlyInputCast()
    mismatched_values = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)
    lengths = torch.tensor([0, 4], dtype=torch.int32)

    mismatched_output = checkpoint(
        mismatched_module,
        mismatched_values,
        lengths,
        use_reentrant=False,
    )
    with pytest.raises(CheckpointError):
        mismatched_output.backward()

    module = _ForwardOnlyInputCast()
    values = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)

    output = checkpoint_full_precision(module, values, lengths)
    output.backward()

    assert module.num_calls == 2
    assert values.grad is not None
    assert values.grad.dtype == torch.bfloat16
