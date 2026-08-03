# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_full_precision(function, *args, **kwargs):
    """Checkpoint a module while casting floating-point inputs inside the region.

    FSDP2 does not repeat ``MixedPrecisionPolicy.cast_forward_inputs`` when a
    checkpointed module is recomputed during backward. Target prediction blocks
    use an FP32 FSDP policy, so relying on the FSDP pre-forward hook makes the
    original forward see FP32 inputs while recomputation can see BF16 inputs.
    Keeping the cast inside the checkpointed function makes both executions
    identical without storing FP32 checkpoint inputs.
    """

    def forward_full_precision(module, *forward_args, **forward_kwargs):
        def to_float32(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value.float()
            return value

        forward_args = tuple(to_float32(value) for value in forward_args)
        forward_kwargs = {
            key: to_float32(value) for key, value in forward_kwargs.items()
        }
        return module(*forward_args, **forward_kwargs)

    return checkpoint(
        forward_full_precision,
        function,
        *args,
        use_reentrant=False,
        **kwargs,
    )
