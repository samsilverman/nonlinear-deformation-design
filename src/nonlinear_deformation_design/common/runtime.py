from __future__ import annotations

import torch

from .device import get_device

# Default device.
DEFAULT_DEVICE = get_device()

# Default floating-point dtype.
DEFAULT_DTYPE = torch.float32
