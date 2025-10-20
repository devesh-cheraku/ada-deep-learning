# code completed with AI assistance.
from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)


        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Make sure adapters are trainable
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

        # Scaling factor
        alpha = float(lora_dim)
        self.register_buffer("alpha_div_rank", torch.tensor(alpha / float(lora_dim), dtype=torch.float32))

        # Recommended inits
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Base path uses 4-bit dequantized weights (frozen) from Linear4Bit.
        LoRA path runs in fp32 and is added as a residual.
        Output dtype matches the input dtype.
        """
               # Cast inputs to float32 for the base 4-bit path (to match reference),
        x_dtype = x.dtype

        base_out = super().forward(x.to(torch.float32))  # Linear4Bit expects/uses fp32 internally
        # LoRA path in full precision (fp32)
        delta = self.lora_b(self.lora_a(x.to(torch.float32))) * self.alpha_div_rank  # fp32

        # Combine paths and cast back to input dtype
        return (base_out + delta.to(base_out.dtype)).to(x_dtype)



class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int, group_size: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
