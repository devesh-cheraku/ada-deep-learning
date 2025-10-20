# code completed with AI assistance.
from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"lora_dim must be a positive int, got {lora_dim}")

        # Adapters trained in full precision (fp32)
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Ensure adapters are trainable (base weights in HalfLinear can remain frozen)
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True


        eff_alpha = lora_dim 
        self.register_buffer("alpha_div_rank", torch.tensor(eff_alpha / lora_dim, dtype=torch.float32))

        # Recommended inits: A ~ Kaiming, B ~ zeros
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path in half precision (frozen weights from HalfLinear)
        return super().forward(x) + (self.lora_b(self.lora_a(x.float())) * self.alpha_div_rank).to(x.dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim=lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim=lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim=lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
