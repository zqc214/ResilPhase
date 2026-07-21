from dataclasses import dataclass


@dataclass
class ResilPhaseDiTConfig:
    num_steps: int = 50
    interval: int = 4
    max_order: int = 4
    first_enhance: int = 2
    mapping_method: str = "chebyshev"
    balance_alpha: float = 0.55

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.interval <= 0:
            raise ValueError("interval must be positive.")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if self.first_enhance < 0:
            raise ValueError("first_enhance must be non-negative.")
        if self.mapping_method not in {"chebyshev", "balanced"}:
            raise ValueError("mapping_method must be 'chebyshev' or 'balanced'.")
