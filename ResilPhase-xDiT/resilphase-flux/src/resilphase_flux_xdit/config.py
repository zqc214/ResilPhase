from dataclasses import dataclass


@dataclass
class ResilPhaseConfig:
    num_steps: int = 50
    fresh_threshold: int = 6
    max_order: int = 1
    first_enhance: int = 3
    mapping_method: str = "balanced"
    balance_alpha: float = 0.55

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.fresh_threshold <= 0:
            raise ValueError("fresh_threshold must be positive.")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if self.first_enhance < 0:
            raise ValueError("first_enhance must be non-negative.")
        if self.mapping_method not in {"balanced", "chebyshev"}:
            raise ValueError("mapping_method must be 'balanced' or 'chebyshev'.")
