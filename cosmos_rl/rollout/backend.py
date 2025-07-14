import enum

from cosmos_rl.policy.config import Config as CosmosConfig


class RolloutBackend(enum.Enum):
    VLLM = "vllm"
    TRT_LLM = "trt_llm"


def determine_backend(config: CosmosConfig) -> RolloutBackend:
    if config.rollout.backend == "vllm":
        return RolloutBackend.VLLM
    elif config.rollout.backend == "trt_llm":
        return RolloutBackend.TRT_LLM
    else:
        raise ValueError(f"[Rollout] Invalid rollout backend: {config.rollout.backend}")
