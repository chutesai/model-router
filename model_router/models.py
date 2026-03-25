"""Model registry and routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Task types for routing decisions."""

    GENERAL_TEXT = "general_text"
    MATH_REASONING = "math_reasoning"
    GENERAL_REASONING = "general_reasoning"
    PROGRAMMING = "programming"
    CREATIVE = "creative"
    VISION = "vision"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for a backend model."""

    model_id: str
    display_name: str
    task_types: list[TaskType]
    priority: int
    max_tokens: int = 8192
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    timeout_seconds: float = 120.0
    exclude_from_routing: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ── Classifier (excluded from routing) ──────────────────────────────
    "classifier": ModelConfig(
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        display_name="Qwen3 Next 80B (Classifier)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=2048,
        timeout_seconds=15.0,
        exclude_from_routing=True,
    ),
    "classifier_fallback": ModelConfig(
        model_id="Qwen/Qwen3-32B-TEE",
        display_name="Qwen3 32B (Classifier Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=1024,
        timeout_seconds=10.0,
        exclude_from_routing=True,
    ),
    "classifier_fallback2": ModelConfig(
        model_id="XiaomiMiMo/MiMo-V2-Flash-TEE",
        display_name="MiMo V2 Flash TEE (Classifier Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=1024,
        timeout_seconds=10.0,
        exclude_from_routing=True,
    ),
    # ── General Text ────────────────────────────────────────────────────
    "general": ModelConfig(
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        display_name="Qwen3 Next 80B",
        task_types=[TaskType.GENERAL_TEXT, TaskType.UNKNOWN],
        priority=2,
        max_tokens=8192,
        timeout_seconds=60.0,
    ),
    "general_fallback": ModelConfig(
        model_id="Qwen/Qwen3-32B-TEE",
        display_name="Qwen3 32B",
        task_types=[TaskType.GENERAL_TEXT],
        priority=3,
        max_tokens=8192,
        timeout_seconds=60.0,
    ),
    "general_fallback2": ModelConfig(
        model_id="XiaomiMiMo/MiMo-V2-Flash-TEE",
        display_name="MiMo V2 Flash TEE",
        task_types=[TaskType.GENERAL_TEXT],
        priority=4,
        max_tokens=4096,
        timeout_seconds=30.0,
    ),
    # ── Math Reasoning ──────────────────────────────────────────────────
    "reasoning": ModelConfig(
        model_id="deepseek-ai/DeepSeek-V3.2-Speciale-TEE",
        display_name="DeepSeek V3.2 Speciale",
        task_types=[TaskType.MATH_REASONING],
        priority=5,
        max_tokens=16384,
        timeout_seconds=120.0,
    ),
    # ── General Reasoning (NEW) ─────────────────────────────────────────
    "general_reasoning": ModelConfig(
        model_id="moonshotai/Kimi-K2.5-TEE",
        display_name="Kimi K2.5",
        task_types=[TaskType.GENERAL_REASONING],
        priority=6,
        max_tokens=16384,
        timeout_seconds=120.0,
    ),
    "general_reasoning_fallback": ModelConfig(
        model_id="zai-org/GLM-5-TEE",
        display_name="GLM 5 (General Reasoning Fallback)",
        task_types=[TaskType.GENERAL_REASONING],
        priority=7,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    "general_reasoning_fallback2": ModelConfig(
        model_id="MiniMaxAI/MiniMax-M2.5-TEE",
        display_name="MiniMax M2.5 (General Reasoning Fallback)",
        task_types=[TaskType.GENERAL_REASONING],
        priority=8,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    # ── Programming ─────────────────────────────────────────────────────
    "programming": ModelConfig(
        model_id="MiniMaxAI/MiniMax-M2.5-TEE",
        display_name="MiniMax M2.5",
        task_types=[TaskType.PROGRAMMING],
        priority=9,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    "programming_fallback": ModelConfig(
        model_id="zai-org/GLM-5-TEE",
        display_name="GLM 5 (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=10,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    "programming_fallback2": ModelConfig(
        model_id="MiniMaxAI/MiniMax-M2.1-TEE",
        display_name="MiniMax M2.1 (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=11,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    "programming_fallback3": ModelConfig(
        model_id="deepseek-ai/DeepSeek-V3.2-TEE",
        display_name="DeepSeek V3.2 (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=12,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    "programming_fallback4": ModelConfig(
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
        display_name="Qwen3 235B (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=13,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    # ── Creative ────────────────────────────────────────────────────────
    "creative": ModelConfig(
        model_id="tngtech/DeepSeek-TNG-R1T2-Chimera",
        display_name="TNG R1T2 Chimera",
        task_types=[TaskType.CREATIVE],
        priority=14,
        max_tokens=16384,
        timeout_seconds=90.0,
    ),
    # ── Vision ──────────────────────────────────────────────────────────
    "vision": ModelConfig(
        model_id="Qwen/Qwen3.5-397B-A17B-TEE",
        display_name="Qwen3.5 397B",
        task_types=[TaskType.VISION],
        priority=15,
        max_tokens=8192,
        supports_vision=True,
        timeout_seconds=120.0,
    ),
    "vision_fallback": ModelConfig(
        model_id="moonshotai/Kimi-K2.5-TEE",
        display_name="Kimi K2.5 (Vision Fallback)",
        task_types=[TaskType.VISION],
        priority=16,
        max_tokens=8192,
        supports_vision=True,
        timeout_seconds=90.0,
    ),
    "vision_fallback2": ModelConfig(
        model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
        display_name="Qwen3 VL 235B (Vision Fallback)",
        task_types=[TaskType.VISION],
        priority=17,
        max_tokens=8192,
        supports_vision=True,
        timeout_seconds=90.0,
    ),
    "vision_fallback3": ModelConfig(
        model_id="chutesai/Mistral-Small-3.2-24B-Instruct-2506",
        display_name="Mistral Small 3.2 (Vision Fallback)",
        task_types=[TaskType.VISION],
        priority=18,
        max_tokens=8192,
        supports_vision=True,
        timeout_seconds=60.0,
    ),
    # ── Universal Fallback (ALL task types including vision) ─────────────
    "universal_fallback": ModelConfig(
        model_id="moonshotai/Kimi-K2.5-TEE",
        display_name="Kimi K2.5 (Universal Fallback)",
        task_types=[
            TaskType.GENERAL_TEXT,
            TaskType.MATH_REASONING,
            TaskType.GENERAL_REASONING,
            TaskType.PROGRAMMING,
            TaskType.CREATIVE,
            TaskType.VISION,
            TaskType.UNKNOWN,
        ],
        priority=99,
        max_tokens=16384,
        supports_vision=True,
        timeout_seconds=120.0,
    ),
}


def get_model_for_task(task_type: TaskType) -> ModelConfig:
    """Get the primary model for a task type."""
    for config in sorted(MODEL_REGISTRY.values(), key=lambda config: config.priority):
        if config.exclude_from_routing:
            continue
        if task_type in config.task_types:
            return config
    return MODEL_REGISTRY["general"]


def get_fallback_models(
    primary_model_id: str,
    task_type: TaskType | None = None,
) -> list[ModelConfig]:
    """Get fallback models when primary fails.

    Prioritizes models sharing the same task type, then adds cross-type
    fallbacks. Deduplicates by model_id so each upstream is tried at most once.
    """
    primary = next(
        (config for config in MODEL_REGISTRY.values() if config.model_id == primary_model_id),
        None,
    )
    if not primary:
        return [MODEL_REGISTRY["general"]]

    same_task: list[ModelConfig] = []
    other: list[ModelConfig] = []

    for config in sorted(MODEL_REGISTRY.values(), key=lambda config: config.priority):
        if config.exclude_from_routing:
            continue
        if config.model_id == primary_model_id:
            continue
        # Vision primary → only consider vision-capable models
        if primary.supports_vision and not config.supports_vision:
            continue
        # Non-vision primary → skip vision-ONLY models (but allow universal fallbacks
        # that support both vision and non-vision task types)
        if not primary.supports_vision:
            is_vision_only = all(t == TaskType.VISION for t in config.task_types)
            if is_vision_only:
                continue

        if task_type and task_type in config.task_types:
            same_task.append(config)
        else:
            other.append(config)

    # Deduplicate by model_id, preserving priority order
    seen: set[str] = set()
    result: list[ModelConfig] = []
    for config in same_task + other:
        if config.model_id not in seen:
            seen.add(config.model_id)
            result.append(config)

    return result[:5]
