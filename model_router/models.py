"""Model registry and routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Default max_tokens for chat-completion entries when the inbound request
# doesn't set one. Chosen to sit just under the smallest output limit among
# the chat-tier models we route to (Kimi K2.5/K2.6 = 65535, gemma-4 = 65536,
# DeepSeek/Qwen/GLM/MiniMax/MiMo = 65536+). Reasoning models (K2.6, R1, etc.)
# spend a chunk of the token budget on `reasoning_content` BEFORE producing
# `content` — a tight cap of 4-16k starves the visible answer and the
# response shape (content=null + finish_reason=length) can be misread by the
# non-streaming empty-detection path. Models with higher caps clamp this
# value server-side. See README.md "max_tokens & capacity" for context.
DEFAULT_CHAT_MAX_TOKENS = 65_535


class TaskType(Enum):
    """Task types for routing decisions."""

    GENERAL_TEXT = "general_text"
    MATH_REASONING = "math_reasoning"
    GENERAL_REASONING = "general_reasoning"
    PROGRAMMING = "programming"
    CREATIVE = "creative"
    VISION = "vision"
    UNKNOWN = "unknown"


def derive_chute_slug(model_id: str) -> str:
    """Compute the public chute hostname slug from a model_id.

    Chutes follows a consistent convention: `chutes-{lower(model_id)}` with
    `/` and `.` replaced by `-` and any double-dashes squashed. Verified
    against every model_id in the registry on 2026-05-05 — all hostnames
    of the form `https://{slug}.chutes.ai/v1/...` resolve. Owners of the
    chute set the slug on creation; if a future model uses a non-standard
    one, override it explicitly via ModelConfig(slug=...).
    """
    cleaned = model_id.lower().replace("/", "-").replace(".", "-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return f"chutes-{cleaned.strip('-')}"


@dataclass
class ModelConfig:
    """Configuration for a backend model."""

    model_id: str
    display_name: str
    task_types: list[TaskType]
    priority: int
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    timeout_seconds: float = 120.0
    exclude_from_routing: bool = False
    # The hostname slug under chutes.ai (e.g. `chutes-moonshotai-kimi-k2-6-tee`).
    # When None, derive_chute_slug(model_id) is used. Set explicitly only when
    # the chute owner registered a non-standard slug.
    slug: Optional[str] = None

    def chute_slug(self) -> str:
        """The slug to address this model's chute directly (no router hop)."""
        return self.slug if self.slug is not None else derive_chute_slug(self.model_id)


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
        model_id="Qwen/Qwen3.6-27B-TEE",
        display_name="Qwen3.6 27B (Classifier Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=1024,
        timeout_seconds=10.0,
        exclude_from_routing=True,
    ),
    "classifier_fallback2": ModelConfig(
        model_id="google/gemma-4-31B-turbo-TEE",
        display_name="Gemma 4 31B Turbo TEE (Classifier Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=1024,
        timeout_seconds=10.0,
        exclude_from_routing=True,
    ),
    "classifier_fallback3": ModelConfig(
        model_id="XiaomiMiMo/MiMo-V2-Flash-TEE",
        display_name="MiMo V2 Flash TEE (Classifier Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=0,
        max_tokens=1024,
        timeout_seconds=10.0,
        exclude_from_routing=True,
    ),
    # ── General Text ────────────────────────────────────────────────────
    # All chat-tier entries below default to DEFAULT_CHAT_MAX_TOKENS so the
    # token budget never artificially clips an answer. Reasoning models
    # consume a chunk of this budget on `reasoning_content` before any
    # `content` is produced — see README.md "max_tokens & capacity".
    "general": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6 (General)",
        task_types=[TaskType.GENERAL_TEXT, TaskType.UNKNOWN],
        priority=2,
        timeout_seconds=60.0,
    ),
    # Qwen 3.6 27B sits one rung below K2.6 in the general_text chain. It's
    # already proven in the vision_fallback path, supports tools, has solid
    # instruction-following, and crucially uses *different* upstream chute
    # capacity than the Moonshot/Kimi pool — so when the wide-saturation
    # event Algowary hit on 2026-05-04 happens (every K2.6/K2.5/MiMo upstream
    # 429s simultaneously) we have a non-correlated next-hop instead of
    # walking the rest of the chain looking for a free slot.
    "general_fallback_qwen36": ModelConfig(
        model_id="Qwen/Qwen3.6-27B-TEE",
        display_name="Qwen3.6 27B (General Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=3,
        timeout_seconds=60.0,
    ),
    # Priorities of the two pre-existing general fallbacks were bumped from
    # 3/4 → 12/13 to keep relative order while inserting Qwen3.6 ahead of
    # them. 12/13 are unused slots; nothing else depends on these specific
    # numbers, only on relative ordering inside the same task_types tier.
    "general_fallback": ModelConfig(
        model_id="XiaomiMiMo/MiMo-V2-Flash-TEE",
        display_name="MiMo V2 Flash TEE",
        task_types=[TaskType.GENERAL_TEXT],
        priority=12,
        timeout_seconds=30.0,
    ),
    "general_fallback2": ModelConfig(
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        display_name="Qwen3 Next 80B (General Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=13,
        timeout_seconds=60.0,
    ),
    # 2026-05-05: Discord saturation events keep happening. Extending the
    # general_text chain with three more cross-pool fallbacks so we stop
    # bottoming out on K2.5 → DeepSeek-TNG-R1T2-Chimera and instead have
    # specifically-tool-capable, fast, well-tested chutes available.
    # Priorities 14/15/16 (free slots — see comment on general_fallback above).
    "general_fallback_qwen3_32b": ModelConfig(
        model_id="Qwen/Qwen3-32B-TEE",
        display_name="Qwen3 32B (General Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=14,
        timeout_seconds=60.0,
    ),
    "general_fallback_gemma": ModelConfig(
        model_id="google/gemma-4-31B-turbo-TEE",
        display_name="Gemma 4 31B Turbo (General Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=15,
        timeout_seconds=30.0,
    ),
    "general_fallback_glm_46v": ModelConfig(
        model_id="zai-org/GLM-4.6V",
        display_name="GLM 4.6V (General Fallback)",
        task_types=[TaskType.GENERAL_TEXT],
        priority=16,
        # 4.6V supports vision too — registered as a general_text option
        # since vision-routing has its own dedicated chain entries.
        supports_vision=True,
        timeout_seconds=60.0,
    ),
    # ── Math Reasoning ──────────────────────────────────────────────────
    "reasoning": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6 (Math Reasoning)",
        task_types=[TaskType.MATH_REASONING],
        priority=5,
        timeout_seconds=120.0,
    ),
    # ── General Reasoning (NEW) ─────────────────────────────────────────
    "general_reasoning": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6",
        task_types=[TaskType.GENERAL_REASONING],
        priority=6,
        timeout_seconds=120.0,
    ),
    "general_reasoning_fallback": ModelConfig(
        model_id="zai-org/GLM-5.1-TEE",
        display_name="GLM 5.1 (Reasoning Fallback)",
        task_types=[TaskType.GENERAL_REASONING, TaskType.MATH_REASONING],
        priority=7,
        timeout_seconds=90.0,
    ),
    "general_reasoning_fallback2": ModelConfig(
        model_id="MiniMaxAI/MiniMax-M2.5-TEE",
        display_name="MiniMax M2.5 (General Reasoning Fallback)",
        task_types=[TaskType.GENERAL_REASONING],
        priority=8,
        timeout_seconds=90.0,
    ),
    # ── Programming ─────────────────────────────────────────────────────
    "programming": ModelConfig(
        model_id="zai-org/GLM-5.1-TEE",
        display_name="GLM 5.1 (Programming)",
        task_types=[TaskType.PROGRAMMING],
        priority=9,
        timeout_seconds=90.0,
    ),
    "programming_fallback": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6 (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=10,
        timeout_seconds=90.0,
    ),
    "programming_fallback2": ModelConfig(
        model_id="MiniMaxAI/MiniMax-M2.5-TEE",
        display_name="MiniMax M2.5 (Programming Fallback)",
        task_types=[TaskType.PROGRAMMING],
        priority=11,
        timeout_seconds=90.0,
    ),
    # ── Creative ────────────────────────────────────────────────────────
    "creative": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6 (Creative)",
        task_types=[TaskType.CREATIVE],
        priority=14,
        timeout_seconds=90.0,
    ),
    "creative_fallback": ModelConfig(
        model_id="tngtech/DeepSeek-TNG-R1T2-Chimera",
        display_name="TNG R1T2 Chimera (Creative Fallback)",
        task_types=[TaskType.CREATIVE],
        priority=50,
        timeout_seconds=90.0,
    ),
    # ── Vision ──────────────────────────────────────────────────────────
    "vision": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6",
        task_types=[TaskType.VISION],
        priority=15,
        supports_vision=True,
        timeout_seconds=90.0,
    ),
    "vision_fallback_qwen36": ModelConfig(
        model_id="Qwen/Qwen3.6-27B-TEE",
        display_name="Qwen3.6 27B (Vision Fallback)",
        task_types=[TaskType.VISION],
        priority=16,
        supports_vision=True,
        timeout_seconds=90.0,
    ),
    "vision_fallback_gemma": ModelConfig(
        model_id="google/gemma-4-31B-turbo-TEE",
        display_name="Gemma 4 31B Turbo TEE (Vision Fallback)",
        task_types=[TaskType.VISION],
        priority=17,
        supports_vision=True,
        timeout_seconds=90.0,
    ),
    # ── Universal Fallback (ALL task types including vision) ─────────────
    "universal_fallback": ModelConfig(
        model_id="moonshotai/Kimi-K2.6-TEE",
        display_name="Kimi K2.6 (Universal Fallback)",
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
        supports_vision=True,
        timeout_seconds=120.0,
    ),
    # ── Legacy Kimi K2.5 (secondary universal fallback) ──────────────────
    "universal_fallback_legacy": ModelConfig(
        model_id="moonshotai/Kimi-K2.5-TEE",
        display_name="Kimi K2.5 (Legacy Fallback)",
        task_types=[
            TaskType.GENERAL_TEXT,
            TaskType.MATH_REASONING,
            TaskType.GENERAL_REASONING,
            TaskType.PROGRAMMING,
            TaskType.CREATIVE,
            TaskType.VISION,
            TaskType.UNKNOWN,
        ],
        priority=100,
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
    matching_configs = [
        config for config in MODEL_REGISTRY.values() if config.model_id == primary_model_id
    ]
    if not matching_configs:
        return [MODEL_REGISTRY["general"]]

    # Vision filter is driven by the *task*, not by the primary. A model can
    # appear in the registry under multiple keys (e.g. Kimi K2.5 shows up as
    # both `general_reasoning` and `vision`), so picking vision support from
    # whichever key sorts first would be fragile.
    vision_context = task_type == TaskType.VISION

    same_task: list[ModelConfig] = []
    other: list[ModelConfig] = []

    for config in sorted(MODEL_REGISTRY.values(), key=lambda config: config.priority):
        if config.exclude_from_routing:
            continue
        if config.model_id == primary_model_id:
            continue
        # Vision task → only consider vision-capable models
        if vision_context and not config.supports_vision:
            continue
        # Non-vision task → skip vision-ONLY models (but allow universal fallbacks
        # that support both vision and non-vision task types)
        if not vision_context:
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

    # Cap chain length so a saturation event doesn't take forever to
    # surface as 503 — but high enough that we exhaust real options
    # first. Bumped from 5 → 8 on 2026-05-05 after adding three more
    # general_text fallbacks (Qwen3-32B, gemma-4-31B-turbo, GLM-4.6V).
    # With per-model timeouts of 30-120s and the bulk of the chain being
    # 30-60s entries, an exhausted-chain 503 still arrives well under
    # 10 minutes even in the worst case.
    return result[:8]


def get_general_text_chain() -> list[ModelConfig]:
    """The full primary → fallback chain for `general_text` tasks.

    Used by the chutes-frontend chat-title generator (and any other client
    that wants a canonical "small ordered list of general-purpose chat
    models that all support tools"). Single source of truth — the alternative
    is hardcoding a list in every consumer and watching it drift.

    Returns primary first, then same-task fallbacks, then cross-task
    fallbacks. Already deduped by model_id by `get_fallback_models`. The
    result is bounded (currently ≤ 9 models = primary + up to 8
    fallbacks) and cheap to recompute.
    """
    primary = get_model_for_task(TaskType.GENERAL_TEXT)
    fallbacks = get_fallback_models(primary.model_id, TaskType.GENERAL_TEXT)
    return [primary] + fallbacks
