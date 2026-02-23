"""LLM-based task classifier for routing decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from .models import MODEL_REGISTRY, TaskType

CLASSIFICATION_PROMPT = """You are a request classifier and quick-answer assistant. Analyze the user's request and determine the best task type.

Available task types:
- general_text: General conversations, explanations, summaries, quick factual questions, greetings, basic Q&A, assistant interactions, task instructions (e.g., "What is 2+2?", "Hello", "Explain quantum computing", "Do X for me", "Thanks!")
- math_reasoning: Complex math, logic puzzles, proofs, multi-step calculations (e.g., "Prove that √2 is irrational", "Solve this differential equation")
- general_reasoning: Multi-step logical reasoning, analysis, planning, strategy, comparisons, debates (e.g., "Compare the pros and cons of...", "What would happen if...", "Design a strategy for...")
- programming: Code generation, debugging, code review, technical implementations (e.g., "Write a Python function to...", "Fix this bug", "Implement a REST API")
- creative: Fiction writing, fantasy stories, poems, roleplay of fictional characters, screenplays, worldbuilding (e.g., "Write a fantasy story about dragons", "Continue this novel chapter", "Roleplay as a medieval knight")
- vision: Requests that reference images or ask about visual content (e.g., "What's in this image?", "Describe this diagram")

Analyze the request and call the classify_task function with your decision.

IMPORTANT CLASSIFICATION RULES:
- If the request mentions "image", "picture", "photo", "screenshot", "diagram", or "visual" → vision
- If the request asks to write code, fix bugs, or implement features → programming
- If the request involves equations, proofs, or complex calculations → math_reasoning
- If the request involves multi-step analysis, comparisons, strategy, or logical reasoning → general_reasoning
- If the request asks for fiction stories, poems, or fictional character roleplay → creative
- Default to general_text for everything else (including simple questions, greetings, and standard conversations)

IMPORTANT: Do NOT classify normal assistant conversations as 'creative'. If the user is talking to an AI assistant about tasks, asking questions, giving instructions, or having a standard conversation, use 'general_text' even if the tone is casual or friendly. 'creative' is ONLY for requests that explicitly ask for fiction, stories, poems, or character roleplay.

SELF-ANSWER OPTIMIZATION:
For general_text requests where you are very confident (confidence >= 0.95) and the answer is trivially simple and factual, provide the answer directly in the direct_answer field. This saves a round-trip to another model. Examples:
- "What is 2+2?" → direct_answer: "4"
- "What's the capital of France?" → direct_answer: "The capital of France is Paris."
- "Hello" → direct_answer: "Hello! How can I help you today?"
- "Thanks!" → direct_answer: "You're welcome! Let me know if you need anything else."
Only use direct_answer for trivially simple, factual, or greeting-type requests. Leave it null for anything that benefits from a more thorough response."""

CLASSIFICATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_task",
            "description": "Classify the task type for optimal model routing",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "general_text",
                            "math_reasoning",
                            "general_reasoning",
                            "programming",
                            "creative",
                            "vision",
                        ],
                        "description": "The classified task type",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score (0-1)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of classification",
                    },
                    "direct_answer": {
                        "type": "string",
                        "description": "For trivially simple general_text with high confidence: the answer itself. Null/omitted otherwise.",
                    },
                },
                "required": ["task_type", "confidence"],
            },
        },
    }
]


@dataclass
class ClassificationResult:
    """Result of classifying a request."""

    task_type: TaskType
    confidence: float
    direct_answer: str | None = None
    classifier_prompt_tokens: int = 0
    classifier_completion_tokens: int = 0


class TaskClassifier:
    """Classifies incoming requests to determine optimal routing."""

    def __init__(self, api_key: str, api_base: str | None = None) -> None:
        self.api_key = api_key
        self.api_base = api_base or os.environ.get("UPSTREAM_API_BASE") or "https://llm.chutes.ai/v1"
        self.classifier_models = [MODEL_REGISTRY["classifier"]]
        if "classifier_fallback" in MODEL_REGISTRY:
            self.classifier_models.append(MODEL_REGISTRY["classifier_fallback"])
        if "classifier_fallback2" in MODEL_REGISTRY:
            self.classifier_models.append(MODEL_REGISTRY["classifier_fallback2"])
        self.client = httpx.AsyncClient(timeout=max(m.timeout_seconds for m in self.classifier_models))

    async def classify(
        self,
        messages: list[dict],
        has_images: bool = False,
    ) -> ClassificationResult:
        """Classify a request to determine task type."""
        if has_images:
            return ClassificationResult(TaskType.VISION, 1.0)

        user_content = self._extract_user_content(messages)
        has_system_prompt = self._has_nontrivial_system_message(messages)

        # LLM classification with fallback chain
        # (no fast-path heuristics — the classifier model handles all routing,
        # and can self-answer simple questions via direct_answer)
        for model_config in self.classifier_models:
            try:
                response = await self.client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_config.model_id,
                        "messages": [
                            {"role": "system", "content": CLASSIFICATION_PROMPT},
                            {
                                "role": "user",
                                "content": f"Classify this request:\n\n{user_content[:2000]}",
                            },
                        ],
                        "tools": CLASSIFICATION_TOOLS,
                        "tool_choice": {"type": "function", "function": {"name": "classify_task"}},
                        "max_tokens": model_config.max_tokens,
                        "temperature": 0,
                    },
                    timeout=model_config.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()

                tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                if tool_calls:
                    args = json.loads(tool_calls[0]["function"]["arguments"])
                    task_type = TaskType(args["task_type"])
                    confidence = float(args.get("confidence", 0.7))
                    direct_answer = args.get("direct_answer")
                    # Only accept direct_answer for general_text with high confidence
                    if direct_answer and (task_type != TaskType.GENERAL_TEXT or confidence < 0.95):
                        direct_answer = None
                    # Never self-answer when a system prompt is present — the
                    # real model must see the system instructions to follow them.
                    if direct_answer and has_system_prompt:
                        direct_answer = None
                    # Capture classifier token usage
                    classifier_usage = data.get("usage", {})
                    classifier_prompt_tokens = int(classifier_usage.get("prompt_tokens", 0))
                    classifier_completion_tokens = int(classifier_usage.get("completion_tokens", 0))
                    return ClassificationResult(
                        task_type,
                        confidence,
                        direct_answer,
                        classifier_prompt_tokens,
                        classifier_completion_tokens,
                    )
            except Exception:
                continue

        return ClassificationResult(TaskType.GENERAL_TEXT, 0.5)

    def _extract_user_content(self, messages: list[dict]) -> str:
        """Extract text content from user messages."""
        parts: list[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
        return " ".join(parts)

    @staticmethod
    def _has_nontrivial_system_message(messages: list[dict]) -> bool:
        """Check if the messages contain a non-trivial system message.

        When a system prompt is present, self-answering must be suppressed
        because the real model needs to see and follow the system instructions.
        A system message shorter than 20 chars (e.g. "You are helpful") is
        considered trivial and won't block self-answering.
        """
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and len(content.strip()) >= 20:
                return True
            if isinstance(content, list):
                text = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                if len(text.strip()) >= 20:
                    return True
        return False

    async def close(self) -> None:
        await self.client.aclose()
