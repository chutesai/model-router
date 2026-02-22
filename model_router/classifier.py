"""LLM-based task classifier for routing decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from .models import MODEL_REGISTRY, TaskType

CLASSIFICATION_PROMPT = """You are a request classifier and quick-answer assistant. Analyze the user's request and determine the best task type.

Available task types:
- simple_text: Quick factual questions, greetings, basic Q&A (e.g., "What is 2+2?", "Hello", "What's the capital of France?")
- general_text: Standard conversations, explanations, summaries (e.g., "Explain quantum computing", "Summarize this article")
- math_reasoning: Complex math, logic puzzles, proofs, multi-step calculations (e.g., "Prove that √2 is irrational", "Solve this differential equation")
- general_reasoning: Multi-step logical reasoning, analysis, planning, strategy, comparisons, debates (e.g., "Compare the pros and cons of...", "What would happen if...", "Design a strategy for...")
- programming: Code generation, debugging, code review, technical implementations (e.g., "Write a Python function to...", "Fix this bug", "Implement a REST API")
- creative: Stories, poems, roleplay, creative writing, fictional scenarios (e.g., "Write a story about...", "Continue this narrative", "Act as a character")
- vision: Requests that reference images or ask about visual content (e.g., "What's in this image?", "Describe this diagram")

Analyze the request and call the classify_task function with your decision.

IMPORTANT CLASSIFICATION RULES:
- If the request mentions "image", "picture", "photo", "screenshot", "diagram", or "visual" → vision
- If the request asks to write code, fix bugs, or implement features → programming
- If the request involves equations, proofs, or complex calculations → math_reasoning
- If the request involves multi-step analysis, comparisons, strategy, or logical reasoning → general_reasoning
- If the request asks for stories, roleplay, or creative content → creative
- If the request is a simple question with a short factual answer → simple_text
- Default to general_text for standard conversations

SELF-ANSWER OPTIMIZATION:
For simple_text requests where you are very confident (confidence >= 0.95) and the answer is short and factual, provide the answer directly in the direct_answer field. This saves a round-trip to another model. Examples:
- "What is 2+2?" → direct_answer: "4"
- "What's the capital of France?" → direct_answer: "The capital of France is Paris."
- "Hello" → direct_answer: "Hello! How can I help you today?"
Only use direct_answer for trivially simple, factual, or greeting-type requests. Leave it null for anything that benefits from a more capable model."""

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
                            "simple_text",
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
                        "description": "For simple_text with high confidence: the answer itself. Null/omitted otherwise.",
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
        user_lower = user_content.lower().strip()

        # Fast-path: very short messages
        if len(user_content) < 30:
            return ClassificationResult(TaskType.SIMPLE_TEXT, 0.95)

        # Fast-path: obvious programming patterns
        programming_patterns = (
            "write a", "create a", "implement", "code", "function",
            "class ", "def ", "const ", "let ", "var ",
            "import ", "from ", "require(", "export ",
            "bash", "shell", "terminal", "command", "run:",
            "echo ", "curl ", "npm ", "pip ", "git ",
        )
        if any(pattern in user_lower for pattern in programming_patterns):
            return ClassificationResult(TaskType.PROGRAMMING, 0.9)

        # Fast-path: reasoning patterns
        reasoning_patterns = (
            "compare", "pros and cons", "analyze", "evaluate",
            "what would happen", "design a strategy", "plan for",
            "trade-off", "tradeoff", "should i", "which is better",
        )
        if any(pattern in user_lower for pattern in reasoning_patterns):
            return ClassificationResult(TaskType.GENERAL_REASONING, 0.85)

        # Fast-path: short non-complex messages
        if len(user_content) < 50 and not any(
            keyword in user_lower
            for keyword in (
                "write", "create", "implement", "code", "function",
                "class", "prove", "solve", "calculate", "equation",
                "story", "roleplay", "compare", "analyze", "evaluate",
            )
        ):
            return ClassificationResult(TaskType.SIMPLE_TEXT, 0.8)

        # LLM classification with fallback chain
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
                    # Only accept direct_answer for simple_text with high confidence
                    if direct_answer and (task_type != TaskType.SIMPLE_TEXT or confidence < 0.95):
                        direct_answer = None
                    return ClassificationResult(task_type, confidence, direct_answer)
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

    async def close(self) -> None:
        await self.client.aclose()
