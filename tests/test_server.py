import unittest

from model_router.models import TaskType, get_fallback_models, get_model_for_task
from model_router.server import (
    AnthropicMessagesRequest,
    _anthropic_to_openai_messages,
    _build_self_answer_openai,
    _build_self_answer_anthropic,
    _detect_images,
    _is_empty_chat_completion,
)


class TestAnthropicConversion(unittest.TestCase):
    def test_anthropic_image_block_converts_to_openai_image_url(self) -> None:
        req = AnthropicMessagesRequest(
            model="model-router",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",  # "hello" (not a real PNG, but format is correct)
                            },
                        },
                    ],
                }
            ],
        )

        msgs = _anthropic_to_openai_messages(req)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIsInstance(msgs[0]["content"], list)
        self.assertTrue(_detect_images(msgs))

        image_parts = [p for p in msgs[0]["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        url = image_parts[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_anthropic_image_only_message_is_not_dropped(self) -> None:
        req = AnthropicMessagesRequest(
            model="model-router",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "aGVsbG8=",
                            },
                        }
                    ],
                }
            ],
        )

        msgs = _anthropic_to_openai_messages(req)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIsInstance(msgs[0]["content"], list)
        self.assertTrue(_detect_images(msgs))


class TestEmptyContentFallback(unittest.TestCase):
    def test_empty_content_without_tool_calls_is_empty(self) -> None:
        self.assertTrue(
            _is_empty_chat_completion(
                {"choices": [{"message": {"role": "assistant", "content": None}}]}
            )
        )

    def test_text_content_is_not_empty(self) -> None:
        self.assertFalse(
            _is_empty_chat_completion(
                {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
            )
        )

    def test_tool_calls_are_not_empty_even_if_content_null(self) -> None:
        self.assertFalse(
            _is_empty_chat_completion(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "noop", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        )


class TestModelRouting(unittest.TestCase):
    def test_programming_routes_to_minimax_m25(self) -> None:
        model = get_model_for_task(TaskType.PROGRAMMING)
        self.assertEqual(model.model_id, "MiniMaxAI/MiniMax-M2.5-TEE")

    def test_general_reasoning_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.GENERAL_REASONING)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.5-TEE")

    def test_vision_routes_to_qwen35(self) -> None:
        model = get_model_for_task(TaskType.VISION)
        self.assertEqual(model.model_id, "Qwen/Qwen3.5-397B-A17B-TEE")

    def test_general_routes_to_qwen3_next(self) -> None:
        model = get_model_for_task(TaskType.GENERAL_TEXT)
        self.assertEqual(model.model_id, "Qwen/Qwen3-Next-80B-A3B-Instruct")

    def test_general_fallback_prefers_qwen32(self) -> None:
        fallbacks = get_fallback_models("Qwen/Qwen3-Next-80B-A3B-Instruct", TaskType.GENERAL_TEXT)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertEqual(fallback_ids[0], "Qwen/Qwen3-32B-TEE")
        self.assertIn("XiaomiMiMo/MiMo-V2-Flash-TEE", fallback_ids)

    def test_programming_fallbacks_are_task_aware(self) -> None:
        fallbacks = get_fallback_models("MiniMaxAI/MiniMax-M2.5-TEE", TaskType.PROGRAMMING)
        fallback_ids = [f.model_id for f in fallbacks]
        # First fallback should be GLM 5 (same task type)
        self.assertEqual(fallback_ids[0], "zai-org/GLM-5-TEE")
        # Kimi K2.5 should appear as universal fallback
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)

    def test_general_reasoning_fallbacks(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.5-TEE", TaskType.GENERAL_REASONING)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertIn("zai-org/GLM-5-TEE", fallback_ids)
        self.assertIn("MiniMaxAI/MiniMax-M2.5-TEE", fallback_ids)

    def test_vision_fallbacks_include_kimi(self) -> None:
        fallbacks = get_fallback_models("Qwen/Qwen3.5-397B-A17B-TEE", TaskType.VISION)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)

    def test_universal_kimi_fallback_for_creative(self) -> None:
        fallbacks = get_fallback_models("tngtech/DeepSeek-TNG-R1T2-Chimera", TaskType.CREATIVE)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)


class TestSelfAnswer(unittest.TestCase):
    def test_openai_self_answer_structure(self) -> None:
        resp = _build_self_answer_openai("Hello!")
        self.assertEqual(resp["model"], "model-router")
        self.assertEqual(resp["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")

    def test_anthropic_self_answer_structure(self) -> None:
        resp = _build_self_answer_anthropic("Hello!", "model-router")
        self.assertEqual(resp["role"], "assistant")
        self.assertEqual(resp["content"][0]["type"], "text")
        self.assertEqual(resp["content"][0]["text"], "Hello!")
        self.assertEqual(resp["stop_reason"], "end_turn")


if __name__ == "__main__":
    unittest.main()
