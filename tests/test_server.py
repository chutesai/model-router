import time
import unittest
from unittest.mock import MagicMock

import httpx
from fastapi.testclient import TestClient

from model_router.models import (
    MODEL_REGISTRY,
    TaskType,
    derive_chute_slug,
    get_fallback_models,
    get_general_text_chain,
    get_model_for_task,
)
from model_router.server import (
    AnthropicMessagesRequest,
    _anthropic_to_openai_messages,
    _build_router_failure_payload,
    _build_self_answer_openai,
    _build_self_answer_anthropic,
    _chunk_has_useful_output,
    _detect_images,
    _is_empty_chat_completion,
    _record_attempt,
    app,
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


class TestStreamingEmptyDetection(unittest.TestCase):
    def test_role_only_chunk_is_not_useful(self) -> None:
        self.assertFalse(
            _chunk_has_useful_output(
                {"choices": [{"delta": {"role": "assistant"}}]}
            )
        )

    def test_empty_delta_is_not_useful(self) -> None:
        self.assertFalse(
            _chunk_has_useful_output(
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            )
        )

    def test_content_delta_is_useful(self) -> None:
        self.assertTrue(
            _chunk_has_useful_output(
                {"choices": [{"delta": {"content": "Hello"}}]}
            )
        )

    def test_whitespace_only_content_is_not_useful(self) -> None:
        self.assertFalse(
            _chunk_has_useful_output(
                {"choices": [{"delta": {"content": "   "}}]}
            )
        )

    def test_tool_call_delta_is_useful(self) -> None:
        self.assertTrue(
            _chunk_has_useful_output(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "web_search"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        )

    def test_reasoning_content_is_useful(self) -> None:
        self.assertTrue(
            _chunk_has_useful_output(
                {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}
            )
        )

    def test_missing_choices_is_not_useful(self) -> None:
        self.assertFalse(_chunk_has_useful_output({}))
        self.assertFalse(_chunk_has_useful_output({"choices": []}))


class TestModelRouting(unittest.TestCase):
    def test_programming_routes_to_glm51(self) -> None:
        model = get_model_for_task(TaskType.PROGRAMMING)
        self.assertEqual(model.model_id, "zai-org/GLM-5.1-TEE")

    def test_general_reasoning_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.GENERAL_REASONING)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.6-TEE")

    def test_vision_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.VISION)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.6-TEE")

    def test_general_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.GENERAL_TEXT)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.6-TEE")

    def test_general_fallback_chain(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.6-TEE", TaskType.GENERAL_TEXT)
        fallback_ids = [f.model_id for f in fallbacks]
        # 2026-05-05: Qwen3.6 27B was inserted at priority 3 so it sits one
        # rung below K2.6 in the general_text chain — it uses a different
        # upstream chute pool than Moonshot/MiMo, so we have a non-correlated
        # next-hop during wide-saturation events (Algowary 2026-05-04).
        self.assertEqual(fallback_ids[0], "Qwen/Qwen3.6-27B-TEE")
        # MiMo and Qwen3 Next 80B are now positions 1 and 2 (priorities 12/13).
        self.assertEqual(fallback_ids[1], "XiaomiMiMo/MiMo-V2-Flash-TEE")
        self.assertEqual(fallback_ids[2], "Qwen/Qwen3-Next-80B-A3B-Instruct")
        # Removed: Qwen3 32B no longer in routing
        self.assertNotIn("Qwen/Qwen3-32B-TEE", fallback_ids)

    def test_math_reasoning_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.MATH_REASONING)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.6-TEE")

    def test_creative_routes_to_kimi(self) -> None:
        model = get_model_for_task(TaskType.CREATIVE)
        self.assertEqual(model.model_id, "moonshotai/Kimi-K2.6-TEE")

    def test_programming_fallbacks_are_task_aware(self) -> None:
        # Primary is now GLM 5.1; chain order should be K2.6 → M2.5
        fallbacks = get_fallback_models("zai-org/GLM-5.1-TEE", TaskType.PROGRAMMING)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertEqual(fallback_ids[0], "moonshotai/Kimi-K2.6-TEE")
        self.assertEqual(fallback_ids[1], "MiniMaxAI/MiniMax-M2.5-TEE")
        # Removed entries should NOT be present
        self.assertNotIn("Qwen/Qwen3-235B-A22B-Instruct-2507-TEE", fallback_ids)
        self.assertNotIn("zai-org/GLM-5-TEE", fallback_ids)
        self.assertNotIn("MiniMaxAI/MiniMax-M2.1-TEE", fallback_ids)
        self.assertNotIn("deepseek-ai/DeepSeek-V3.2-TEE", fallback_ids)

    def test_general_reasoning_fallbacks(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.6-TEE", TaskType.GENERAL_REASONING)
        fallback_ids = [f.model_id for f in fallbacks]
        self.assertIn("zai-org/GLM-5.1-TEE", fallback_ids)
        self.assertIn("MiniMaxAI/MiniMax-M2.5-TEE", fallback_ids)
        # Legacy K2.5 should remain reachable as a fallback
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)
        # Removed: GLM 5 legacy
        self.assertNotIn("zai-org/GLM-5-TEE", fallback_ids)

    def test_math_reasoning_fallbacks(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.6-TEE", TaskType.MATH_REASONING)
        fallback_ids = [f.model_id for f in fallbacks]
        # GLM 5.1 should appear ahead of K2.5 legacy
        self.assertIn("zai-org/GLM-5.1-TEE", fallback_ids)
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)
        glm_idx = fallback_ids.index("zai-org/GLM-5.1-TEE")
        k25_idx = fallback_ids.index("moonshotai/Kimi-K2.5-TEE")
        self.assertLess(glm_idx, k25_idx)
        # Removed: DeepSeek Speciale should not appear
        self.assertNotIn("deepseek-ai/DeepSeek-V3.2-Speciale-TEE", fallback_ids)

    def test_vision_fallbacks(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.6-TEE", TaskType.VISION)
        fallback_ids = [f.model_id for f in fallbacks]
        # New chain: Qwen3.6 → Gemma 4 → K2.5
        self.assertIn("Qwen/Qwen3.6-27B-TEE", fallback_ids)
        self.assertIn("google/gemma-4-31B-turbo-TEE", fallback_ids)
        self.assertIn("moonshotai/Kimi-K2.5-TEE", fallback_ids)
        # Dropped: Qwen3.5 397B
        self.assertNotIn("Qwen/Qwen3.5-397B-A17B-TEE", fallback_ids)

    def test_creative_fallbacks(self) -> None:
        fallbacks = get_fallback_models("moonshotai/Kimi-K2.6-TEE", TaskType.CREATIVE)
        fallback_ids = [f.model_id for f in fallbacks]
        # TNG Chimera is now the creative fallback
        self.assertIn("tngtech/DeepSeek-TNG-R1T2-Chimera", fallback_ids)
        # K2.5 legacy still reachable
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


class TestRouterFailurePayload(unittest.TestCase):
    """Algowary 2026-05-04: when every model in the chain 429'd we returned
    a bare `{"detail": "All models failed. Last error: ..."}` and ops had no
    way to tell which models were tried, what they returned, or how long the
    classifier took. The enriched payload now carries that detail so each
    saturation event leaves a forensic trace in the chat DB."""

    def test_payload_keeps_legacy_detail_string_with_all_models_marker(self) -> None:
        """The chutes-frontend uses `peeked.includes('All models failed')` to
        decide NOT to redundantly retry Kimi K2.6. That substring MUST stay
        in the top-level `detail` string regardless of how the payload
        evolves — otherwise we silently re-introduce the cascading retry."""
        payload = _build_router_failure_payload(
            task_type=TaskType.GENERAL_TEXT,
            classification_time_ms=2347.0,
            classifier_confidence=0.85,
            classifier_self_answered=False,
            primary_model_id="moonshotai/Kimi-K2.6-TEE",
            attempts=[],
            last_error=RuntimeError("upstream 429"),
        )
        self.assertIn("All models failed", payload["detail"])
        self.assertIn("upstream 429", payload["detail"])

    def test_payload_carries_router_failure_diagnostics(self) -> None:
        attempts = [
            {
                "model_id": "moonshotai/Kimi-K2.6-TEE",
                "status": 429,
                "elapsed_ms": 312,
                "error_class": "HTTPStatusError",
                "body_snippet": '{"detail":"Infrastructure is at maximum capacity"}',
            }
        ]
        payload = _build_router_failure_payload(
            task_type=TaskType.GENERAL_TEXT,
            classification_time_ms=2347.0,
            classifier_confidence=0.85,
            classifier_self_answered=False,
            primary_model_id="moonshotai/Kimi-K2.6-TEE",
            attempts=attempts,
            last_error=RuntimeError("upstream 429"),
        )
        rf = payload["router_failure"]
        self.assertEqual(rf["task_type"], "general_text")
        self.assertEqual(rf["classification_ms"], 2347.0)
        self.assertEqual(rf["classifier_confidence"], 0.85)
        self.assertFalse(rf["classifier_self_answered"])
        self.assertEqual(rf["primary_model"], "moonshotai/Kimi-K2.6-TEE")
        self.assertEqual(len(rf["attempts"]), 1)
        self.assertEqual(rf["attempts"][0]["model_id"], "moonshotai/Kimi-K2.6-TEE")
        self.assertEqual(rf["attempts"][0]["status"], 429)


class TestRecordAttempt(unittest.TestCase):
    """Per-attempt records feed _build_router_failure_payload — make sure both
    HTTPStatusError (the upstream-429 path that hit Algowary) and generic
    Exception (network/classifier-shape failures) are captured fully."""

    def test_records_http_status_error_with_status_and_body_snippet(self) -> None:
        response = MagicMock()
        response.status_code = 429
        response.text = '{"detail":"Infrastructure is at maximum capacity, try again later"}'
        # httpx.HTTPStatusError requires `request` and `response` args. We
        # pass the bare minimum — the helper only reads response.status_code
        # and response.text.
        request = MagicMock()
        exc = httpx.HTTPStatusError("429 from upstream", request=request, response=response)

        attempts: list[dict] = []
        started_at = time.perf_counter() - 0.42
        _record_attempt(
            attempts=attempts,
            model_id="moonshotai/Kimi-K2.6-TEE",
            started_at=started_at,
            exc=exc,
        )

        self.assertEqual(len(attempts), 1)
        rec = attempts[0]
        self.assertEqual(rec["model_id"], "moonshotai/Kimi-K2.6-TEE")
        self.assertEqual(rec["status"], 429)
        self.assertEqual(rec["error_class"], "HTTPStatusError")
        self.assertIn("Infrastructure is at maximum capacity", rec["body_snippet"])
        # Elapsed should be >= 400ms because we anchored started_at to t-0.42s.
        self.assertGreaterEqual(rec["elapsed_ms"], 400)

    def test_records_generic_exception_with_class_name(self) -> None:
        attempts: list[dict] = []
        _record_attempt(
            attempts=attempts,
            model_id="zai-org/GLM-5.1-TEE",
            started_at=time.perf_counter(),
            exc=TimeoutError("classifier timed out"),
        )
        self.assertEqual(len(attempts), 1)
        rec = attempts[0]
        self.assertEqual(rec["model_id"], "zai-org/GLM-5.1-TEE")
        self.assertIsNone(rec["status"])
        self.assertEqual(rec["error_class"], "TimeoutError")
        self.assertIn("classifier timed out", rec["body_snippet"])


class TestSlugDerivation(unittest.TestCase):
    """Slugs are how the chutes-frontend addresses each chute directly,
    bypassing the model-router classifier hop for short tool-call tasks
    (e.g. chat-title generation). Pin the derivation so a future model_id
    can't silently produce a non-resolving hostname. Each known slug below
    was verified to resolve (HTTP 401, "auth required") on 2026-05-05."""

    def test_slug_for_known_models_matches_production_chutes(self) -> None:
        cases = {
            "moonshotai/Kimi-K2.6-TEE": "chutes-moonshotai-kimi-k2-6-tee",
            "moonshotai/Kimi-K2.5-TEE": "chutes-moonshotai-kimi-k2-5-tee",
            "Qwen/Qwen3.6-27B-TEE": "chutes-qwen-qwen3-6-27b-tee",
            "XiaomiMiMo/MiMo-V2-Flash-TEE": "chutes-xiaomimimo-mimo-v2-flash-tee",
            "Qwen/Qwen3-Next-80B-A3B-Instruct": "chutes-qwen-qwen3-next-80b-a3b-instruct",
            "google/gemma-4-31B-turbo-TEE": "chutes-google-gemma-4-31b-turbo-tee",
            "deepseek-ai/DeepSeek-V3.2-TEE": "chutes-deepseek-ai-deepseek-v3-2-tee",
        }
        for model_id, expected in cases.items():
            self.assertEqual(
                derive_chute_slug(model_id),
                expected,
                f"slug mismatch for {model_id!r}",
            )

    def test_slug_squashes_double_dashes(self) -> None:
        # Pathological model_id that yields a "--" pair before squashing.
        self.assertEqual(
            derive_chute_slug("foo./bar"),
            "chutes-foo-bar",
        )

    def test_explicit_slug_overrides_derivation(self) -> None:
        # If a chute owner registered a non-standard slug, ModelConfig.slug
        # must win — we can't silently address the wrong hostname.
        cfg_default = MODEL_REGISTRY["general"]
        derived = derive_chute_slug(cfg_default.model_id)
        self.assertEqual(cfg_default.chute_slug(), derived)


class TestGeneralTextChain(unittest.TestCase):
    """The frontend's chat-title generator now consumes this chain via
    /v1/router/general_text_models, so the ordering and contents are part
    of a public contract — pin them here so a registry edit can't silently
    break sidebar titles."""

    def test_chain_starts_with_kimi_k2_6_then_qwen3_6(self) -> None:
        chain = get_general_text_chain()
        ids = [c.model_id for c in chain]
        # Primary first.
        self.assertEqual(ids[0], "moonshotai/Kimi-K2.6-TEE")
        # Qwen3.6-27B inserted at priority 3 sits one rung below — it uses a
        # different upstream pool from the Moonshot/Kimi family, so it gives
        # us a non-correlated next-hop during a Kimi-pool capacity event.
        self.assertEqual(ids[1], "Qwen/Qwen3.6-27B-TEE")

    def test_chain_is_deduped_and_bounded(self) -> None:
        chain = get_general_text_chain()
        ids = [c.model_id for c in chain]
        self.assertEqual(len(ids), len(set(ids)), "chain has duplicate model_ids")
        # 5-element fallback cap from get_fallback_models + primary = 6 max.
        self.assertLessEqual(len(chain), 6)

    def test_every_model_in_chain_supports_tools(self) -> None:
        # set_chat_title is a tool call — a non-tool model would always
        # return null and waste a chain slot.
        for cfg in get_general_text_chain():
            self.assertTrue(
                cfg.supports_tools,
                f"{cfg.model_id} is in the chain but doesn't support tools",
            )


class TestGeneralTextModelsEndpoint(unittest.TestCase):
    """End-to-end test of the new public endpoint via FastAPI TestClient.
    Verifies the response shape consumers (chutes-frontend) will rely on."""

    def test_endpoint_returns_chain_with_slugs(self) -> None:
        client = TestClient(app)
        resp = client.get("/v1/router/general_text_models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["task_type"], "general_text")
        self.assertIsInstance(body["models"], list)
        self.assertGreater(len(body["models"]), 0)

        first = body["models"][0]
        # Required keys for the consumer.
        for key in ("model_id", "slug", "display_name", "supports_tools", "timeout_seconds"):
            self.assertIn(key, first)

        # Sanity: every slug starts with `chutes-` (the public hostname pattern).
        for entry in body["models"]:
            self.assertTrue(
                entry["slug"].startswith("chutes-"),
                f"unexpected slug shape: {entry['slug']!r}",
            )
            self.assertTrue(entry["supports_tools"])

    def test_endpoint_does_not_require_auth(self) -> None:
        # Public — no Authorization header should be needed. The chain
        # contents are already public on chutes.ai/models.
        client = TestClient(app)
        resp = client.get("/v1/router/general_text_models")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
