import unittest


from model_router.server import (
    AnthropicMessagesRequest,
    _anthropic_to_openai_messages,
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


if __name__ == "__main__":
    unittest.main()

