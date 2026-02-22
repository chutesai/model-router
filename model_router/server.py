"""FastAPI server for composite model routing.

Supports both OpenAI Chat Completions and Anthropic Messages API formats.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Literal, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .classifier import ClassificationResult, TaskClassifier
from .metrics import metrics
from .models import ModelConfig, TaskType, get_fallback_models, get_model_for_task

app = FastAPI(title="Model Router", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: Optional[TaskClassifier] = None
api_key: str = ""
api_base: str = ""


def ensure_router_ready() -> None:
    """Ensure router globals are initialized."""
    global classifier, api_key, api_base
    if not api_base:
        api_base = (
            os.environ.get("UPSTREAM_API_BASE")
            or os.environ.get("CHUTES_API_BASE")
            or os.environ.get("CHUTES_API_URL")
            or "https://llm.chutes.ai/v1"
        )
    if not api_key:
        api_key = os.environ.get("CHUTES_API_KEY") or os.environ.get("UPSTREAM_API_KEY") or ""
    if classifier is None and api_key:
        classifier = TaskClassifier(api_key, api_base)


def _extract_client_key(
    *,
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> str:
    """Extract the caller-provided API key from common headers."""
    if isinstance(x_api_key, str) and x_api_key.strip():
        return x_api_key.strip()
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def _require_router_auth(
    *,
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> None:
    """Prevent unauthenticated access to the router."""
    keys = {
        key
        for key in (
            os.environ.get("ROUTER_API_KEY"),
            api_key,
        )
        if key
    }
    if not keys:
        raise HTTPException(status_code=503, detail="No API key configured for model-router.")
    client_key = _extract_client_key(x_api_key=x_api_key, authorization=authorization)
    if not client_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if client_key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API key")


class ChatCompletionRequest(BaseModel):
    """Minimal OpenAI-compatible chat request for routing."""

    model: str
    messages: list[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None

    model_config = ConfigDict(extra="allow")


# --- Anthropic Messages API Models ---


class AnthropicContentBlock(BaseModel):
    type: Literal["text", "image", "tool_use", "tool_result"] = "text"
    text: Optional[str] = None
    source: Optional[dict] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    tool_use_id: Optional[str] = None
    content: Optional[str | list[dict]] = None


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock | dict]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=4096)
    system: Optional[str | list[dict]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None

    model_config = ConfigDict(extra="allow")


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[dict]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# --- Format Conversion Helpers ---


def _extract_system_text(system: Optional[str | list[dict]]) -> Optional[str]:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def _anthropic_tools_to_openai(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    if not tools:
        return None
    openai_tools: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        function: dict[str, Any] = {"name": name}
        description = tool.get("description")
        if description:
            function["description"] = description
        parameters = tool.get("input_schema") or tool.get("parameters")
        if parameters:
            function["parameters"] = parameters
        openai_tools.append({"type": "function", "function": function})
    return openai_tools or None


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.lower()
        if choice == "auto":
            return "auto"
        if choice == "any":
            return "required"
        if choice == "none":
            return "none"
        return None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            name = tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
            return "required"
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "none":
            return "none"
    return None


def _coerce_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return json.dumps(content)
    return str(content)


def _anthropic_to_openai_messages(
    anthropic_request: AnthropicMessagesRequest,
) -> list[dict]:
    openai_messages = []

    system_text = _extract_system_text(anthropic_request.system)
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for msg in anthropic_request.messages:
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
            continue

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in msg.content:
            block_dict: dict[str, Any]
            if isinstance(block, AnthropicContentBlock):
                block_dict = block.model_dump()
            elif isinstance(block, dict):
                block_dict = block
            else:
                continue

            block_type = block_dict.get("type")
            if block_type == "text":
                text = block_dict.get("text") or ""
                if text:
                    content_parts.append({"type": "text", "text": text})
            elif block_type == "image":
                source = block_dict.get("source") or {}
                if isinstance(source, dict):
                    source_type = source.get("type")
                    if source_type == "base64":
                        media_type = source.get("media_type") or "image/png"
                        data = source.get("data")
                        if isinstance(data, str) and data.strip():
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                                }
                            )
                    elif source_type in {"url", "image_url"}:
                        url = source.get("url")
                        if isinstance(url, str) and url.strip():
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": url.strip()}}
                            )
            elif block_type == "tool_use" and msg.role == "assistant":
                tool_id = block_dict.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
                name = block_dict.get("name") or ""
                input_obj = block_dict.get("input") or {}
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(input_obj),
                        },
                    }
                )
            elif block_type == "tool_result" and msg.role == "user":
                tool_use_id = block_dict.get("tool_use_id") or block_dict.get("id")
                tool_results.append(
                    {
                        "tool_call_id": tool_use_id,
                        "content": _coerce_tool_content(block_dict.get("content")),
                    }
                )

        has_image_part = any(part.get("type") == "image_url" for part in content_parts)
        text_content = "\n".join(
            [
                part.get("text", "")
                for part in content_parts
                if isinstance(part, dict) and part.get("type") == "text"
            ]
        ).strip()
        openai_content: Any = content_parts if has_image_part else text_content

        if msg.role == "assistant":
            if text_content or has_image_part or tool_calls:
                openai_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": openai_content,
                }
                if tool_calls:
                    openai_message["tool_calls"] = tool_calls
                openai_messages.append(openai_message)
        else:
            if text_content or has_image_part:
                openai_messages.append({"role": msg.role, "content": openai_content})
            for tool_result in tool_results:
                if tool_result.get("tool_call_id"):
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"],
                        }
                    )

    return openai_messages


def _openai_to_anthropic_response(
    openai_response: dict,
    requested_model: str,
) -> dict:
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    finish_reason = choice.get("finish_reason")

    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = "tool_use" if tool_calls else stop_reason_map.get(finish_reason, "end_turn")

    usage = openai_response.get("usage", {})

    content_blocks: list[dict] = []
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or ""
        args = function.get("arguments") or {}
        input_obj: dict[str, Any]
        if isinstance(args, str):
            if args.strip():
                try:
                    input_obj = json.loads(args)
                except Exception:
                    input_obj = {"_raw": args}
            else:
                input_obj = {}
        elif isinstance(args, dict):
            input_obj = args
        else:
            input_obj = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
                "name": name,
                "input": input_obj,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=content_blocks,
        model=requested_model,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
    ).model_dump()


@app.on_event("startup")
async def startup() -> None:
    ensure_router_ready()


@app.on_event("shutdown")
async def shutdown() -> None:
    if classifier:
        await classifier.close()


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "model-router"}


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "model-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "chutes",
            }
        ],
    }


@app.get("/v1/router/metrics")
async def get_metrics() -> dict:
    return metrics.to_dict()


# --- Anthropic Messages API Endpoint ---


@app.post("/v1/messages")
async def anthropic_messages(
    request: AnthropicMessagesRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
    anthropic_beta: Optional[str] = Header(None, alias="anthropic-beta"),
) -> Any:
    """Anthropic Messages API compatible endpoint."""
    ensure_router_ready()
    _require_router_auth(x_api_key=x_api_key, authorization=authorization)
    start_time = time.perf_counter()
    requested_model = request.model

    openai_messages = _anthropic_to_openai_messages(request)
    has_images = _detect_images(openai_messages)

    if classifier is None:
        result = ClassificationResult(TaskType.GENERAL_TEXT, 0.5)
    else:
        result = await classifier.classify(openai_messages, has_images)

    task_type = result.task_type
    classification_time_ms = (time.perf_counter() - start_time) * 1000

    # Self-answer: classifier already provided the answer for simple requests
    if result.direct_answer:
        metrics.record_request(
            task_type=task_type.value,
            model_id="classifier-self-answer",
            classification_time_ms=classification_time_ms,
            used_fallback=False,
        )
        anthropic_response = _build_self_answer_anthropic(
            result.direct_answer, requested_model
        )
        if request.stream:
            return _stream_anthropic_self_answer(anthropic_response)
        return anthropic_response

    primary_model = get_model_for_task(task_type)
    fallbacks = get_fallback_models(primary_model.model_id, task_type)

    models_to_try = [primary_model] + fallbacks
    last_error: Exception | None = None

    for index, model_config in enumerate(models_to_try):
        used_fallback = index > 0
        try:
            if request.stream:
                response = await _anthropic_stream_response(
                    request, openai_messages, model_config
                )
            else:
                response = await _anthropic_non_stream_response(
                    request, openai_messages, model_config
                )

            metrics.record_request(
                task_type=task_type.value,
                model_id=model_config.model_id,
                classification_time_ms=classification_time_ms,
                used_fallback=used_fallback,
            )
            return response
        except httpx.HTTPStatusError as exc:
            last_error = exc
            metrics.record_error(model_config.model_id)
            status = exc.response.status_code
            if status == 429 or status >= 500:
                continue
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            last_error = exc
            metrics.record_error(model_config.model_id)
            continue

    raise HTTPException(status_code=503, detail=f"All models failed. Last error: {last_error}")


async def _anthropic_stream_response(
    anthropic_request: AnthropicMessagesRequest,
    openai_messages: list[dict],
    model_config: ModelConfig,
) -> StreamingResponse:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    openai_tools = _anthropic_tools_to_openai(anthropic_request.tools)
    openai_tool_choice = _anthropic_tool_choice_to_openai(anthropic_request.tool_choice)

    async def stream_from_message(message: dict) -> AsyncIterator[str]:
        message_start = {
            "type": "message_start",
            "message": {
                "id": message.get("id", msg_id),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": message.get("model", anthropic_request.model),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": (message.get("usage") or {}).get("input_tokens", 0),
                    "output_tokens": 0,
                },
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        content_blocks = message.get("content") or []
        for index, block in enumerate(content_blocks):
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
            if block.get("type") == "text":
                text = block.get("text") or ""
                if text:
                    content_delta = {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"
            elif block.get("type") == "tool_use":
                input_obj = block.get("input") or {}
                partial_json = json.dumps(input_obj)
                if partial_json:
                    content_delta = {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"
            content_block_stop = {"type": "content_block_stop", "index": index}
            yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

        message_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason"),
                "stop_sequence": message.get("stop_sequence"),
            },
            "usage": {
                "output_tokens": (message.get("usage") or {}).get("output_tokens", 0)
            },
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    payload: dict[str, Any] = {
        "model": model_config.model_id,
        "messages": openai_messages,
        "max_tokens": anthropic_request.max_tokens or model_config.max_tokens,
    }
    if anthropic_request.temperature is not None:
        payload["temperature"] = anthropic_request.temperature
    if anthropic_request.stop_sequences:
        payload["stop"] = anthropic_request.stop_sequences
    if openai_tools:
        payload["tools"] = openai_tools
    if openai_tool_choice is not None:
        payload["tool_choice"] = openai_tool_choice

    # If tools are present, we do a non-streaming upstream call and emit a
    # synthetic Anthropic stream. Run the call before returning so fallbacks
    # can trigger on failures/empty content.
    if openai_tools or openai_tool_choice is not None:
        async with httpx.AsyncClient(timeout=model_config.timeout_seconds) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={**payload, "stream": False},
            )
            response.raise_for_status()
            openai_data = response.json()
        if _is_empty_chat_completion(openai_data):
            raise RuntimeError(f"Upstream returned empty content for model {model_config.model_id}")
        anthropic_message = _openai_to_anthropic_response(openai_data, anthropic_request.model)

        async def stream_generator() -> AsyncIterator[str]:
            async for chunk in stream_from_message(anthropic_message):
                yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Router-Model": model_config.model_id,
            },
        )

    async def stream_generator() -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=model_config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={**payload, "stream": True},
            ) as response:
                response.raise_for_status()

                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": anthropic_request.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                content_block_start = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = (data.get("choices") or [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            content_delta = {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": content},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"
                    except json.JSONDecodeError:
                        continue

                content_block_stop = {"type": "content_block_stop", "index": 0}
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

                message_delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

                yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Router-Model": model_config.model_id,
        },
    )


async def _anthropic_non_stream_response(
    anthropic_request: AnthropicMessagesRequest,
    openai_messages: list[dict],
    model_config: ModelConfig,
) -> dict:
    payload = {
        "model": model_config.model_id,
        "messages": openai_messages,
        "max_tokens": anthropic_request.max_tokens or model_config.max_tokens,
        "stream": False,
    }
    if anthropic_request.temperature is not None:
        payload["temperature"] = anthropic_request.temperature
    if anthropic_request.stop_sequences:
        payload["stop"] = anthropic_request.stop_sequences
    openai_tools = _anthropic_tools_to_openai(anthropic_request.tools)
    openai_tool_choice = _anthropic_tool_choice_to_openai(anthropic_request.tool_choice)
    if openai_tools:
        payload["tools"] = openai_tools
    if openai_tool_choice is not None:
        payload["tool_choice"] = openai_tool_choice

    async with httpx.AsyncClient(timeout=model_config.timeout_seconds) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        openai_data = response.json()
        if _is_empty_chat_completion(openai_data):
            raise RuntimeError(f"Upstream returned empty content for model {model_config.model_id}")
        return _openai_to_anthropic_response(openai_data, anthropic_request.model)


# --- OpenAI Chat Completions Endpoint ---


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
) -> Any:
    """Classify and route chat completion requests."""
    ensure_router_ready()
    _require_router_auth(x_api_key=x_api_key, authorization=authorization)
    start_time = time.perf_counter()
    has_images = _detect_images(request.messages)

    if classifier is None:
        result = ClassificationResult(TaskType.GENERAL_TEXT, 0.5)
    else:
        result = await classifier.classify(request.messages, has_images)

    task_type = result.task_type
    classification_time_ms = (time.perf_counter() - start_time) * 1000

    # Self-answer: classifier already provided the answer for simple requests
    if result.direct_answer:
        metrics.record_request(
            task_type=task_type.value,
            model_id="classifier-self-answer",
            classification_time_ms=classification_time_ms,
            used_fallback=False,
        )
        openai_response = _build_self_answer_openai(result.direct_answer)
        if request.stream:
            return _stream_self_answer_openai(openai_response)
        return openai_response

    primary_model = get_model_for_task(task_type)
    fallbacks = get_fallback_models(primary_model.model_id, task_type)

    models_to_try = [primary_model] + fallbacks
    last_error: Exception | None = None

    for index, model_config in enumerate(models_to_try):
        used_fallback = index > 0
        try:
            if request.stream:
                response = await _stream_response(request, model_config)
            else:
                response = await _non_stream_response(request, model_config)

            metrics.record_request(
                task_type=task_type.value,
                model_id=model_config.model_id,
                classification_time_ms=classification_time_ms,
                used_fallback=used_fallback,
            )
            return response
        except httpx.HTTPStatusError as exc:
            last_error = exc
            metrics.record_error(model_config.model_id)
            status = exc.response.status_code
            if status == 429 or status >= 500:
                continue
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            last_error = exc
            metrics.record_error(model_config.model_id)
            continue

    raise HTTPException(status_code=503, detail=f"All models failed. Last error: {last_error}")


async def _stream_response(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
) -> StreamingResponse:

    async def stream_generator():
        async with httpx.AsyncClient(timeout=model_config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=_build_payload(request, model_config, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue
                        try:
                            data = json.loads(payload)
                            data["model"] = "model-router"
                            yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError:
                            yield f"{line}\n\n"
                    else:
                        yield f"{line}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Router-Model": model_config.model_id,
        },
    )


async def _non_stream_response(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
) -> dict:
    async with httpx.AsyncClient(timeout=model_config.timeout_seconds) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=_build_payload(request, model_config, stream=False),
        )
        response.raise_for_status()
        data = response.json()
        if _is_empty_chat_completion(data):
            raise RuntimeError(f"Upstream returned empty content for model {model_config.model_id}")
        data["model"] = "model-router"
        return data


def _build_payload(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
    stream: bool,
) -> dict:
    payload = request.model_dump(exclude_none=True)
    payload["model"] = model_config.model_id
    payload["stream"] = stream
    payload["max_tokens"] = request.max_tokens or model_config.max_tokens
    return payload


def _detect_images(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _is_empty_chat_completion(openai_response: dict) -> bool:
    """True if an upstream response has no usable assistant output.

    Some upstream models return `message.content: null` (and no tool calls)
    in successful responses. Treat this as a failure so we can fall back.
    """

    choices = openai_response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return True
    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return False
    content = message.get("content")
    if isinstance(content, str):
        return not content.strip()
    return True


# --- Self-Answer Helpers (classifier answers simple questions directly) ---


def _build_self_answer_openai(answer: str) -> dict:
    """Build an OpenAI-format response from a classifier self-answer."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "model-router",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_self_answer_openai(openai_response: dict) -> StreamingResponse:
    """Stream a self-answer as OpenAI SSE chunks."""
    answer = openai_response["choices"][0]["message"]["content"]
    resp_id = openai_response["id"]

    async def generator() -> AsyncIterator[str]:
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "model-router",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "model-router",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Router-Model": "classifier-self-answer",
        },
    )


def _build_self_answer_anthropic(answer: str, requested_model: str) -> dict:
    """Build an Anthropic-format response from a classifier self-answer."""
    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": answer}],
        model=requested_model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
    ).model_dump()


def _stream_anthropic_self_answer(anthropic_response: dict) -> StreamingResponse:
    """Stream a self-answer as Anthropic SSE events."""
    msg_id = anthropic_response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    answer = ""
    for block in anthropic_response.get("content", []):
        if block.get("type") == "text":
            answer = block.get("text", "")
            break

    async def generator() -> AsyncIterator[str]:
        message_start = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_response.get("model", "model-router"),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

        if answer:
            content_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": answer},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"

        content_block_stop = {"type": "content_block_stop", "index": 0}
        yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Router-Model": "classifier-self-answer",
        },
    )
