"""Tests for the Codex backend: request shaping, event parsing and the shared login (no network)."""

import base64
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import mimir.client.llm.codex_backend as codex_backend
from mimir.client.llm.base import ChatTurn, PromptTooLongError
from mimir.client.llm.codex_backend import (
    CodexAuthError,
    CodexBackend,
    _CodexAuth,
    _consume_events,
    _finish_reason,
    _http_error,
    _tools_to_responses,
)
from mimir.client.query_engine.streaming import _is_retryable


def _jwt(exp: float, client_id: str = "app_test") -> str:
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{part({'alg': 'none'})}.{part({'exp': exp, 'client_id': client_id})}.sig"


def _sse(*events: dict) -> list[str]:
    return [f"data: {json.dumps(e)}" for e in events]


class _CodexHome:
    """A temporary CODEX_HOME holding an auth.json, for the duration of a test."""

    def __init__(self, access_token: str, refresh_token: str = "rt-1") -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "auth.json")
        self.write(access_token, refresh_token)
        self.env = patch.dict(os.environ, {"CODEX_HOME": self.dir.name})
        self.env.start()

    def write(self, access_token: str, refresh_token: str) -> None:
        with open(self.path, "w") as fh:
            json.dump({
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {"access_token": access_token, "refresh_token": refresh_token,
                           "id_token": "id-1", "account_id": "acct-1"},
                "last_refresh": "2026-01-01T00:00:00.000000Z",
            }, fh)

    def read(self) -> dict:
        with open(self.path) as fh:
            return json.load(fh)

    def close(self) -> None:
        self.env.stop()
        self.dir.cleanup()


class _Resp:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


class ToolConversionTests(unittest.TestCase):
    def test_ollama_function_to_responses_tool(self) -> None:
        out = _tools_to_responses([{
            "type": "function",
            "function": {"name": "read_file", "description": "Read a file",
                         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
        }])
        self.assertEqual(out, [{
            "type": "function", "name": "read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            "strict": False,
        }])

    def test_malformed_tools_are_skipped(self) -> None:
        self.assertEqual(_tools_to_responses([{"type": "function"}, {}, {"function": {}}]), [])


class PrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = CodexBackend()

    def test_system_becomes_instructions(self) -> None:
        instructions, items = self.backend._prepare([
            {"role": "system", "content": "You are MIMIR."},
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ])
        self.assertEqual(instructions, "You are MIMIR.\n\nBe terse.")
        self.assertEqual(items, [{"type": "message", "role": "user",
                                  "content": [{"type": "input_text", "text": "hi"}]}])

    def test_tool_round_trip_becomes_call_and_output_items(self) -> None:
        _, items = self.backend._prepare([
            {"role": "user", "content": "read x"},
            {"role": "assistant", "content": "sure",
             "tool_calls": [{"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "x"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": {"ok": True}},
        ])
        self.assertEqual(items[1], {"type": "message", "role": "assistant",
                                    "content": [{"type": "output_text", "text": "sure"}]})
        self.assertEqual(items[2], {"type": "function_call", "call_id": "call_1",
                                    "name": "read_file", "arguments": '{"path": "x"}'})
        self.assertEqual(items[3], {"type": "function_call_output", "call_id": "call_1",
                                    "output": '{"ok": true}'})

    def test_a_tool_use_turn_replays_its_reasoning(self) -> None:
        output = [
            {"type": "reasoning", "id": "rs_1", "status": "completed",
             "summary": [], "encrypted_content": "opaque"},
            {"type": "function_call", "id": "fc_1", "status": "completed",
             "call_id": "call_1", "name": "calc", "arguments": "{}"},
        ]
        self.backend._cache_items(output)
        _, items = self.backend._prepare([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "1"},
        ])
        # The reasoning item keeps its id; the call loses it; neither keeps a status.
        self.assertEqual(items[1], {"type": "reasoning", "id": "rs_1", "summary": [],
                                    "encrypted_content": "opaque"})
        self.assertEqual(items[2], {"type": "function_call", "call_id": "call_1",
                                    "name": "calc", "arguments": "{}"})

    def test_temperature_is_never_sent(self) -> None:
        with patch.object(CodexBackend, "_model_info", return_value={}):
            body = self.backend._build_body("m", [{"role": "user", "content": "x"}], [],
                                            thinking=False, options={"temperature": 0.2})
        self.assertNotIn("temperature", body)
        self.assertFalse(body["store"])
        self.assertTrue(body["stream"])


class EffortTests(unittest.TestCase):
    INFO = {"default_reasoning_level": "medium",
            "supported_reasoning_levels": [{"effort": e} for e in
                                           ("low", "medium", "high", "xhigh", "max", "ultra")]}

    def setUp(self) -> None:
        self.backend = CodexBackend()
        self.info = patch.object(CodexBackend, "_model_info", return_value=self.INFO)
        self.info.start()
        self.addCleanup(self.info.stop)

    def test_auto_takes_the_model_default(self) -> None:
        self.assertEqual(self.backend._effort("m", True, {"thinking_budget": -1}), "medium")

    def test_off_is_the_weakest_rung(self) -> None:
        self.assertEqual(self.backend._effort("m", False, {}), "low")

    def test_deep_is_the_strongest_rung_below_ultra(self) -> None:
        self.assertEqual(self.backend._effort("m", True, {"thinking_budget": 16384}), "max")

    def test_env_pins_it(self) -> None:
        with patch.dict(os.environ, {"MIMIR_CODEX_EFFORT": "xhigh"}):
            self.assertEqual(self.backend._effort("m", False, {}), "xhigh")
        with patch.dict(os.environ, {"MIMIR_CODEX_EFFORT": "none"}):
            self.assertIsNone(self.backend._effort("m", True, {}))


class ContextWindowTests(unittest.TestCase):
    def test_effective_share_of_the_window(self) -> None:
        backend = CodexBackend()
        with patch.object(CodexBackend, "_model_info",
                          return_value={"context_window": 200_000, "effective_context_window_percent": 50}):
            self.assertEqual(backend._fetch_context_window("m"), 100_000)

    def test_listing_without_a_percent_uses_the_cli_default(self) -> None:
        backend = CodexBackend()
        with patch.object(CodexBackend, "_model_info", return_value={"context_window": 272_000}):
            self.assertEqual(backend._fetch_context_window("m"), 258_400)


class EventStreamTests(unittest.TestCase):
    def _run(self, lines, cancel=None):
        text, thinking = [], []
        check = ChatTurn(streaming=True, cancel_flag=cancel).check_cancelled
        final = _consume_events(lines, check, on_text=text.append, on_thinking=thinking.append)
        return final, "".join(text), "".join(thinking)

    def test_text_summaries_and_items(self) -> None:
        call = {"type": "function_call", "call_id": "c1", "name": "calc", "arguments": '{"e": 1}'}
        final, text, thinking = self._run(_sse(
            {"type": "response.reasoning_summary_part.added"},
            {"type": "response.reasoning_summary_text.delta", "delta": "**A**"},
            {"type": "response.reasoning_summary_part.added"},
            {"type": "response.reasoning_summary_text.delta", "delta": "**B**"},
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.output_item.done", "item": call},
            {"type": "response.completed", "response": {"status": "completed", "output": [],
                                                       "usage": {"input_tokens": 7}}},
        ))
        self.assertEqual(text, "Hello")
        self.assertEqual(thinking, "**A**\n\n**B**")
        self.assertEqual(final["output"], [call])
        self.assertEqual(codex_backend._tool_calls_from(final["output"]),
                         [{"id": "c1", "function": {"name": "calc", "arguments": {"e": 1}}}])

    def test_an_overlong_prompt_is_not_retried(self) -> None:
        with self.assertRaises(PromptTooLongError):
            self._run(_sse({"type": "response.failed", "response": {
                "error": {"code": "context_length_exceeded", "message": "too long"}}}))

    def test_a_stream_cut_short_is_an_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ended before"):
            self._run(_sse({"type": "response.output_text.delta", "delta": "x"}))

    def test_cancel_stops_the_read(self) -> None:
        flag = threading.Event()
        flag.set()
        import asyncio
        with self.assertRaises(asyncio.CancelledError):
            self._run(_sse({"type": "response.output_text.delta", "delta": "x"}), cancel=flag)

    def test_finish_reasons(self) -> None:
        self.assertEqual(_finish_reason({"status": "completed"}, True), "tool_calls")
        self.assertEqual(_finish_reason({"status": "completed"}, False), "stop")
        self.assertEqual(_finish_reason({"status": "incomplete",
                                         "incomplete_details": {"reason": "max_output_tokens"}}, False),
                         "length")


class _StreamResp:
    """An open streamed response, as ``_open_stream`` returns it."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def close(self) -> None:
        pass


class ChatTests(unittest.TestCase):
    """``chat`` end to end, the stream faked: what reaches the callbacks and the message."""

    _EVENTS = _sse(
        {"type": "response.reasoning_summary_part.added"},
        {"type": "response.reasoning_summary_text.delta", "delta": "thinking it over"},
        {"type": "response.output_text.delta", "delta": "Let me "},
        {"type": "response.output_text.delta", "delta": "check."},
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "c1", "name": "calc", "arguments": '{"e": 2}'}},
        {"type": "response.completed", "response": {
            "status": "completed", "output": [], "usage": {"input_tokens": 1234}}},
    )

    def _chat(self, streaming: bool, tokens: list[str]) -> dict:
        backend = CodexBackend()
        with patch.object(CodexBackend, "_open_stream", return_value=_StreamResp(self._EVENTS)), \
                patch.object(CodexBackend, "_model_info", return_value={}):
            return backend.chat("m", [{"role": "user", "content": "2+2"}], [], True, streaming,
                                {}, token_callback=tokens.append)

    def test_a_streamed_turn_carries_text_reasoning_calls_and_usage(self) -> None:
        tokens: list[str] = []
        msg = self._chat(True, tokens)
        self.assertEqual(tokens, ["Let me ", "check."])
        self.assertEqual(msg["content"], "Let me check.")
        self.assertEqual(msg["thinking"], "thinking it over")
        self.assertEqual(msg["tool_calls"],
                         [{"id": "c1", "function": {"name": "calc", "arguments": {"e": 2}}}])
        self.assertEqual(msg["finish_reason"], "tool_calls")
        self.assertEqual(msg["prompt_tokens"], 1234)

    def test_without_streaming_the_text_arrives_in_one_piece(self) -> None:
        tokens: list[str] = []
        msg = self._chat(False, tokens)
        self.assertEqual(tokens, ["Let me check."])
        self.assertEqual(msg["content"], "Let me check.")


class HttpErrorTests(unittest.TestCase):
    def test_unknown_model_is_not_retried(self) -> None:
        exc = _http_error(400, json.dumps({"detail": "The 'x' model is not supported"}), "x")
        self.assertIn("not supported", str(exc))
        self.assertFalse(_is_retryable(exc))

    def test_refused_login_is_not_retried(self) -> None:
        exc = _http_error(401, "{}", "m")
        self.assertIsInstance(exc, CodexAuthError)
        self.assertFalse(_is_retryable(exc))

    def test_usage_limit_and_server_errors_are_retried(self) -> None:
        self.assertTrue(_is_retryable(_http_error(429, '{"detail": "usage limit"}', "m")))
        self.assertTrue(_is_retryable(_http_error(502, "bad gateway", "m")))

    def test_context_overflow_is_prompt_too_long(self) -> None:
        exc = _http_error(400, json.dumps({"error": {"message": "exceeds the context window"}}), "m")
        self.assertIsInstance(exc, PromptTooLongError)


class AuthTests(unittest.TestCase):
    def test_headers_carry_token_and_account(self) -> None:
        home = _CodexHome(_jwt(time.time() + 3600))
        self.addCleanup(home.close)
        headers = _CodexAuth().headers()
        self.assertEqual(headers["chatgpt-account-id"], "acct-1")
        self.assertTrue(headers["Authorization"].startswith("Bearer "))

    def test_an_expiring_token_is_refreshed_and_written_back(self) -> None:
        home = _CodexHome(_jwt(time.time() + 10))
        self.addCleanup(home.close)
        new_token = _jwt(time.time() + 3600)
        with patch.object(codex_backend.httpx, "post", return_value=_Resp(200, {
                "access_token": new_token, "refresh_token": "rt-2", "id_token": "id-2"})) as post:
            headers = _CodexAuth().headers()
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["refresh_token"], "rt-1")
        self.assertEqual(sent["client_id"], "app_test")
        self.assertEqual(headers["Authorization"], f"Bearer {new_token}")
        # Written back in the CLI's layout, so the CLI keeps working on the new pair.
        saved = home.read()
        self.assertEqual(saved["tokens"]["refresh_token"], "rt-2")
        self.assertEqual(saved["tokens"]["account_id"], "acct-1")
        self.assertEqual(saved["auth_mode"], "chatgpt")
        self.assertNotEqual(saved["last_refresh"], "2026-01-01T00:00:00.000000Z")
        self.assertEqual(os.stat(home.path).st_mode & 0o777, 0o600)

    def test_a_token_the_cli_already_rotated_is_adopted_not_refreshed(self) -> None:
        stale = _jwt(time.time() + 3600)
        home = _CodexHome(stale)
        self.addCleanup(home.close)
        auth = _CodexAuth()
        auth.headers()
        rotated = _jwt(time.time() + 7200)
        home.write(rotated, "rt-cli")
        with patch.object(codex_backend.httpx, "post") as post:
            headers = auth.refresh_after_rejection(stale)
        post.assert_not_called()
        self.assertEqual(headers["Authorization"], f"Bearer {rotated}")

    def test_no_login_says_what_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as empty, patch.dict(os.environ, {"CODEX_HOME": empty}):
            with self.assertRaisesRegex(CodexAuthError, "codex login"):
                _CodexAuth().headers()

    def test_a_refused_refresh_is_not_retried(self) -> None:
        home = _CodexHome(_jwt(time.time() + 10))
        self.addCleanup(home.close)
        with patch.object(codex_backend.httpx, "post", return_value=_Resp(400, {"error": "invalid_grant"})):
            with self.assertRaises(CodexAuthError) as ctx:
                _CodexAuth().headers()
        self.assertFalse(_is_retryable(ctx.exception))


class FactoryTests(unittest.TestCase):
    def test_llm_backend_codex_selects_it(self) -> None:
        from mimir.client.llm.factory import clear_backend_cache, get_backend
        clear_backend_cache()
        self.addCleanup(clear_backend_cache)
        with patch.dict(os.environ, {"LLM_BACKEND": "codex"}):
            self.assertIsInstance(get_backend(), CodexBackend)


if __name__ == "__main__":
    unittest.main()
