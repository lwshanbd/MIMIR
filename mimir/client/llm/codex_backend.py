"""The OpenAI Codex backend, selected by ``LLM_BACKEND=codex``: a ChatGPT plan, no key.

Reads the login ``codex login`` left in ``$CODEX_HOME/auth.json`` and talks to the Codex
CLI's own endpoint (the Responses API, streamed, ``store: false``; not a published API).
A refresh re-reads the file first and writes the new pair back, so the CLI stays logged
in. Encrypted reasoning items are cached by tool-call id and replayed on the next step.
See docs/SETUP.md, "OpenAI Codex".
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from .base import ChatTurn, LLMBackend, PromptTooLongError
from .openai_compat import effort_level

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
# The Codex CLI's OAuth client. Read from the access token's own claims when present,
# so this only matters for a token that does not carry one.
_DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
# Sent as ``client_version`` on the model listing, which rejects a request without one.
# The Codex CLI's own cached value is preferred (see _client_version).
_DEFAULT_CLIENT_VERSION = "0.162.0"

# Refresh this long before the access token's own expiry, so a long streamed answer
# does not start on a token that dies halfway through the request.
_REFRESH_MARGIN_S = 300
# A reasoning model can stay silent for minutes between events on a hard step.
_READ_TIMEOUT_S = 600.0
# Used when neither the live listing nor the CLI's cache names the model's window.
_FALLBACK_WINDOW = 272_000
_DEFAULT_WINDOW_PERCENT = 95
_RAW_ITEM_CACHE_CAP = 4096
# "ultra" hands work to sub-agents of its own; MIMIR runs its own loop and sub-agents,
# so the depth ladder stops at the rung below it.
_EXCLUDED_EFFORTS = frozenset({"ultra"})


class CodexAuthError(RuntimeError):
    """No usable Codex login. The message tells the user what to run.

    The word "authentication" in every message is what keeps the retry loop
    (``streaming._is_retryable``) from re-sending a request no retry can fix.
    """


def _codex_home() -> str:
    return os.environ.get("CODEX_HOME", "").strip() or os.path.expanduser("~/.codex")


def _auth_path() -> str:
    return os.path.join(_codex_home(), "auth.json")


def _base_url() -> str:
    return (os.environ.get("MIMIR_CODEX_BASE_URL", "").strip() or _DEFAULT_BASE_URL).rstrip("/")


def _jwt_claims(token: str) -> dict:
    """The payload of a JWT, unverified — only read for ``exp`` and ``client_id``."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _expires_soon(token: str, margin: float = _REFRESH_MARGIN_S) -> bool:
    exp = _jwt_claims(token).get("exp")
    if not isinstance(exp, (int, float)):
        return False  # no expiry to read: let a 401 decide
    return exp - time.time() < margin


def _read_auth_file(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise CodexAuthError(
            f"Codex authentication not found: {path} does not exist. "
            f"Run `codex login` (with a ChatGPT account) and connect again."
        ) from None
    except (OSError, ValueError) as exc:
        raise CodexAuthError(f"Codex authentication unreadable at {path}: {exc}") from None
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        raise CodexAuthError(
            f"Codex authentication in {path} holds no ChatGPT login (an API-key login "
            f"cannot use the subscription). Run `codex login` and choose ChatGPT."
        )
    return data


def _write_auth_file(path: str, data: dict) -> None:
    """Replace *path* atomically, keeping it private to the user."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".auth.", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _CodexAuth:
    """The Codex CLI's ChatGPT login, refreshed in place when it runs out."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or _auth_path()
        self._lock = threading.Lock()
        self._tokens: dict | None = None

    def _load(self) -> dict:
        self._tokens = dict(_read_auth_file(self.path)["tokens"])
        return self._tokens

    def headers(self) -> dict[str, str]:
        """Request headers for a call made now, refreshing first if the token is due."""
        with self._lock:
            tokens = self._tokens or self._load()
            if _expires_soon(tokens["access_token"]):
                tokens = self._refresh_locked(tokens["access_token"])
            return self._headers_for(tokens)

    def refresh_after_rejection(self, rejected_token: str) -> dict[str, str]:
        """Headers after the server refused *rejected_token* (a 401)."""
        with self._lock:
            return self._headers_for(self._refresh_locked(rejected_token))

    @staticmethod
    def _headers_for(tokens: dict) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        account = tokens.get("account_id")
        if account:
            headers["chatgpt-account-id"] = account
        return headers

    def _refresh_locked(self, stale_token: str) -> dict:
        # The CLI may have rotated the pair since we read it. Its fresh token is then
        # the answer, and spending the refresh token here would log the CLI out.
        on_disk = self._load()
        if on_disk["access_token"] != stale_token and not _expires_soon(on_disk["access_token"]):
            return on_disk
        refresh_token = on_disk.get("refresh_token")
        if not refresh_token:
            raise CodexAuthError(
                "Codex authentication expired and holds no refresh token. Run `codex login`.")
        client_id = _jwt_claims(on_disk["access_token"]).get("client_id") or _DEFAULT_CLIENT_ID
        try:
            resp = httpx.post(_TOKEN_URL, timeout=30.0, json={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": "openid profile email",
            })
        except httpx.HTTPError as exc:
            # Transient: no "authentication" in the text, so the turn is retried.
            raise RuntimeError(f"Codex token refresh could not reach {_TOKEN_URL}: {exc}") from exc
        if resp.status_code != 200:
            raise CodexAuthError(
                f"Codex authentication refresh refused ({resp.status_code}): "
                f"{resp.text[:300]}. Run `codex login` again.")
        fresh = resp.json()
        data = _read_auth_file(self.path)
        tokens = data["tokens"]
        for key in ("access_token", "refresh_token", "id_token"):
            if fresh.get(key):
                tokens[key] = fresh[key]
        data["last_refresh"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        _write_auth_file(self.path, data)
        self._tokens = dict(tokens)
        logger.info("codex: access token refreshed and written back to %s", self.path)
        return self._tokens


# ── conversion ──────────────────────────────────────────────────────────────────

def _tools_to_responses(tools: list[dict]) -> list[dict]:
    """Ollama function-calling tool defs to Responses API function tools.

    Ollama: ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Responses: ``{"type": "function", "name", "description", "parameters", "strict"}``
    """
    out: list[dict] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            # MIMIR's schemas are not written for strict mode (optional fields,
            # open objects), which would reject them.
            "strict": False,
        })
    return out


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _replayable(item: dict) -> dict:
    """An output item as it may be sent back as input.

    ``status`` is output-only. Item ids are kept for reasoning items alone: with
    ``store: false`` the server can resolve nothing else by id.
    """
    out = {k: v for k, v in item.items() if k != "status"}
    if out.get("type") != "reasoning":
        out.pop("id", None)
    return out


# ── models ──────────────────────────────────────────────────────────────────────

def _client_version() -> str:
    try:
        with open(os.path.join(_codex_home(), "models_cache.json"), encoding="utf-8") as fh:
            version = json.load(fh).get("client_version")
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass
    return _DEFAULT_CLIENT_VERSION


def _cached_catalog() -> list[dict]:
    """The model list the Codex CLI last fetched, from its own cache file."""
    try:
        with open(os.path.join(_codex_home(), "models_cache.json"), encoding="utf-8") as fh:
            models = json.load(fh).get("models")
        return [m for m in models if isinstance(m, dict) and m.get("slug")]
    except Exception:
        return []


class CodexBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self._auth = _CodexAuth()
        self._catalog: list[dict] | None = None
        # tool-call-id tuple -> the response's raw output items, so a tool-use turn
        # replays with its reasoning. See the module docs.
        self._raw_items: dict[tuple, list[dict]] = {}
        # One cache key for the process: consecutive steps share their prefix, and
        # the server only reuses a prefix it can find under the same key.
        self._cache_key = str(uuid.uuid4())

    # ── models ──────────────────────────────────────────────────────────────────

    def _models(self) -> list[dict]:
        """The account's model catalog: live once per process, else the CLI's cache."""
        if self._catalog is not None:
            return self._catalog
        catalog: list[dict] = []
        try:
            resp = httpx.get(
                f"{_base_url()}/models",
                params={"client_version": _client_version()},
                headers={**self._auth.headers(), "originator": "codex_cli_rs"},
                timeout=15.0,
            )
            if resp.status_code == 200:
                body = resp.json()
                models = body.get("models") if isinstance(body, dict) else body
                catalog = [m for m in models or [] if isinstance(m, dict) and m.get("slug")]
        except Exception:
            catalog = []
        if not catalog:
            catalog = _cached_catalog()
        else:
            self._catalog = catalog  # only a live answer is final for the process
        return catalog

    def _model_info(self, model: str) -> dict:
        for m in self._models():
            if m.get("slug") == model:
                return m
        return {}

    def served_models(self) -> list[str]:
        try:
            listed = [m for m in self._models() if m.get("visibility", "list") == "list"]
            listed.sort(key=lambda m: m.get("priority") if isinstance(m.get("priority"), int) else 1 << 30)
            return [m["slug"] for m in listed]
        except Exception:
            return []

    def _fetch_context_window(self, model: str) -> int | None:
        info = self._model_info(model)
        window = info.get("context_window")
        if not isinstance(window, int) or window <= 0:
            window = _FALLBACK_WINDOW
        # The CLI itself only fills this share of the window; the rest is the
        # server's own headroom. The live listing omits the figure that the CLI's
        # cache carries, and the cache has only ever said 95.
        percent = info.get("effective_context_window_percent")
        if not (isinstance(percent, (int, float)) and 0 < percent <= 100):
            percent = _DEFAULT_WINDOW_PERCENT
        return int(window * percent / 100)

    def _effort(self, model: str, thinking: bool, options: dict) -> str | None:
        """The reasoning-effort rung for this call, or None to send no reasoning block.

        ``MIMIR_CODEX_EFFORT`` pins it. Otherwise the depth ladder maps onto the
        model's own rungs as it does for an effort-family vLLM model; "auto" (budget
        -1) takes the model's own default rather than the middle rung.
        """
        pinned = os.environ.get("MIMIR_CODEX_EFFORT", "").strip().lower()
        if pinned:
            return None if pinned == "none" else pinned
        info = self._model_info(model)
        levels = [
            lv.get("effort") for lv in info.get("supported_reasoning_levels") or []
            if isinstance(lv, dict) and lv.get("effort") and lv.get("effort") not in _EXCLUDED_EFFORTS
        ]
        if not levels:
            levels = ["low", "medium", "high"]
        if not thinking:
            return levels[0]
        budget = options.get("thinking_budget")
        if budget is None or budget < 0:
            default = info.get("default_reasoning_level")
            if default in levels:
                return default
        return effort_level(levels, budget)

    # ── message conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _tc_key(tool_calls: list) -> tuple:
        return tuple(tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id"))

    def _prepare(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split internal history into (instructions, Responses API input items).

        Pairing is already reconciled (``LLMBackend.chat``): the API rejects a
        ``function_call_output`` with no matching ``function_call``, and the converse.
        """
        system_parts: list[str] = []
        items: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")

            if role == "system":
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    system_parts.append(text)
                continue

            if role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": _as_text(msg.get("content")) or "",
                })
                continue

            if role == "assistant":
                tool_calls = [tc for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict)]
                cached = self._raw_items.get(self._tc_key(tool_calls)) if tool_calls else None
                if cached is not None:
                    items.extend(cached)
                    continue
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    items.append({
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    })
                for tc in tool_calls:
                    call_id = tc.get("id")
                    if not call_id:
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    args = fn.get("arguments")
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                    })
                continue

            # user (and any unknown role) -> plain user text
            text = msg.get("content")
            if isinstance(text, str) and text != "":
                items.append({
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                })
        return "\n\n".join(system_parts), items

    def _cache_items(self, output: list[dict]) -> None:
        ids = tuple(
            it.get("call_id") for it in output
            if it.get("type") == "function_call" and it.get("call_id")
        )
        if not ids:
            return
        if len(self._raw_items) >= _RAW_ITEM_CACHE_CAP:
            self._raw_items.clear()
        self._raw_items[ids] = [_replayable(it) for it in output]

    def _build_body(
        self, model: str, messages: list[dict], tools: list[dict],
        thinking: bool, options: dict,
    ) -> dict:
        instructions, items = self._prepare(messages)
        body: dict = {
            "model": model,
            "instructions": instructions,
            "input": items,
            "store": False,
            "stream": True,
            "prompt_cache_key": self._cache_key,
        }
        fn_tools = _tools_to_responses(tools)
        if fn_tools:
            body["tools"] = fn_tools
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True
        effort = self._effort(model, thinking, options)
        if effort:
            body["reasoning"] = {"effort": effort, "summary": "auto"}
            body["include"] = ["reasoning.encrypted_content"]
        max_tokens = options.get("max_tokens") or options.get("num_predict")
        if max_tokens:
            body["max_output_tokens"] = max(1, int(max_tokens))
        return body

    # ── chat ────────────────────────────────────────────────────────────────────

    def _open_stream(self, client: httpx.Client, body: dict) -> httpx.Response:
        """Send the request, refreshing the login once if the server refuses it."""
        def _send(auth_headers: dict) -> httpx.Response:
            request = client.build_request(
                "POST", f"{_base_url()}/responses", json=body,
                headers={
                    **auth_headers,
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "codex_cli_rs",
                    "session_id": self._cache_key,
                    "accept": "text/event-stream",
                },
            )
            return client.send(request, stream=True)

        headers = self._auth.headers()
        resp = _send(headers)
        if resp.status_code == 401:
            resp.close()
            rejected = headers["Authorization"].removeprefix("Bearer ")
            resp = _send(self._auth.refresh_after_rejection(rejected))
        if resp.status_code != 200:
            detail = resp.read().decode("utf-8", "replace")
            resp.close()
            raise _http_error(resp.status_code, detail, body.get("model", ""))
        return resp

    def _complete(self, model: str, messages: list[dict], tools: list[dict],
                  thinking: bool, streaming: bool, options: dict, turn: ChatTurn) -> None:
        body = self._build_body(model, messages, tools, thinking, options)

        # The endpoint only streams. A caller that asked for no streaming gets the
        # answer in one piece at the end instead of token by token.
        held_text: list[str] = []
        timeout = httpx.Timeout(30.0, read=_READ_TIMEOUT_S)
        with httpx.Client(timeout=timeout) as client:
            resp = self._open_stream(client, body)
            try:
                final = _consume_events(
                    resp.iter_lines(), turn.check_cancelled,
                    on_text=turn.text if streaming else held_text.append,
                    on_thinking=turn.reasoning,
                )
            finally:
                resp.close()
        turn.text("".join(held_text))

        output = [it for it in final.get("output") or [] if isinstance(it, dict)]
        self._cache_items(output)
        turn.tool_calls.extend(_tool_calls_from(output))
        turn.stopped(_finish_reason(final, bool(turn.tool_calls)))
        usage = final.get("usage") or {}
        prompt_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            turn.prompt_tokens = prompt_tokens
            details = usage.get("input_tokens_details") or {}
            logger.info("codex usage: input=%d cached=%s output=%s", prompt_tokens,
                        details.get("cached_tokens"), usage.get("output_tokens"))


def _http_error(status: int, detail: str, model: str) -> Exception:
    """The exception for a refused request, worded so the retry loop reads it right."""
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            detail = (err.get("message") if isinstance(err, dict) else None) \
                or parsed.get("detail") or detail
    except ValueError:
        pass
    detail = str(detail)[:500]
    low = detail.lower()
    if "context" in low and ("window" in low or "length" in low or "exceed" in low):
        return PromptTooLongError(f"Codex: the prompt does not fit {model!r}: {detail}")
    if status in (401, 403):
        return CodexAuthError(
            f"Codex authentication refused ({status}): {detail}. Run `codex login` again.")
    if status == 400:
        return RuntimeError(f"Codex rejected the request (invalid_request_error): {detail}")
    # 429 (usage limit, rate limit) and 5xx stay retryable.
    return RuntimeError(f"Codex request failed ({status}): {detail}")


def _consume_events(
    lines: Any,
    check_cancelled: Callable[[], None],
    *,
    on_text: Callable[[str], Any],
    on_thinking: Callable[[str], None],
) -> dict:
    """Read a Responses API event stream; return the final response object.

    Text and reasoning summaries are handed on as they arrive. Output items are
    collected from ``response.output_item.done`` as well as read from the final
    response, because the final object is the one the server may abbreviate.
    """
    items: list[dict] = []
    final: dict | None = None
    summary_parts = 0
    for line in lines:
        check_cancelled()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        etype = event.get("type")
        if etype == "response.output_text.delta":
            on_text(event.get("delta") or "")
        elif etype == "response.reasoning_summary_text.delta":
            on_thinking(event.get("delta") or "")
        elif etype == "response.reasoning_summary_part.added":
            # Each summary part is its own paragraph, headed in bold.
            if summary_parts:
                on_thinking("\n\n")
            summary_parts += 1
        elif etype == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                items.append(item)
        elif etype in ("response.completed", "response.incomplete"):
            final = event.get("response") or {}
            break
        elif etype == "response.failed":
            raise _response_failure(event.get("response") or {})
        elif etype == "error":
            raise _response_failure({"error": event})
    if final is None:
        raise RuntimeError("Codex stream ended before the response completed")
    if items:
        final = {**final, "output": items}
    return final


def _response_failure(response: dict) -> Exception:
    err = response.get("error") or {}
    code = str(err.get("code") or "")
    message = str(err.get("message") or err or "unknown error")
    if code == "context_length_exceeded":
        return PromptTooLongError(f"Codex: {message}")
    return RuntimeError(f"Codex response failed ({code or 'no code'}): {message}")


def _tool_calls_from(output: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for item in output:
        if item.get("type") != "function_call" or not item.get("name"):
            continue
        raw = item.get("arguments") or ""
        try:
            arguments = json.loads(raw) if raw else {}
        except ValueError:
            arguments = {}
        calls.append({
            "id": item.get("call_id") or item.get("id") or f"call_{len(calls)}",
            "function": {"name": item["name"], "arguments": arguments},
        })
    return calls


def _finish_reason(final: dict, has_tool_calls: bool) -> str:
    if final.get("status") == "incomplete":
        reason = (final.get("incomplete_details") or {}).get("reason")
        if reason == "content_filter":
            return "content_filter"
        return "length" if reason == "max_output_tokens" else "unknown"
    return "tool_calls" if has_tool_calls else "stop"
