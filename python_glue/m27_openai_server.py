import json
import os
import secrets
import signal
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from m27_mojo_bridge import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    M27MojoBridge,
    render_openai_messages,
    render_text_completion_prompt,
)


MODEL_ID = os.environ.get("M27_MODEL_ID", "minimax-m27-butterquant")
HOST = os.environ.get("M27_HOST", "127.0.0.1")
PORT = int(os.environ.get("M27_PORT", "33322"))
API_KEY = os.environ.get("M27_API_KEY")
ADMIN_KEY = os.environ.get("M27_ADMIN_KEY")
DISABLE_AUTH = os.environ.get("M27_DISABLE_AUTH", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _max_tokens(data: dict[str, Any]) -> int:
    value = (
        data.get("max_completion_tokens")
        or data.get("max_tokens")
        or data.get("max_new_tokens")
        or DEFAULT_MAX_NEW_TOKENS
    )
    value = int(value)
    if value <= 0:
        raise ValueError("max tokens must be positive")
    return value


def _chat_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _completion_id() -> str:
    return "cmpl-" + uuid.uuid4().hex


class M27OpenAIHandler(BaseHTTPRequestHandler):
    bridge: M27MojoBridge
    server_version = "M27OpenAI/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "healthy"})
            return
        if path == "/v1/auth/permission":
            permission = self._check_auth()
            if permission is None:
                return
            self._send_json(200, {"permission": permission})
            return
        if path == "/v1/models":
            if self._check_auth() is None:
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/completions":
            self._handle_completion()
            return
        if path == "/v1/chat/completions":
            self._handle_chat_completion()
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def _handle_completion(self) -> None:
        if self._check_auth() is None:
            return

        try:
            data = self._read_json()
            prompt = data.get("prompt")
            if isinstance(prompt, list):
                if len(prompt) != 1:
                    raise ValueError("only one prompt is supported")
                prompt = prompt[0]
            rendered = render_text_completion_prompt(prompt)
            max_new_tokens = _max_tokens(data)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if data.get("stream", False):
            self._stream_completion(rendered, max_new_tokens)
        else:
            self._complete_completion(rendered, max_new_tokens)

    def _handle_chat_completion(self) -> None:
        if self._check_auth() is None:
            return

        try:
            data = self._read_json()
            messages = data.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            rendered = render_openai_messages(messages)
            max_new_tokens = _max_tokens(data)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if data.get("stream", False):
            self._stream_chat(rendered, max_new_tokens)
        else:
            self._complete_chat(rendered, max_new_tokens)

    def _unused(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if self._check_auth() is None:
            return

        try:
            data = self._read_json()
            messages = data.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            rendered = render_openai_messages(messages)
            max_new_tokens = _max_tokens(data)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if data.get("stream", False):
            self._stream_chat(rendered, max_new_tokens)
        else:
            self._complete_chat(rendered, max_new_tokens)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _complete_completion(
        self,
        rendered: tuple[str, str, str],
        max_new_tokens: int,
    ) -> None:
        request_id = _completion_id()
        created = int(time.time())
        try:
            text = self.bridge.generate_rendered(
                *rendered, max_new_tokens=max_new_tokens
            )
        except Exception as exc:
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        self._send_json(
            200,
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _stream_completion(
        self,
        rendered: tuple[str, str, str],
        max_new_tokens: int,
    ) -> None:
        request_id = _completion_id()
        created = int(time.time())
        connected = True

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for chunk in self.bridge.stream_rendered(
                *rendered, max_new_tokens=max_new_tokens
            ):
                if connected:
                    connected = self._write_sse(
                        {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": chunk,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

            if connected:
                connected = self._write_sse(
                    {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "text": "", "finish_reason": "stop"}
                        ],
                    }
                )
            if connected:
                self._write_sse("[DONE]")
        except Exception as exc:
            if connected:
                self._write_sse({"error": {"message": str(exc)}})

    def _complete_chat(
        self,
        rendered: tuple[str, str, str],
        max_new_tokens: int,
    ) -> None:
        request_id = _chat_id()
        created = int(time.time())
        try:
            content = self.bridge.generate_rendered(
                *rendered, max_new_tokens=max_new_tokens
            )
        except Exception as exc:
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        self._send_json(
            200,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _stream_chat(
        self,
        rendered: tuple[str, str, str],
        max_new_tokens: int,
    ) -> None:
        request_id = _chat_id()
        created = int(time.time())
        connected = True

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            connected = self._write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )

            for chunk in self.bridge.stream_rendered(
                *rendered, max_new_tokens=max_new_tokens
            ):
                if connected:
                    connected = self._write_sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

            if connected:
                connected = self._write_sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                )
            if connected:
                self._write_sse("[DONE]")
        except Exception as exc:
            if connected:
                self._write_sse({"error": {"message": str(exc)}})

    def _write_sse(self, data: dict[str, Any] | str) -> bool:
        if isinstance(data, str):
            payload = data
        else:
            payload = json.dumps(data, ensure_ascii=False)
        try:
            self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "authorization, content-type, x-api-key, x-admin-key",
        )

    def _check_auth(self) -> str | None:
        if DISABLE_AUTH:
            return "admin"

        try:
            permission = self._key_permission()
        except ValueError as exc:
            self._send_json(401, {"error": {"message": str(exc)}})
            return None
        return permission

    def _key_permission(self) -> str:
        admin_key = self.headers.get("x-admin-key")
        api_key = self.headers.get("x-api-key")
        authorization = self.headers.get("authorization")

        if admin_key is not None:
            if ADMIN_KEY and secrets.compare_digest(admin_key, ADMIN_KEY):
                return "admin"
            raise ValueError("Invalid admin key")

        if api_key is not None:
            if API_KEY and secrets.compare_digest(api_key, API_KEY):
                return "api"
            if ADMIN_KEY and secrets.compare_digest(api_key, ADMIN_KEY):
                return "admin"
            raise ValueError("Invalid API key")

        if authorization is not None:
            parts = authorization.split(" ", 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise ValueError("Invalid authorization header")
            key = parts[1]
            if ADMIN_KEY and secrets.compare_digest(key, ADMIN_KEY):
                return "admin"
            if API_KEY and secrets.compare_digest(key, API_KEY):
                return "api"
            raise ValueError("Invalid bearer token")

        raise ValueError("Please provide an API key")

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    if not DISABLE_AUTH and not API_KEY and not ADMIN_KEY:
        raise SystemExit("Set M27_API_KEY or M27_ADMIN_KEY, or set M27_DISABLE_AUTH=1")

    tokenizer_path = os.environ.get("M27_TOKENIZER_PATH")
    model_dir = os.environ.get("M27_MODEL_DIR")
    system_prompt = os.environ.get("M27_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    print("Loading MiniMax-M2.7 Mojo session...", flush=True)
    M27OpenAIHandler.bridge = M27MojoBridge(
        system_prompt=system_prompt,
        tokenizer_path=tokenizer_path,
        model_dir=model_dir,
    )
    print("MiniMax-M2.7 Mojo session ready", flush=True)

    httpd = ThreadingHTTPServer((HOST, PORT), M27OpenAIHandler)
    print(
        "Serving MiniMax-M2.7 OpenAI-compatible API at http://%s:%d/v1"
        % (HOST, PORT),
        flush=True,
    )

    def stop_server(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
