import sys
import threading
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
)
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"


def _load_python_glue():
    sys.path.insert(0, str(REPO_ROOT))

    import mojo.importer as mojo_importer

    sys.meta_path.insert(0, mojo_importer.MojoImporter())
    import python_glue

    return python_glue


_python_glue = _load_python_glue()


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _chat_preamble(system_prompt: str) -> str:
    return "]~!b[]~b]system\n" + system_prompt + "[e~[\n"


def _user_block(text: str) -> str:
    return "]~b]user\n" + text + "[e~[\n"


def _assistant_history_block(text: str) -> str:
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :].strip("\n")
    return "]~b]ai\n" + text + "[e~[\n"


def render_openai_messages(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not messages:
        raise ValueError("messages must not be empty")

    system_parts = []
    latest_user_index = -1

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("messages must contain objects")
        role = message.get("role")
        text = _message_content_text(message.get("content"))
        if role in {"system", "developer"} and text:
            system_parts.append(text)
        elif role == "user":
            latest_user_index = index

    if latest_user_index < 0:
        raise ValueError("messages must contain at least one user message")

    system_prompt = "\n\n".join(system_parts) if system_parts else DEFAULT_SYSTEM_PROMPT
    history = _chat_preamble(system_prompt)

    for message in messages[:latest_user_index]:
        role = message.get("role")
        text = _message_content_text(message.get("content"))
        if not text:
            continue
        if role == "user":
            history += _user_block(text)
        elif role == "assistant":
            history += _assistant_history_block(text)

    latest_user = _message_content_text(messages[latest_user_index].get("content"))
    if not latest_user:
        raise ValueError("latest user message must not be empty")

    return system_prompt, history, latest_user


class M27MojoBridge:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tokenizer_path: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        args = [system_prompt]
        if tokenizer_path is not None:
            args.append(tokenizer_path)
            if model_dir is not None:
                args.append(model_dir)
        elif model_dir is not None:
            args.append(DEFAULT_TOKENIZER_PATH)
            args.append(model_dir)

        self._session = _python_glue.M27Session(*args)
        self._system_prompt = system_prompt
        self._lock = threading.Lock()

    def reset_if_needed(self, system_prompt: str) -> None:
        if system_prompt != self._system_prompt:
            self._session.reset(system_prompt)
            self._system_prompt = system_prompt

    def sync_history(self, system_prompt: str, history: str) -> None:
        self._session.sync_history(system_prompt, history)
        self._system_prompt = system_prompt

    def stream_turn(
        self,
        system_prompt: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        with self._lock:
            self.reset_if_needed(system_prompt)
            self._session.start_turn_with_limit(user_text, int(max_new_tokens))
            while True:
                chunk = self._session.next_chunk()
                if chunk is None:
                    break
                yield str(chunk)

    def generate_turn(
        self,
        system_prompt: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        with self._lock:
            self.reset_if_needed(system_prompt)
            return str(self._session.send_with_limit(user_text, int(max_new_tokens)))

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        system_prompt, history, user_text = render_openai_messages(messages)
        yield from self.stream_rendered(system_prompt, history, user_text, max_new_tokens)

    def stream_rendered(
        self,
        system_prompt: str,
        history: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        with self._lock:
            self.sync_history(system_prompt, history)
            self._session.start_turn_with_limit(user_text, int(max_new_tokens))
            while True:
                chunk = self._session.next_chunk()
                if chunk is None:
                    break
                yield str(chunk)

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        system_prompt, history, user_text = render_openai_messages(messages)
        return self.generate_rendered(system_prompt, history, user_text, max_new_tokens)

    def generate_rendered(
        self,
        system_prompt: str,
        history: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        with self._lock:
            self.sync_history(system_prompt, history)
            return str(self._session.send_with_limit(user_text, int(max_new_tokens)))
