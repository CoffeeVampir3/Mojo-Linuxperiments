import sys
import threading
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
)
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"
USER_MARKER = "<｜User｜>"
ASSISTANT_MARKER = "<｜Assistant｜>"


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


def render_text_completion_prompt(prompt: str) -> tuple[str, str, str, str]:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    first_user = prompt.find(USER_MARKER)
    if first_user < 0:
        raise ValueError("prompt is missing <｜User｜> marker")

    system_prompt = prompt[:first_user].strip()
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    history = _chat_preamble(system_prompt)
    pos = first_user
    latest_user = None
    assistant_prefill = ""

    while pos < len(prompt):
        if not prompt.startswith(USER_MARKER, pos):
            raise ValueError("prompt must alternate <｜User｜> and <｜Assistant｜> markers")

        user_start = pos + len(USER_MARKER)
        assistant_pos = prompt.find(ASSISTANT_MARKER, user_start)
        if assistant_pos < 0:
            raise ValueError("prompt is missing final <｜Assistant｜> marker")

        user_text = prompt[user_start:assistant_pos].strip("\n")
        after_assistant = assistant_pos + len(ASSISTANT_MARKER)
        next_user = prompt.find(USER_MARKER, after_assistant)

        if next_user < 0:
            latest_user = user_text
            assistant_prefill = prompt[after_assistant:].strip("\n")
            break

        assistant_text = prompt[after_assistant:next_user].strip("\n")
        history += _user_block(user_text)
        history += _assistant_history_block(assistant_text)
        pos = next_user

    if latest_user is None or latest_user.strip() == "":
        raise ValueError("latest user turn is empty")

    return system_prompt, history, latest_user, assistant_prefill


def _normalize_stop_strings(stop_strings: list[str] | None) -> list[str]:
    if not stop_strings:
        return []
    stops = []
    seen = set()
    for stop in stop_strings:
        if not isinstance(stop, str):
            raise ValueError("stop values must be strings")
        if stop and stop not in seen:
            stops.append(stop)
            seen.add(stop)
    return stops


class _StopFilter:
    def __init__(self, stop_strings: list[str]) -> None:
        self.stop_strings = stop_strings
        self.max_stop_len = max(len(stop) for stop in stop_strings)
        self.pending = ""
        self.text = ""
        self.stopped = False

    def _stop_index(self) -> int | None:
        first = None
        for stop in self.stop_strings:
            index = self.pending.find(stop)
            if index >= 0 and (first is None or index < first):
                first = index
        return first

    def push(self, chunk: str) -> str:
        if self.stopped:
            return ""

        self.pending += chunk
        index = self._stop_index()
        if index is not None:
            out = self.pending[:index]
            self.text += out
            self.pending = ""
            self.stopped = True
            return out

        keep = self.max_stop_len - 1
        if keep <= 0:
            out = self.pending
            self.pending = ""
            self.text += out
            return out
        if len(self.pending) <= keep:
            return ""

        out = self.pending[:-keep]
        self.pending = self.pending[-keep:]
        self.text += out
        return out

    def finish(self) -> str:
        out = self.pending
        self.pending = ""
        self.text += out
        return out

    def generated_text(self) -> str:
        return self.text + self.pending


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
        stop_strings: list[str] | None = None,
    ) -> Iterator[str]:
        system_prompt, history, user_text = render_openai_messages(messages)
        yield from self.stream_rendered(
            system_prompt, history, user_text, max_new_tokens, stop_strings
        )

    def stream_rendered(
        self,
        system_prompt: str,
        history: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_strings: list[str] | None = None,
        assistant_prefill: str = "",
    ) -> Iterator[str]:
        stops = _normalize_stop_strings(stop_strings)
        stop_filter = _StopFilter(stops) if stops else None
        generated_text = ""
        completed = False

        with self._lock:
            self.sync_history(system_prompt, history)
            if assistant_prefill:
                self._session.start_turn_with_prefix(
                    user_text, assistant_prefill, int(max_new_tokens)
                )
            else:
                self._session.start_turn_with_limit(user_text, int(max_new_tokens))
            try:
                while True:
                    chunk = self._session.next_chunk()
                    if chunk is None:
                        completed = True
                        if stop_filter is not None:
                            tail = stop_filter.finish()
                            if tail:
                                yield tail
                        break

                    text = str(chunk)
                    if stop_filter is None:
                        generated_text += text
                        yield text
                        continue

                    out = stop_filter.push(text)
                    if stop_filter.stopped:
                        generated_text = stop_filter.text
                        self._session.finish_turn_with_text(generated_text)
                        completed = True
                        if out:
                            yield out
                        return
                    if out:
                        yield out

                completed = True
            finally:
                if not completed:
                    if stop_filter is not None:
                        generated_text = stop_filter.generated_text()
                    self._session.finish_turn_with_text(generated_text)

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_strings: list[str] | None = None,
    ) -> str:
        system_prompt, history, user_text = render_openai_messages(messages)
        return self.generate_rendered(
            system_prompt, history, user_text, max_new_tokens, stop_strings
        )

    def generate_rendered(
        self,
        system_prompt: str,
        history: str,
        user_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_strings: list[str] | None = None,
        assistant_prefill: str = "",
    ) -> str:
        if stop_strings or assistant_prefill:
            return "".join(
                self.stream_rendered(
                    system_prompt,
                    history,
                    user_text,
                    max_new_tokens,
                    stop_strings,
                    assistant_prefill,
                )
            )

        with self._lock:
            self.sync_history(system_prompt, history)
            return str(self._session.send_with_limit(user_text, int(max_new_tokens)))

    def stream_completion_prompt(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_strings: list[str] | None = None,
    ) -> Iterator[str]:
        system_prompt, history, user_text, assistant_prefill = (
            render_text_completion_prompt(prompt)
        )
        yield from self.stream_rendered(
            system_prompt,
            history,
            user_text,
            max_new_tokens,
            stop_strings,
            assistant_prefill,
        )

    def generate_completion_prompt(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop_strings: list[str] | None = None,
    ) -> str:
        system_prompt, history, user_text, assistant_prefill = (
            render_text_completion_prompt(prompt)
        )
        return self.generate_rendered(
            system_prompt,
            history,
            user_text,
            max_new_tokens,
            stop_strings,
            assistant_prefill,
        )
