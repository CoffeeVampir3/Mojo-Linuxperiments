import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "python_glue"))

from m27_mojo_bridge import DEFAULT_SYSTEM_PROMPT, M27MojoBridge


def main() -> None:
    bridge = M27MojoBridge(DEFAULT_SYSTEM_PROMPT)
    print("MiniMax-M2.7 Python bridge. Type /quit, quit, or exit to stop.")

    while True:
        try:
            user_text = input("you> ")
        except EOFError:
            print()
            break

        if user_text in {"/quit", "quit", "exit"}:
            break
        if not user_text:
            continue

        print("assistant>")
        for chunk in bridge.stream_turn(DEFAULT_SYSTEM_PROMPT, user_text):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
