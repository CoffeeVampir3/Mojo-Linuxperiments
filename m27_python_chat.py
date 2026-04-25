import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import mojo.importer as mojo_importer

# Mojo's importer is appended behind Python's normal PathFinder by default.
# For a directory-backed Mojo package like python_glue/__init__.mojo, PathFinder
# would otherwise create an empty namespace package before Mojo sees it.
sys.meta_path.insert(0, mojo_importer.MojoImporter())
import python_glue


SYSTEM_PROMPT = (
    "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
)


def main() -> None:
    session = python_glue.M27Session(SYSTEM_PROMPT)
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
        session.start_turn(user_text)
        while True:
            chunk = session.next_chunk()
            if chunk is None:
                break
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
