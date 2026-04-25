# Python Glue Notes

This directory is a small Python-to-Mojo binding POC for MiniMax-M2.7. Python
owns the public API shape and networking surface; Mojo owns tokenizer/model
state, KV reuse, generation, and prompt-history mutation.

## Binding Model

Python calls Mojo through Mojo's native Python extension support:

- `mojo.importer` compiles `.mojo` sources into a Python extension module.
- The extension root exports a CPython initializer:

  ```mojo
  @export
  def PyInit_python_glue() -> PythonObject:
      var module = PythonModuleBuilder("python_glue")
      ...
      return module.finalize()
  ```

- `PythonModuleBuilder` exposes functions and types to Python.
- `PythonTypeBuilder` is reached through `module.add_type[T](...)`.
- Mutable Python-visible Mojo objects work by binding methods that receive
  `py_self: PythonObject` and recover the stored Mojo value with:

  ```mojo
  var self_ptr = py_self.downcast_value_ptr[Self]()
  ```

That pointer is the mechanism used by `M27Session` methods to mutate Mojo-side
model/session state across Python calls.

## Import Edge

`mojo.importer` appends its finder behind Python's normal import machinery.
For a directory-backed Mojo package such as `python_glue/__init__.mojo`, Python's
standard `PathFinder` can otherwise create an empty namespace package before
Mojo sees the source package.

The Python driver therefore installs a `MojoImporter` at the front of
`sys.meta_path` before importing `python_glue`:

```python
import mojo.importer as mojo_importer
sys.meta_path.insert(0, mojo_importer.MojoImporter())
import python_glue
```

Without this, `import python_glue` may succeed but produce a module with no
`M27Session` attribute.

## Constructor Edge

The binding layer may pass a null `kwargs` object when no keyword arguments are
provided. Calling `len(kwargs)` in that state can raise a Python-side
`ValueError: null argument to internal routine`.

For now, `M27Session.py_init` is intentionally positional-only and does not
inspect `kwargs`.

## Runtime Edge

Do not set `MOJO_ENABLE_RUNTIME=0` for this path. Python is importing and
calling a Mojo extension, so Mojo runtime initialization must be available.

## Worker Pool Edge

The bridge currently uses `IsolatedBurstPool` because that matches the remote
benchmark machine. This pool is deliberately spin-oriented for latency. Idle
workers will consume CPU unless they are explicitly parked.

`M27Session` wakes workers around model work and parks them when control returns
to Python:

- after model initialization
- after `start_turn()` finishes prefill
- before each `next_chunk()` generation step
- before returning a chunk to Python
- when a turn completes

This matters for an API server because Python may wait on network backpressure
between streamed chunks.

## Streaming Shape

The API-facing shape is:

```python
session.start_turn("Hello")
while True:
    chunk = session.next_chunk()
    if chunk is None:
        break
    yield chunk
```

`next_chunk()` returns complete text chunks or `None` when the assistant turn is
done. The Mojo side keeps generated token IDs until the turn completes so it can
update prompt history using the official MiniMax template behavior.

The blocking `send()` and `send_with_limit()` methods remain for smoke testing,
but API streaming should prefer `start_turn()` / `next_chunk()`.

## Text Chunking

Raw BPE tokens are not safe API chunks. A token can split UTF-8 bytes or a user
perceived grapheme. `StreamTextDecoder` buffers:

- incomplete UTF-8 byte sequences
- the final grapheme cluster of the current text buffer

This avoids emitting broken Unicode or partial emoji/combining sequences in the
normal streaming path.

## KV Cache Contract

`HistoryCache.kv_committed_tokens` tracks only the logical token prefix that is
physically represented in the model KV cache.

A token sampled from logits is not in KV yet. It becomes committed only after a
later `forward()` consumes that token as input and writes its K/V slot. This is
why the cache tracker is named `kv_committed_tokens` rather than just
`history_tokens`.

If the final displayed token has not been committed, the next turn will
re-prefill from the last true common prefix. That may redo one token of work,
but it avoids claiming stale or missing KV state.

## Current Scope

This is intentionally a single-session POC:

- one `M27Session`
- one active generation at a time
- no multi-conversation KV slot manager
- no Python-side HTTP server in this module

The clean next layer is a Python API wrapper with a process-global `M27Session`
and a lock around `start_turn()` / `next_chunk()`.
