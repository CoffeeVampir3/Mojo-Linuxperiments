from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from python_glue.session import M27Session


@export
def PyInit_python_glue() -> PythonObject:
    try:
        var module = PythonModuleBuilder("python_glue")
        _ = module.add_type[M27Session]("M27Session").def_py_init[
            M27Session.py_init
        ]().def_method[M27Session.py_send](
            "send",
            docstring="Generate one assistant response for a user message.",
        ).def_method[M27Session.py_send_with_limit](
            "send_with_limit",
            docstring="Generate one assistant response with a token limit.",
        ).def_method[M27Session.py_start_turn](
            "start_turn",
            docstring="Start a streamed assistant response for a user message.",
        ).def_method[M27Session.py_start_turn_with_limit](
            "start_turn_with_limit",
            docstring="Start a streamed assistant response with a token limit.",
        ).def_method[M27Session.py_next_chunk](
            "next_chunk",
            docstring="Return the next streamed text chunk, or None when done.",
        ).def_method[M27Session.py_reset](
            "reset",
            docstring="Reset prompt history and set a new system prompt.",
        )
        return module.finalize()
    except e:
        abort(t"failed to create python_glue module: {e}")
