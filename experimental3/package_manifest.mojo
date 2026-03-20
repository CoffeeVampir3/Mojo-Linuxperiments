from pathlib import Path
from memory import Span

from jsontools.parser import (
    Parser,
    ParseError,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
)


struct StaticSlotDiskLayout(Copyable, Writable):
    var file_idx: Int
    var file_offset: Int

    fn __init__(out self, file_idx: Int, file_offset: Int):
        self.file_idx = file_idx
        self.file_offset = file_offset


struct StaticPackageManifest(Movable):
    var model_id: String
    var files: List[String]
    var slot_layout: List[StaticSlotDiskLayout]

    fn __init__(
        out self,
        model_id: String,
        var files: List[String],
        var slot_layout: List[StaticSlotDiskLayout],
    ):
        self.model_id = model_id
        self.files = files^
        self.slot_layout = slot_layout^


fn parse_string_array(mut parser: Parser) raises ParseError -> List[String]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for files", parser.pos)
    parser.skip_whitespace()
    var items = List[String]()
    if parser.consume(RBRACKET):
        return items^
    while True:
        items.append(parser.parse_string())
        if not parser.delimited_next(RBRACKET):
            break
    return items^


fn parse_slot_layout_entry(mut parser: Parser) raises ParseError -> StaticSlotDiskLayout:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for slot layout entry", parser.pos)
    parser.skip_whitespace()
    var file_idx = parser.parse_uint()
    if not parser.delimited_next(RBRACKET):
        raise ParseError("expected file offset in slot layout entry", parser.pos)
    var file_offset = parser.parse_uint()
    parser.skip_whitespace()
    if not parser.consume(RBRACKET):
        raise ParseError("expected ']' after slot layout entry", parser.pos)
    return StaticSlotDiskLayout(file_idx, file_offset)


fn parse_slot_layout_array(mut parser: Parser) raises ParseError -> List[StaticSlotDiskLayout]:
    if not parser.consume(LBRACKET):
        raise ParseError("expected '[' for slot layout", parser.pos)
    parser.skip_whitespace()
    var items = List[StaticSlotDiskLayout]()
    if parser.consume(RBRACKET):
        return items^
    while True:
        items.append(parse_slot_layout_entry(parser))
        if not parser.delimited_next(RBRACKET):
            break
    return items^


fn parse_static_package_manifest(path: Path) -> Optional[StaticPackageManifest]:
    var bytes: List[Byte]
    try:
        bytes = path.read_bytes()
    except e:
        print("package manifest read error:", e)
        return None

    var parser = Parser(Span(bytes))

    var model_id = String("")
    var files = List[String]()
    var slot_layout = List[StaticSlotDiskLayout]()
    var got_model_id = False
    var got_files = False
    var got_slot_layout = False

    try:
        parser.skip_whitespace()
        if not parser.consume(LBRACE):
            raise ParseError("expected '{' at root", parser.pos)

        parser.skip_whitespace()
        if parser.consume(RBRACE):
            raise ParseError("empty package manifest", parser.pos)

        while True:
            var key = parser.object_key()
            if key == "model_id":
                model_id = parser.parse_string()
                got_model_id = True
            elif key == "files":
                files = parse_string_array(parser)
                got_files = True
            elif key == "slot_layout":
                slot_layout = parse_slot_layout_array(parser)
                got_slot_layout = True
            else:
                parser.skip_value()

            if not parser.delimited_next(RBRACE):
                break

        parser.skip_whitespace()
        if parser.has_more():
            raise ParseError("trailing content in package manifest", parser.pos)
    except e:
        print("package manifest parse error at", e.pos, ":", e.message)
        return None

    if not got_model_id or not got_files or not got_slot_layout:
        print("package manifest missing required fields")
        return None

    if len(files) == 0:
        print("package manifest must list at least one checkpoint file")
        return None

    return StaticPackageManifest(model_id^, files^, slot_layout^)
