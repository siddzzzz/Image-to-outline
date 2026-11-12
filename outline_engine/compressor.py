"""
High-Performance Vector Path Minification & Binary ZigZag Varint Compression.

Provides two levels of database payload optimization:
1. Relative SVG Delta Minification (m, l, c, z): Standard SVG path string that
   shrinks 3-digit absolute coordinates into 1-2 digit relative deltas (~25-40% smaller).
2. Binary ZigZag Varint Packing: Encodes command stream and signed (dx, dy) deltas
   into a compact binary/base64 format (~60-75% smaller than ASCII SVG).
"""

import base64
import re
import struct
from typing import List, Tuple


class PathCompressor:
    """
    Compresses and decompresses SVG Cubic Bézier path strings (M, L, C, Z)
    using relative delta encoding and binary ZigZag varint packing.
    """

    _TOKEN_RE = re.compile(r"[MLCZmlcz]|-?\d+(?:\.\d+)?")

    @classmethod
    def parse_absolute_commands(cls, d: str) -> List[Tuple[str, List[int]]]:
        """Parses an absolute SVG path string into a list of (cmd, [int_args])."""
        if not d:
            return []

        tokens = cls._TOKEN_RE.findall(d)
        cmds: List[Tuple[str, List[int]]] = []
        i = 0
        n = len(tokens)

        while i < n:
            t = tokens[i]
            if t == "Z" or t == "z":
                cmds.append(("Z", []))
                i += 1
            elif t in ("M", "L"):
                if i + 2 < n:
                    cmds.append((t, [int(round(float(tokens[i + 1]))), int(round(float(tokens[i + 2])))]))
                    i += 3
                else:
                    break
            elif t == "C":
                if i + 6 < n:
                    pts = [int(round(float(tokens[i + 1 + k]))) for k in range(6)]
                    cmds.append(("C", pts))
                    i += 7
                else:
                    break
            else:
                i += 1

        return cmds

    @staticmethod
    def _fmt_pair(dx: int, dy: int) -> str:
        """Formats a coordinate pair, omitting space before a negative sign."""
        return f"{dx}{dy}" if dy < 0 else f"{dx} {dy}"

    @classmethod
    def to_relative_svg(cls, d: str) -> str:
        """
        Converts an absolute SVG path string (M, L, C, Z) into a minified
        relative SVG path string (m, l, c, z) with redundant whitespace stripped.
        """
        cmds = cls.parse_absolute_commands(d)
        if not cmds:
            return ""

        out: List[str] = []
        cx, cy = 0, 0
        sx, sy = 0, 0

        for cmd, args in cmds:
            if cmd == "M":
                dx, dy = args[0] - cx, args[1] - cy
                cx, cy = args[0], args[1]
                sx, sy = cx, cy
                if not out:
                    out.append(f"M{cls._fmt_pair(cx, cy)}")
                else:
                    out.append(f"m{cls._fmt_pair(dx, dy)}")
            elif cmd == "L":
                dx, dy = args[0] - cx, args[1] - cy
                cx, cy = args[0], args[1]
                out.append(f"l{cls._fmt_pair(dx, dy)}")
            elif cmd == "C":
                d1x, d1y = args[0] - cx, args[1] - cy
                d2x, d2y = args[2] - cx, args[3] - cy
                dx, dy = args[4] - cx, args[5] - cy
                cx, cy = args[4], args[5]
                p1 = cls._fmt_pair(d1x, d1y)
                p2 = cls._fmt_pair(d2x, d2y)
                p3 = cls._fmt_pair(dx, dy)
                s2 = p2 if p2.startswith("-") else f" {p2}"
                s3 = p3 if p3.startswith("-") else f" {p3}"
                out.append(f"c{p1}{s2}{s3}")
            elif cmd == "Z":
                cx, cy = sx, sy
                out.append("z")

        return "".join(out)

    @staticmethod
    def _write_varint(buf: bytearray, val: int) -> None:
        """ZigZag-encodes a signed integer and appends it as an LEB128 varint."""
        zz = (val << 1) if val >= 0 else ((-val << 1) - 1)
        while zz >= 0x80:
            buf.append((zz & 0x7F) | 0x80)
            zz >>= 7
        buf.append(zz)

    @staticmethod
    def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
        """Reads an LEB128 varint and ZigZag-decodes it back to a signed integer."""
        zz = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            zz |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        val = (zz >> 1) ^ -(zz & 1)
        return val, pos

    @classmethod
    def encode_binary(
        cls,
        d: str,
        width: int,
        height: int,
        stroke_width: float | None = None,
        is_centerline: bool = True,
    ) -> bytes:
        """
        Packs an SVG path string and canvas metadata into a compact binary stream
        using ZigZag-encoded varints for relative (dx, dy) deltas.
        """
        sw_tenths = int(round((stroke_width or 0.0) * 10.0)) if is_centerline else 0
        sw_tenths = max(0, min(255, sw_tenths))
        flags = 1 if is_centerline else 0

        buf = bytearray(struct.pack(">BBHHB", 1, flags, width, height, sw_tenths))
        cmds = cls.parse_absolute_commands(d)

        cx, cy = 0, 0
        sx, sy = 0, 0

        for cmd, args in cmds:
            if cmd == "M":
                buf.append(0)
                cls._write_varint(buf, args[0] - cx)
                cls._write_varint(buf, args[1] - cy)
                cx, cy = args[0], args[1]
                sx, sy = cx, cy
            elif cmd == "L":
                buf.append(1)
                cls._write_varint(buf, args[0] - cx)
                cls._write_varint(buf, args[1] - cy)
                cx, cy = args[0], args[1]
            elif cmd == "C":
                buf.append(2)
                for k in range(0, 6, 2):
                    cls._write_varint(buf, args[k] - cx)
                    cls._write_varint(buf, args[k + 1] - cy)
                cx, cy = args[4], args[5]
            elif cmd == "Z":
                buf.append(3)
                cx, cy = sx, sy

        return bytes(buf)

    @classmethod
    def encode_base64(
        cls,
        d: str,
        width: int,
        height: int,
        stroke_width: float | None = None,
        is_centerline: bool = True,
    ) -> str:
        """Encodes path data to a compact Base64 string for direct JSON/DB storage."""
        raw = cls.encode_binary(
            d=d,
            width=width,
            height=height,
            stroke_width=stroke_width,
            is_centerline=is_centerline,
        )
        return base64.b64encode(raw).decode("ascii")

    @classmethod
    def decode_binary(cls, data: bytes) -> dict:
        """Decodes a compact binary vector stream back into {"w", "h", "d", "sw"}."""
        if len(data) < 7:
            raise ValueError("Invalid compact vector binary payload: too short.")

        version, flags, w, h, sw_tenths = struct.unpack(">BBHHB", data[:7])
        if version != 1:
            raise ValueError(f"Unsupported compact vector version: {version}")

        is_centerline = bool(flags & 1)
        pos = 7
        n = len(data)
        cmds: List[str] = []
        cx, cy = 0, 0
        sx, sy = 0, 0

        while pos < n:
            op = data[pos]
            pos += 1
            if op == 0:  # M
                dx, pos = cls._read_varint(data, pos)
                dy, pos = cls._read_varint(data, pos)
                cx += dx
                cy += dy
                sx, sy = cx, cy
                cmds.append(f"M{cx} {cy}")
            elif op == 1:  # L
                dx, pos = cls._read_varint(data, pos)
                dy, pos = cls._read_varint(data, pos)
                cx += dx
                cy += dy
                cmds.append(f"L{cx} {cy}")
            elif op == 2:  # C
                d1x, pos = cls._read_varint(data, pos)
                d1y, pos = cls._read_varint(data, pos)
                d2x, pos = cls._read_varint(data, pos)
                d2y, pos = cls._read_varint(data, pos)
                dx, pos = cls._read_varint(data, pos)
                dy, pos = cls._read_varint(data, pos)
                cmds.append(f"C{cx + d1x} {cy + d1y} {cx + d2x} {cy + d2y} {cx + dx} {cy + dy}")
                cx += dx
                cy += dy
            elif op == 3:  # Z
                cx, cy = sx, sy
                cmds.append("Z")
            else:
                raise ValueError(f"Unknown opcode {op} at offset {pos - 1}")

        result = {"w": w, "h": h, "d": "".join(cmds)}
        if is_centerline and sw_tenths > 0:
            result["sw"] = round(sw_tenths / 10.0, 1)
        return result

    @classmethod
    def decode_base64(cls, b64_str: str) -> dict:
        """Decodes a Base64 compact vector string back into {"w", "h", "d", "sw"}."""
        raw = base64.b64decode(b64_str.encode("ascii"))
        return cls.decode_binary(raw)
