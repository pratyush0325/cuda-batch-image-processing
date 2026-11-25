"""Minimal PGM reader and grayscale PNG encoder; Python standard library only."""

from pathlib import Path
import struct
import zlib


def read_pgm(path):
    data = Path(path).read_bytes()
    offset = 0
    tokens = []
    for _ in range(4):
        while offset < len(data):
            if data[offset:offset + 1] == b"#":
                end = data.find(b"\n", offset)
                offset = len(data) if end < 0 else end + 1
            elif data[offset] in b" \t\r\n":
                offset += 1
            else:
                break
        begin = offset
        while offset < len(data) and data[offset] not in b" \t\r\n#":
            offset += 1
        tokens.append(data[begin:offset])
    if tokens[0] != b"P5" or tokens[3] != b"255":
        raise ValueError("Expected P5/max255 PGM: " + str(path))
    width, height = map(int, tokens[1:3])
    if data[offset:offset + 2] == b"\r\n":
        offset += 2
    elif data[offset:offset + 1] in (b" ", b"\t", b"\r", b"\n"):
        offset += 1
    else:
        raise ValueError("Missing PGM raster separator")
    pixels = data[offset:]
    if width <= 0 or height <= 0 or len(pixels) != width * height:
        raise ValueError("PGM raster size mismatch: " + str(path))
    return width, height, pixels


def png_bytes(width, height, pixels):
    if len(pixels) != width * height:
        raise ValueError("PNG pixel size mismatch")

    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content +
                struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff))

    raster = b"".join(b"\0" + pixels[y * width:(y + 1) * width]
                      for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raster)) + chunk(b"IEND", b""))
