"""Tool tests use temporary fixtures; they are not GPU execution evidence."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from image_utils import png_bytes, read_pgm


class ToolTests(unittest.TestCase):
    def test_png_and_pgm_binary_pixels(self):
        pixels = bytes((10, 13, 32, 35, 0, 255))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.pgm"
            path.write_bytes(b"P5\n# comment\n3 2\n255\n" + pixels)
            self.assertEqual(read_pgm(path), (3, 2, pixels))
        png = png_bytes(3, 2, pixels)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        offset, compressed = 8, b""
        while offset < len(png):
            size = struct.unpack(">I", png[offset:offset + 4])[0]
            kind = png[offset + 4:offset + 8]
            content = png[offset + 8:offset + 8 + size]
            if kind == b"IDAT":
                compressed += content
            offset += 12 + size
        self.assertEqual(zlib.decompress(compressed), b"\0" + pixels[:3] + b"\0" + pixels[3:])

    def test_generator_reproducible_and_detects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "input"
            command = [sys.executable, str(ROOT / "tools/generate_dataset.py"),
                       "--output", str(directory), "--count", "5",
                       "--width", "17", "--height", "19"]
            subprocess.run(command, check=True, capture_output=True)
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
            self.assertEqual(len(list(directory.glob("*.pgm"))), 5)
            self.assertEqual(read_pgm(directory / "image_00000.pgm")[:2], (17, 19))
            (directory / "image_00000.pgm").write_bytes(b"changed")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_report_requires_passed_cuda_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "summary.json").write_text(json.dumps({"status": "failed", "backend": "CPU"}))
            (run / "images.csv").write_text("filename\n")
            (run / "batches.csv").write_text("batch_id\n")
            command = [sys.executable, str(ROOT / "tools/build_report.py"),
                       "--run", str(run), "--input", str(run)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((run / "gallery.html").exists())


if __name__ == "__main__":
    unittest.main()
