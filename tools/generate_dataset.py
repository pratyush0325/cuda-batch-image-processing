#!/usr/bin/env python3
"""Generate a deterministic, explicitly synthetic batch of grayscale images."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def generate_image(width, height, seed, index):
    rng = random.Random(seed + 1009 * index)
    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    radius = rng.uniform(0.08, 0.22)
    left, top = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
    slope = rng.uniform(-0.6, 0.6)
    intercept = rng.uniform(0.25, 0.75)
    sigma = (8, 12, 16, 20)[index % 4]
    pixels = bytearray(width * height)
    for y in range(height):
        ny = y / max(1, height - 1)
        for x in range(width):
            nx = x / max(1, width - 1)
            value = 25 + 65 * nx + 30 * ny
            if (nx - cx) ** 2 + (ny - cy) ** 2 < radius ** 2:
                value += 85
            if left < nx < left + 0.25 and top < ny < top + 0.2:
                value += 55
            if abs(ny - slope * nx - intercept) < 0.012:
                value += 85
            if index % 3 == 0:
                value += 12 * ((x // max(1, width // 16)) % 2)
            value += rng.gauss(0, sigma)
            pixels[y * width + x] = max(0, min(255, round(value)))
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    return header + pixels, sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 1 <= args.count <= 100000 or not all(
            1 <= value <= 8192 for value in (args.width, args.height)):
        parser.error("count must be 1..100000 and dimensions 1..8192")
    config = dict(count=args.count, width=args.width, height=args.height,
                  seed=args.seed)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config") != config:
            parser.error("existing dataset parameters differ; use a new directory")
        records = manifest.get("images", [])
        if len(records) != args.count or len(list(args.output.glob("*.pgm"))) != args.count:
            parser.error("existing dataset count is inconsistent")
        for record in records:
            path = args.output / record["filename"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                parser.error("existing dataset is missing/changed: " + str(path))
        print(f"Reused verified synthetic dataset: {args.count} images")
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output is not empty; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.count):
        data, sigma = generate_image(args.width, args.height, args.seed, index)
        filename = f"image_{index:05d}.pgm"
        (args.output / filename).write_bytes(data)
        records.append(dict(filename=filename,
                            sha256=hashlib.sha256(data).hexdigest(),
                            noise_sigma=sigma))
    manifest = dict(dataset="synthetic geometric scenes with Gaussian noise",
                    generator="generate_dataset.py v1", config=config,
                    license="MIT", images=records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {args.count} synthetic {args.width}x{args.height} images in {args.output}")


if __name__ == "__main__":
    main()
