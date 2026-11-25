#!/usr/bin/env python3
"""Build a portable evidence gallery and measured description after a CUDA run."""

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

from image_utils import png_bytes, read_pgm


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    summary = json.loads((run / "summary.json").read_text())
    images = csv_rows(run / "images.csv")
    batches = csv_rows(run / "batches.csv")
    if (summary.get("status") != "passed" or summary.get("backend") != "CUDA" or
            len(images) != summary["images"] or
            summary["validated_images"] != len(images) or
            len(batches) != summary["batches"] or
            sum(int(row["images"]) for row in batches) != len(images) or
            any(int(row["edge_mismatches"]) != 0 or
                float(row["max_blur_error"]) > 0.0001 for row in images)):
        raise SystemExit("Evidence is incomplete or validation did not pass")
    filenames = [row["filename"] for row in images]
    if len(set(filenames)) != len(filenames):
        raise SystemExit("Evidence contains duplicate input filenames")
    generator = args.input / "manifest.json"
    expected_hashes = {}
    if generator.exists():
        manifest = json.loads(generator.read_text())
        expected_hashes = {row["filename"]: row["sha256"]
                           for row in manifest["images"]}
    with (run / "input_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "bytes", "sha256"))
        for filename in filenames:
            if Path(filename).name != filename:
                raise SystemExit("Input filename must be a plain basename")
            path = args.input / filename
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if expected_hashes and expected_hashes.get(filename) != digest:
                raise SystemExit("Input changed since generation: " + filename)
            writer.writerow((filename, len(data), digest))
    kernel_ms = sum(summary[key] for key in
                    ("horizontal_ms", "vertical_ms", "sobel_ms"))
    transfer_ms = summary["upload_ms"] + summary["download_ms"]
    scale_ok = len(images) >= 200
    cards = []
    for row in images:
        sample = int(row["sample_index"])
        if sample < 0:
            continue
        panels = []
        for suffix, label in (("input", "Noisy input"), ("blur", "Gaussian blur"),
                              ("edges", "Sobel edges")):
            width, height, pixels = read_pgm(run / "samples" / f"{sample}_{suffix}.pgm")
            png = png_bytes(width, height, pixels)
            (run / "samples" / f"{sample}_{suffix}.png").write_bytes(png)
            encoded = base64.b64encode(png).decode("ascii")
            panels.append(f'<figure><img src="data:image/png;base64,{encoded}" '
                          f'alt="{label}"><figcaption>{label}</figcaption></figure>')
        cards.append('<section><h2>' + html.escape(row["filename"]) + '</h2><div class="panels">' +
                     ''.join(panels) + '</div></section>')
    stages = [("Upload", summary["upload_ms"]),
              ("Horizontal blur", summary["horizontal_ms"]),
              ("Vertical blur", summary["vertical_ms"]),
              ("Sobel", summary["sobel_ms"]),
              ("Download", summary["download_ms"])]
    maximum = max(value for _, value in stages) or 1
    bars = ''.join(f'<tr><td>{name}</td><td><div class="bar" '
                   f'style="width:{value / maximum * 100:.2f}%"></div></td>'
                   f'<td>{value:.3f} ms</td></tr>' for name, value in stages)
    status = "Volume target met: at least 200 images" if scale_ok else "SMALL RUN: use at least 200 images for the default rubric target"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUDA batch image processing — execution evidence</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;background:#f4f7fb;color:#142338}}
h1{{font-size:32px}}section,.summary{{background:white;padding:20px;border-radius:12px;margin:20px 0}}
.panels{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}figure{{margin:0}}img{{width:100%;image-rendering:auto}}
figcaption{{margin-top:6px}}table{{width:100%;border-collapse:collapse}}td{{padding:8px}}td:nth-child(2){{width:60%}}
.bar{{height:18px;background:#087b8b;border-radius:4px}}@media(max-width:650px){{.panels{{grid-template-columns:1fr}}}}</style>
<h1>Gaussian blur + Sobel on CUDA</h1><div class="summary">
<p><strong>{len(images)} unique images · {summary['pixels']:,} pixels · {summary['batches']} batches</strong></p>
<p>{status}</p><p>GPU: {html.escape(summary['device'])}; compute capability {summary['compute_capability']}.</p>
<p>All images passed CPU-reference comparison. Maximum blur error: {summary['max_blur_error']}; edge mismatches: 0.</p>
<p>Kernel total: {kernel_ms:.3f} ms; transfers: {transfer_ms:.3f} ms; CPU reference: {summary['cpu_reference_ms']:.3f} ms.</p>
<p>Processing-loop wall time: {summary['loop_wall_ms']:.3f} ms, including warmup, file reads, CPU validation and sample writes.</p>
<p>GPU event timings exclude first-use warmup. They are not application end-to-end speedups.</p>
<table><caption>Total event time by stage</caption>{bars}</table></div>
<p>The default input is explicitly synthetic: geometric scenes, gradients and Gaussian noise. Samples below are produced by the recorded run.</p>
{''.join(cards)}</html>'''
    (run / "gallery.html").write_text(page)
    description = f'''# Measured CUDA project description

This project processes batches of grayscale images using custom CUDA kernels
for separable 5×5 Gaussian blur followed by thresholded 3×3 Sobel edge detection.
The default dataset contains 256 independently generated synthetic scenes with
varying geometry and Gaussian noise. No claim about natural-image performance
or recognition accuracy is made.

This run processed **{len(images)} unique {summary['width']}×{summary['height']}
images ({summary['pixels']:,} pixels)** in {summary['batches']} batches on
**{summary['device']}**. Every image was checked against a single-threaded C++
reference. Maximum blur error was {summary['max_blur_error']}; binary edge
differences were zero.

CUDA event totals were {kernel_ms:.3f} ms for kernels and {transfer_ms:.3f} ms
for transfers. CPU reference time was {summary['cpu_reference_ms']:.3f} ms.
The processing-loop wall time was {summary['loop_wall_ms']:.3f} ms and includes
warmup, I/O, CPU checks and sample writes. CPU reference timing includes scratch
allocation; GPU kernel timings exclude buffer allocation and transfers. These
measurements therefore describe separate timing scopes, not a universal speedup.

The implementation uses shared-memory tiles with halos, constant Gaussian
weights, pinned staging buffers and reusable device allocations. A key design
issue is that threads outside a partial tile must still reach its barrier;
returning early would make boundary tiles unsafe. Consistent clamp-to-edge
rules and deterministic floating-point settings make reference validation
reproducible. Batching bounds device memory while amortizing launch overhead.

The logs, image manifest with SHA-256 digests, per-image checks, per-batch
timings and gallery document execution. The synthetic dataset and one-device
measurement limit how broadly performance conclusions can be generalized.
Future work could test natural images, overlap transfers with computation,
or compare fused kernels with the three-stage implementation.

Volume check: {status}.
'''
    (run / "PROJECT_DESCRIPTION.md").write_text(description)
    project = Path(__file__).resolve().parents[1]
    source_records = []
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if (path.is_file() and
                (relative.parts[0] in ("src", "include", "tools", "tests") or
                 relative.as_posix() in ("Makefile", "run.sh", ".clang-format",
                                         "README.md", "PROJECT_DESCRIPTION.md",
                                         "LICENSE"))
                and "__pycache__" not in relative.parts):
            source_records.append(dict(path=relative.as_posix(),
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (run / "source_manifest.json").write_text(json.dumps(source_records, indent=2) + "\n")
    print(f"Evidence report: {run / 'gallery.html'}")
    print(status)


if __name__ == "__main__":
    main()
