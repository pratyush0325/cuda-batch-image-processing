# CUDA batch image processing

Custom CUDA Gaussian blur and Sobel edge detection over **256 synthetic
256×256 grayscale images**: 16,777,216 input pixels in one program execution.
The default run uses batches of 16 images, not 256 separate process launches.
The application requires a CUDA GPU and has no CPU fallback. Every GPU result
is checked against a single-threaded C++ reference before success is recorded.

This package includes a genuine local CUDA execution on an NVIDIA GeForce
RTX 3070 Laptop GPU. All 256 images matched the CPU reference with zero blur
error and zero edge-mask mismatches. The recorded run is in
[`artifacts/run_20261125T173437Z_22134/`](artifacts/run_20261125T173437Z_22134/).
Open its `gallery.html` for samples and measured timings. No CPU-emulation
fixtures are included as execution evidence.

## Quick start on a local NVIDIA GPU

Download the complete ZIP, then extract it. 

```bash
unzip cuda-batch-image-project.zip
cd cuda-batch-image-project
bash tools/check_environment.sh
bash run.sh
```

Requirements: Linux, CUDA toolkit with C++17 support, a compatible NVIDIA GPU
and driver, a compatible C++17 host compiler (GCC 9+ recommended), GNU Make,
and Python 3.6+. See `LOCAL_SETUP.md` for the current laptop setup status.
Python scripts use only the standard library. OpenCV, NPP,
Pillow, NumPy, and internet downloads are not required.

`run.sh` checks prerequisites, runs CPU/tool tests, builds with NVCC, generates/reuses the deterministic
256-image dataset, records the environment, processes all images, checks all GPU
outputs, and creates a portable HTML gallery and measured project description.
Each invocation uses a new directory under `artifacts/`.

If the GPU has limited memory, use `BATCH_SIZE=4 bash run.sh`. If an older GCC 8
toolchain reports undefined filesystem references, try `make build
LDLIBS=-lstdc++fs` before the run. An unsupported compiler/CUDA combination must
be resolved using a compatible host compiler and CUDA toolkit.

## Command-line interface

```bash
make build
./build/batch_image --help
./build/batch_image --input data/input --output artifacts/custom_run \
    --batch 16 --threshold 80 --samples 8
```

| Argument | Meaning | Default |
| --- | --- | --- |
| `--input` | Directory of equal-sized `.pgm` files | `data/input` |
| `--output` | New output run directory | `artifacts/manual_run` |
| `--batch` | Images per CUDA batch, 1–256 | 16 |
| `--threshold` | Positive Sobel gradient threshold, up to 4096 | 80 |
| `--limit` | Optional maximum number of input files | All files |
| `--samples` | First 1–64 images to save for the gallery | 8 |

Input is binary 8-bit PGM (`P5`, maximum value 255); dimensions must be 1–8192
on each axis and identical across the batch. Real images can also be used after
conversion to this format. The generated dataset is explicitly synthetic, with
gradients, circles, rectangles, lines, stripes, and Gaussian noise. It is not a
photographic dataset or evidence of real-world detection accuracy. Seeds,
parameters, and hashes are stored in `data/input/manifest.json`. The generator
reuses an existing dataset only if its parameters and all hashes match.

## Algorithms and CUDA design

1. **Horizontal Gaussian kernel:** five taps with weights `[1,4,6,4,1]/16`.
2. **Vertical Gaussian kernel:** another five taps, equivalent to the outer
   product 5×5 filter while using ten taps per pixel instead of 25.
3. **Sobel kernel:** a 3×3 horizontal/vertical gradient and a binary test
   `gx² + gy² >= threshold²`. This avoids a square root.

Each launch uses 16×16 threads; grid Z selects the image within a batch. The
three kernels cooperatively load shared-memory tiles with halos. Threads in
partial boundary tiles participate in loads and barriers before checking output
bounds. Clamp-to-edge boundary handling matches the CPU implementation.
The Gaussian coefficients are in CUDA constant memory.

Device buffers and pinned host staging buffers are reused across batches.
Pinned staging supports asynchronous transfers; launches and copies use one
ordered stream. Transfers and kernels are separately timed with CUDA events.
This implementation does **not** claim transfer/computation overlap.

The build disables fused multiply-add contraction on CPU/GPU to make the strict
binary-edge comparison reproducible near the threshold. This is a deliberate
correctness/performance tradeoff. Blur maximum absolute error must be ≤0.0001;
edge masks must match the CPU reference at every pixel.

## Evidence and timing interpretation

A completed `artifacts/run_.../` directory contains:

| File | Evidence |
| --- | --- |
| `console.log`, `environment.txt` | Actual invocation, device/toolkit/driver context and per-batch progress |
| `summary.json` | CUDA backend, device, unique image/pixel counts, validation and measured totals |
| `images.csv` | One validation record for every processed image |
| `batches.csv` | Image counts and upload/kernel/download/CPU times for every batch |
| `input_manifest.csv` | Every processed filename, size and SHA-256 digest |
| `source_manifest.json` | Source files and hashes for the recorded implementation |
| `samples/` | Input, blur and edge outputs for the first eight images, PGM and PNG |
| `gallery.html` | Self-contained visual report with embedded PNGs and timing bars |
| `PROJECT_DESCRIPTION.md` | Description populated with that run's measured results |

GPU event totals exclude a first-batch warmup and all file I/O. CPU reference
timing includes its scratch allocation and three passes. Processing-loop wall
time includes warmup, file reads, CPU validation and sample writes, but excludes
initial CUDA buffer allocation, dataset generation and the Python report.
These are distinct timing scopes; a kernel-only ratio is not an end-to-end
application speedup. No speedup is assumed in advance.

In `summary.json`, `driver_version` and `runtime_version` are the integer CUDA
API versions returned by CUDA (13050 means CUDA 13.5), not the NVIDIA driver
release number. The NVIDIA release is recorded separately in `environment.txt`.

The report checks for passed CUDA results and a consistent image/batch count.
It flags runs with fewer than 200 images as small runs. The default 256-image
run is intended to satisfy the assignment's hundreds-of-small-images option.
The report additionally compares input hashes with the generator manifest.

## Tests

```bash
make test                 # g++ and Python only; no GPU required
```

Tests compare the separable CPU filter to an independent integer 5×5
convolution, exercise 1-pixel dimensions and odd sizes, verify Sobel fixtures
and PGM binary raster handling, and check generator reproducibility and report
failure paths. Passing CPU tests is not proof that CUDA ran. `bash run.sh` is
the GPU integration and full-dataset correctness check.

## Submission workflow

1. Run `bash run.sh` on your NVIDIA GPU and confirm all 256 images pass.
2. Inspect the gallery and measured description. Keep the entire successful
   run directory, including CSVs, logs and sample images.
3. Put the complete source tree and that evidence in a public repository. The
   `.gitignore` intentionally keeps `artifacts/` visible to Git while ignoring
   reproducible input data and build products.
4. Submit the actual public repository URL, the execution artifacts, and the
   generated description. Verify the repository is readable without signing in.

Create a public GitHub repository, then upload the extracted contents of this
project folder at its root, including `artifacts/`. If using Git locally,
GitHub displays the repository-specific commands for connecting and pushing
an existing project; use those commands after committing these files.

The included artifact directory already contains genuine GPU execution proof.
For the course, provide the public repository URL and the measured project
description. The rubric's subjective style/description criteria remain subject
to peer review.

## Code style and references

The C++ code uses include guards, two-space indentation, explicit ownership,
descriptive names, small modules and Google-style naming; `.clang-format`
selects the Google preset. Explicit exceptions and namespace-wide `using`
directives are avoided. Run `clang-format -i include/*.h src/*.cc src/*.cu
tests/*.cc` if that tool is available. Full rubric compliance is not guaranteed
by a formatter.

- [CUDA thread/block model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [CUDA event timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Course example template](https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate)

Source code and generated synthetic inputs are provided under the MIT license.
