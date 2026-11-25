# Source-package validation

The following checks were completed locally while preparing this package:

- C++17 CPU reference and image I/O compiled with `-Wall -Wextra -Werror`.
- The separable blur matched an independent integer 5×5 convolution, including
  single-pixel axes, odd sizes and multiple images.
- Known constant-image and linear-ramp Sobel cases passed.
- PGM raster bytes containing whitespace and `#` round-tripped correctly;
  truncated input was rejected.
- Python dataset reproducibility, altered-input rejection, PNG encoding, and
  report rejection of failed/non-CUDA summaries passed.
- A test-only coroutine-based CPU emulator executed the actual tiled kernel
  bodies, pausing every thread at its shared-memory barrier before resuming the
  block. Input dimensions 1×1, 17×19, 32×33, 31×1, 16×16 and 256×256 passed
  complete reference comparison, including partial batches.
- The full default workload of 256 unique 256×256 images passed the same
  emulated end-to-end path in 16 batches, with all 16,777,216 pixels checked.
- Report generation, sample PNGs, data/source manifests, and small/full run
  labeling were exercised using temporary test fixtures.

The checks above were preparatory CPU tests, not GPU evidence. Subsequently,
an isolated official NVIDIA CUDA 13.5.37 compiler was staged locally and the
actual project compiled successfully with GCC 16.2.1. A real full CUDA run then
processed 256 images on the NVIDIA GeForce RTX 3070 Laptop GPU in 16 batches.
All 16,777,216 pixels matched the CPU reference: zero maximum blur error and
zero binary-edge mismatches.

The genuine logs, metrics, manifests and images are saved in
`artifacts/run_20261125T173437Z_22134/`. This evidence is included with the source.
It is independent of the temporary emulation fixtures described above. Keep
this genuine artifact directory with the source when publishing the repository.
