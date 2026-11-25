# Measured CUDA project description

This project processes batches of grayscale images using custom CUDA kernels
for separable 5×5 Gaussian blur followed by thresholded 3×3 Sobel edge detection.
The default dataset contains 256 independently generated synthetic scenes with
varying geometry and Gaussian noise. No claim about natural-image performance
or recognition accuracy is made.

This run processed **256 unique 256×256
images (16,777,216 pixels)** in 16 batches on
**NVIDIA GeForce RTX 3070 Laptop GPU**. Every image was checked against a single-threaded C++
reference. Maximum blur error was 0.0; binary edge
differences were zero.

CUDA event totals were 4.010 ms for kernels and 16.871 ms
for transfers. CPU reference time was 301.577 ms.
The processing-loop wall time was 414.385 ms and includes
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

Volume check: Volume target met: at least 200 images.
