# Local execution environment

The included evidence was generated on the following local machine:

| Component | Recorded configuration |
| --- | --- |
| Operating system | Fedora 45 |
| Kernel | 7.3.4-300.fc45.x86_64 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB |
| Compute capability | 8.6 |
| NVIDIA driver release | 620.73.04 at the successful full-run check |
| CUDA compiler/runtime | 13.5.37 / CUDA 13.5 |
| Host compiler | GCC/G++ 16.2.1 |

The project compiled with the real NVIDIA compiler and processed 256 images in
16 batches. All 16,777,216 pixels passed CPU-reference validation, with zero
blur error and zero binary-edge mismatches. Logs, metrics, sample outputs and
the visual report are in `artifacts/run_20261125T173437Z_22134/`.

## Toolkit setup used for this run

Five official NVIDIA components were staged in a workspace-local toolkit:
NVCC, CUDA runtime development files, CCCL, NVVM and compiler runtime headers.
Downloads were checked against SHA-256 hashes in NVIDIA's CUDA 13.5.0
redistribution manifest. Component licenses were retained with that toolkit.
No system driver package, package repository or shell startup file was changed.
The NVIDIA toolkit binaries are not included in this source repository.

An initial NVML monitoring mismatch was observed during setup. A small real
CUDA check succeeded, and the later full-run `nvidia-smi` check also succeeded.
The recorded successful run, rather than that earlier monitoring failure,
establishes the execution result. No restart was performed by the setup commands.

## Running a copy on another machine

Use a compatible NVIDIA GPU, working driver, CUDA development toolkit and host
C++17 compiler. The toolkit's `nvcc` executable must be in PATH, or its path can
be supplied through the `NVCC` environment variable. Then run from the project
root:

```bash
nvcc --version
bash tools/check_environment.sh
bash run.sh
```

The setup was tested on Fedora 45, although NVIDIA's CUDA 13.5 qualification
table lists Fedora 44. This successful run does not establish official support
for every Fedora 45 installation. Match the toolkit and compiler to the machine.

- [NVIDIA Linux installation and compiler support](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA redistribution packages](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#tarball-and-zip-archive-deliverables)

The assignment requires GPU computation and sufficient input data; the supplied
instructions do not explicitly restrict execution to a hosted lab. This project
uses a real local GPU and includes the resulting execution evidence.
