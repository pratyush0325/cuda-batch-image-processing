#include "gpu_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace batch_image {
namespace {

constexpr int kTile = 16;
__constant__ float kWeights[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16,
                                 4.0f / 16, 1.0f / 16};

__device__ int Clamp(int value, int limit) {
  return max(0, min(limit - 1, value));
}

__global__ void GaussianHorizontal(const uint8_t* input, float* output,
                                   int width, int height) {
  __shared__ float tile[kTile][kTile + 4];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x = blockIdx.x * kTile + tx;
  const int y = blockIdx.y * kTile + ty;
  const size_t base = static_cast<size_t>(blockIdx.z) * width * height;
  for (int column = tx; column < kTile + 4; column += kTile) {
    const int source_x =
        Clamp(static_cast<int>(blockIdx.x) * kTile + column - 2, width);
    tile[ty][column] = input[base +
                             static_cast<size_t>(Clamp(y, height)) * width +
                             source_x];
  }
  __syncthreads();
  // Threads outside a partial tile still participate in halo loads/barriers.
  if (x >= width || y >= height) return;
  float value = 0;
  for (int offset = 0; offset < 5; ++offset) {
    value += kWeights[offset] * tile[ty][tx + offset];
  }
  output[base + static_cast<size_t>(y) * width + x] = value;
}

__global__ void GaussianVertical(const float* input, float* output,
                                 int width, int height) {
  __shared__ float tile[kTile + 4][kTile];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x = blockIdx.x * kTile + tx;
  const int y = blockIdx.y * kTile + ty;
  const size_t base = static_cast<size_t>(blockIdx.z) * width * height;
  for (int row = ty; row < kTile + 4; row += kTile) {
    const int source_y =
        Clamp(static_cast<int>(blockIdx.y) * kTile + row - 2, height);
    tile[row][tx] = input[base + static_cast<size_t>(source_y) * width +
                          Clamp(x, width)];
  }
  __syncthreads();
  if (x >= width || y >= height) return;
  float value = 0;
  for (int offset = 0; offset < 5; ++offset) {
    value += kWeights[offset] * tile[ty + offset][tx];
  }
  output[base + static_cast<size_t>(y) * width + x] = value;
}

__global__ void SobelEdges(const float* input, uint8_t* output,
                           int width, int height, float threshold) {
  __shared__ float tile[kTile + 2][kTile + 2];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x = blockIdx.x * kTile + tx;
  const int y = blockIdx.y * kTile + ty;
  const size_t base = static_cast<size_t>(blockIdx.z) * width * height;
  for (int index = ty * kTile + tx; index < (kTile + 2) * (kTile + 2);
       index += kTile * kTile) {
    const int row = index / (kTile + 2);
    const int column = index % (kTile + 2);
    const int source_x =
        Clamp(static_cast<int>(blockIdx.x) * kTile + column - 1, width);
    const int source_y =
        Clamp(static_cast<int>(blockIdx.y) * kTile + row - 1, height);
    tile[row][column] = input[base + static_cast<size_t>(source_y) * width +
                              source_x];
  }
  __syncthreads();
  if (x >= width || y >= height) return;
  const float gx = -tile[ty][tx] + tile[ty][tx + 2] -
                   2 * tile[ty + 1][tx] + 2 * tile[ty + 1][tx + 2] -
                   tile[ty + 2][tx] + tile[ty + 2][tx + 2];
  const float gy = -tile[ty][tx] - 2 * tile[ty][tx + 1] - tile[ty][tx + 2] +
                   tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] +
                   tile[ty + 2][tx + 2];
  output[base + static_cast<size_t>(y) * width + x] =
      gx * gx + gy * gy >= threshold * threshold ? 255 : 0;
}

bool CheckCuda(cudaError_t status, const char* operation, std::string* error) {
  if (status == cudaSuccess) return true;
  *error = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

}  // namespace

struct GpuPipeline::Impl {
  int width = 0;
  int height = 0;
  int capacity = 0;
  uint8_t* input = nullptr;
  float* temporary = nullptr;
  float* blurred = nullptr;
  uint8_t* edges = nullptr;
  uint8_t* pinned_input = nullptr;
  float* pinned_blurred = nullptr;
  uint8_t* pinned_edges = nullptr;
  cudaEvent_t events[6] = {};
  DeviceInfo info;

  ~Impl() {
    if (input) cudaFree(input);
    if (temporary) cudaFree(temporary);
    if (blurred) cudaFree(blurred);
    if (edges) cudaFree(edges);
    if (pinned_input) cudaFreeHost(pinned_input);
    if (pinned_blurred) cudaFreeHost(pinned_blurred);
    if (pinned_edges) cudaFreeHost(pinned_edges);
    for (cudaEvent_t event : events) {
      if (event) cudaEventDestroy(event);
    }
  }
};

GpuPipeline::GpuPipeline() : impl_(new Impl) {}
GpuPipeline::~GpuPipeline() = default;

const DeviceInfo& GpuPipeline::device_info() const { return impl_->info; }

bool GpuPipeline::Initialize(int width, int height, int capacity,
                              std::string* error) {
  if (impl_->width != 0 || width <= 0 || height <= 0 || width > 8192 ||
      height > 8192 || capacity < 1 || capacity > 256) {
    *error = "Invalid dimensions/batch size, or pipeline already initialized";
    return false;
  }
  int device = 0;
  cudaDeviceProp properties;
  if (!CheckCuda(cudaGetDevice(&device), "CUDA device unavailable", error) ||
      !CheckCuda(cudaGetDeviceProperties(&properties, device),
                 "Read device properties", error)) return false;
  if (properties.maxThreadsPerBlock < kTile * kTile) {
    *error = "GPU does not support the 256-thread tile";
    return false;
  }
  impl_->info.name = properties.name;
  impl_->info.compute_major = properties.major;
  impl_->info.compute_minor = properties.minor;
  if (!CheckCuda(cudaDriverGetVersion(&impl_->info.driver_version),
                 "Read driver version", error) ||
      !CheckCuda(cudaRuntimeGetVersion(&impl_->info.runtime_version),
                 "Read runtime version", error)) return false;
  size_t free_bytes = 0, total_bytes = 0;
  if (!CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                 "Read available GPU memory", error)) return false;
  const size_t pixels = static_cast<size_t>(width) * height * capacity;
  if (static_cast<double>(pixels) * 10 > free_bytes * 0.8) {
    *error = "Batch needs too much device memory; reduce --batch";
    return false;
  }
  impl_->width = width;
  impl_->height = height;
  impl_->capacity = capacity;
  if (!CheckCuda(cudaMalloc(&impl_->input, pixels), "Allocate device input",
                 error) ||
      !CheckCuda(cudaMalloc(&impl_->temporary, pixels * sizeof(float)),
                 "Allocate horizontal buffer", error) ||
      !CheckCuda(cudaMalloc(&impl_->blurred, pixels * sizeof(float)),
                 "Allocate blur buffer", error) ||
      !CheckCuda(cudaMalloc(&impl_->edges, pixels), "Allocate device edges",
                 error) ||
      !CheckCuda(cudaMallocHost(&impl_->pinned_input, pixels),
                 "Allocate pinned input", error) ||
      !CheckCuda(cudaMallocHost(&impl_->pinned_blurred, pixels * sizeof(float)),
                 "Allocate pinned blur output", error) ||
      !CheckCuda(cudaMallocHost(&impl_->pinned_edges, pixels),
                 "Allocate pinned edge output", error)) return false;
  for (cudaEvent_t& event : impl_->events) {
    if (!CheckCuda(cudaEventCreate(&event), "Create CUDA timing event", error))
      return false;
  }
  return true;
}

bool GpuPipeline::Process(const std::vector<uint8_t>& input, int count,
                           float threshold, BatchResult* result,
                           std::string* error) {
  const size_t pixels = static_cast<size_t>(impl_->width) * impl_->height * count;
  if (count < 1 || count > impl_->capacity || input.size() != pixels) {
    *error = "Batch does not match allocated dimensions/capacity";
    return false;
  }
  std::copy(input.begin(), input.end(), impl_->pinned_input);
  cudaEvent_t* events = impl_->events;
  if (!CheckCuda(cudaEventRecord(events[0]), "Record upload start", error) ||
      !CheckCuda(cudaMemcpyAsync(impl_->input, impl_->pinned_input, pixels,
                                cudaMemcpyHostToDevice), "Upload batch", error) ||
      !CheckCuda(cudaEventRecord(events[1]), "Record upload stop", error))
    return false;
  const dim3 block(kTile, kTile);
  const dim3 grid((impl_->width + kTile - 1) / kTile,
                  (impl_->height + kTile - 1) / kTile, count);
  GaussianHorizontal<<<grid, block>>>(impl_->input, impl_->temporary,
                                      impl_->width, impl_->height);
  if (!CheckCuda(cudaGetLastError(), "Launch horizontal blur", error) ||
      !CheckCuda(cudaEventRecord(events[2]), "Record horizontal stop", error))
    return false;
  GaussianVertical<<<grid, block>>>(impl_->temporary, impl_->blurred,
                                    impl_->width, impl_->height);
  if (!CheckCuda(cudaGetLastError(), "Launch vertical blur", error) ||
      !CheckCuda(cudaEventRecord(events[3]), "Record vertical stop", error))
    return false;
  SobelEdges<<<grid, block>>>(impl_->blurred, impl_->edges, impl_->width,
                             impl_->height, threshold);
  if (!CheckCuda(cudaGetLastError(), "Launch Sobel", error) ||
      !CheckCuda(cudaEventRecord(events[4]), "Record Sobel stop", error) ||
      !CheckCuda(cudaMemcpyAsync(impl_->pinned_blurred, impl_->blurred,
                                pixels * sizeof(float), cudaMemcpyDeviceToHost),
                 "Download blur output", error) ||
      !CheckCuda(cudaMemcpyAsync(impl_->pinned_edges, impl_->edges, pixels,
                                cudaMemcpyDeviceToHost), "Download edges",
                 error) ||
      !CheckCuda(cudaEventRecord(events[5]), "Record download stop", error) ||
      !CheckCuda(cudaEventSynchronize(events[5]), "Wait for GPU batch", error))
    return false;
  float* times[] = {&result->timings.upload_ms,
                     &result->timings.horizontal_ms,
                     &result->timings.vertical_ms, &result->timings.sobel_ms,
                     &result->timings.download_ms};
  for (int i = 0; i < 5; ++i) {
    if (!CheckCuda(cudaEventElapsedTime(times[i], events[i], events[i + 1]),
                   "Read GPU timing", error)) return false;
  }
  result->blurred.assign(impl_->pinned_blurred, impl_->pinned_blurred + pixels);
  result->edges.assign(impl_->pinned_edges, impl_->pinned_edges + pixels);
  return true;
}

}  // namespace batch_image
