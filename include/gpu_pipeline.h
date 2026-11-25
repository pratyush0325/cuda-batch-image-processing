#ifndef CUDA_BATCH_IMAGE_PROJECT_GPU_PIPELINE_H_
#define CUDA_BATCH_IMAGE_PROJECT_GPU_PIPELINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace batch_image {

struct DeviceInfo {
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

struct GpuTimings {
  float upload_ms = 0;
  float horizontal_ms = 0;
  float vertical_ms = 0;
  float sobel_ms = 0;
  float download_ms = 0;
};

struct BatchResult {
  std::vector<float> blurred;
  std::vector<uint8_t> edges;
  GpuTimings timings;
};

class GpuPipeline {
 public:
  GpuPipeline();
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  bool Initialize(int width, int height, int capacity, std::string* error);
  bool Process(const std::vector<uint8_t>& input, int count, float threshold,
               BatchResult* result, std::string* error);
  const DeviceInfo& device_info() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace batch_image

#endif  // CUDA_BATCH_IMAGE_PROJECT_GPU_PIPELINE_H_
