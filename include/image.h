#ifndef CUDA_BATCH_IMAGE_PROJECT_IMAGE_H_
#define CUDA_BATCH_IMAGE_PROJECT_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace batch_image {

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

bool ReadPgm(const std::string& path, Image* image, std::string* error);
bool WritePgm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels, std::string* error);
std::vector<uint8_t> Quantize(const std::vector<float>& pixels);

// Clamp-to-edge boundary conditions are used by both backends.
void ProcessCpu(const uint8_t* input, int width, int height, int count,
                float threshold, std::vector<float>* blurred,
                std::vector<uint8_t>* edges);

}  // namespace batch_image

#endif  // CUDA_BATCH_IMAGE_PROJECT_IMAGE_H_
