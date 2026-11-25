#include "image.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace batch_image {
namespace {

constexpr float kWeights[5] = {1.0f / 16, 4.0f / 16, 6.0f / 16,
                               4.0f / 16, 1.0f / 16};

int Clamp(int coordinate, int limit) {
  return std::max(0, std::min(limit - 1, coordinate));
}

}  // namespace

void ProcessCpu(const uint8_t* input, int width, int height, int count,
                float threshold, std::vector<float>* blurred,
                std::vector<uint8_t>* edges) {
  const size_t plane = static_cast<size_t>(width) * height;
  std::vector<float> temporary(plane * count);
  blurred->resize(plane * count);
  edges->resize(plane * count);
  for (int image = 0; image < count; ++image) {
    const size_t base = plane * image;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float value = 0;
        for (int offset = -2; offset <= 2; ++offset) {
          value += kWeights[offset + 2] *
                   input[base + static_cast<size_t>(y) * width +
                         Clamp(x + offset, width)];
        }
        temporary[base + static_cast<size_t>(y) * width + x] = value;
      }
    }
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float value = 0;
        for (int offset = -2; offset <= 2; ++offset) {
          value += kWeights[offset + 2] *
                   temporary[base +
                             static_cast<size_t>(Clamp(y + offset, height)) *
                                 width + x];
        }
        (*blurred)[base + static_cast<size_t>(y) * width + x] = value;
      }
    }
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float patch[3][3];
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            patch[row][column] = (*blurred)[
                base + static_cast<size_t>(Clamp(y + row - 1, height)) * width +
                Clamp(x + column - 1, width)];
          }
        }
        const float gx = -patch[0][0] + patch[0][2] - 2 * patch[1][0] +
                         2 * patch[1][2] - patch[2][0] + patch[2][2];
        const float gy = -patch[0][0] - 2 * patch[0][1] - patch[0][2] +
                         patch[2][0] + 2 * patch[2][1] + patch[2][2];
        (*edges)[base + static_cast<size_t>(y) * width + x] =
            gx * gx + gy * gy >= threshold * threshold ? 255 : 0;
      }
    }
  }
}

}  // namespace batch_image
