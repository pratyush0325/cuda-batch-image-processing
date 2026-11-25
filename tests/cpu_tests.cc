#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

bool Expect(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

std::vector<float> DirectBlur(const std::vector<uint8_t>& input, int width,
                               int height, int count) {
  const int coefficients[5] = {1, 4, 6, 4, 1};
  const size_t plane = static_cast<size_t>(width) * height;
  std::vector<float> output(input.size());
  for (int image = 0; image < count; ++image) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int numerator = 0;
        for (int dy = -2; dy <= 2; ++dy) {
          for (int dx = -2; dx <= 2; ++dx) {
            const int sy = std::max(0, std::min(height - 1, y + dy));
            const int sx = std::max(0, std::min(width - 1, x + dx));
            numerator += coefficients[dy + 2] * coefficients[dx + 2] *
                         input[image * plane + static_cast<size_t>(sy) * width +
                               sx];
          }
        }
        output[image * plane + static_cast<size_t>(y) * width + x] =
            numerator / 256.0f;
      }
    }
  }
  return output;
}

}  // namespace

int main() {
  std::mt19937 random(2026);
  for (const auto dimensions : {std::pair<int, int>(1, 1), {1, 17}, {19, 1},
                                {17, 19}, {32, 33}}) {
    const int width = dimensions.first, height = dimensions.second;
    std::vector<uint8_t> input(static_cast<size_t>(width) * height * 3);
    for (uint8_t& value : input) value = random() % 256;
    std::vector<float> blurred;
    std::vector<uint8_t> edges;
    batch_image::ProcessCpu(input.data(), width, height, 3, 80, &blurred, &edges);
    if (!Expect(blurred == DirectBlur(input, width, height, 3),
                 "separable blur must equal independent integer 5x5 reference"))
      return 1;
    if (!Expect(edges.size() == input.size(), "batched output size")) return 1;
  }
  std::vector<uint8_t> constant(17 * 19 * 2, 123);
  std::vector<float> blurred;
  std::vector<uint8_t> edges;
  batch_image::ProcessCpu(constant.data(), 17, 19, 2, 80, &blurred, &edges);
  if (!Expect(std::all_of(blurred.begin(), blurred.end(),
                           [](float value) { return value == 123; }) &&
               std::all_of(edges.begin(), edges.end(),
                           [](uint8_t value) { return value == 0; }),
               "constant images remain constant with zero edges")) return 1;
  std::vector<uint8_t> ramp(33 * 17);
  for (int y = 0; y < 17; ++y)
    for (int x = 0; x < 33; ++x) ramp[y * 33 + x] = x * 4;
  batch_image::ProcessCpu(ramp.data(), 33, 17, 1, 31, &blurred, &edges);
  if (!Expect(edges[8 * 33 + 16] == 255, "Sobel ramp gradient is 32")) return 1;
  batch_image::ProcessCpu(ramp.data(), 33, 17, 1, 33, &blurred, &edges);
  if (!Expect(edges[8 * 33 + 16] == 0, "Sobel threshold excludes gradient 32"))
    return 1;
  const std::string path = "build/test_roundtrip.pgm";
  const std::vector<uint8_t> bytes = {10, 13, 32, 35, 0, 255};
  std::string error;
  batch_image::Image image;
  if (!Expect(batch_image::WritePgm(path, 3, 2, bytes, &error) &&
               batch_image::ReadPgm(path, &image, &error) && image.pixels == bytes,
               "binary header/raster whitespace roundtrip")) return 1;
  {
    std::ofstream truncated(path, std::ios::binary);
    truncated << "P5\n# comment\n3 2\n255\nabc";
  }
  if (!Expect(!batch_image::ReadPgm(path, &image, &error),
               "truncated PGM rejected")) return 1;
  std::remove(path.c_str());
  std::cout << "PASS: CPU reference, Sobel fixtures, batch boundaries and PGM I/O\n";
  return 0;
}
