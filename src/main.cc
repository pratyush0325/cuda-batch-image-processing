#include "gpu_pipeline.h"
#include "image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace batch_image {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
  std::string input = "data/input";
  std::string output = "artifacts/manual_run";
  int batch = 16;
  int limit = 0;
  int samples = 8;
  float threshold = 80;
};

bool PositiveInteger(const std::string& text, int maximum, int* value) {
  if (text.empty() || text.size() > 9 ||
      text.find_first_not_of("0123456789") != std::string::npos) return false;
  const long number = std::strtol(text.c_str(), nullptr, 10);
  if (number < 1 || number > maximum) return false;
  *value = static_cast<int>(number);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) return false;
    const std::string key(argv[i]), value(argv[i + 1]);
    if (key == "--input") options->input = value;
    else if (key == "--output") options->output = value;
    else if (key == "--batch") {
      if (!PositiveInteger(value, 256, &options->batch)) return false;
    } else if (key == "--limit") {
      if (!PositiveInteger(value, 100000, &options->limit)) return false;
    } else if (key == "--samples") {
      if (!PositiveInteger(value, 64, &options->samples)) return false;
    } else if (key == "--threshold") {
      char* end = nullptr;
      options->threshold = std::strtof(value.c_str(), &end);
      if (end == value.c_str() || *end != '\0' ||
          !std::isfinite(options->threshold) || options->threshold <= 0 ||
          options->threshold > 4096) return false;
    } else return false;
  }
  return true;
}

std::string JsonString(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') {
      stream << '\\' << character;
    } else if (character < 32) {
      stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(character) << std::dec;
    } else stream << character;
  }
  stream << '"';
  return stream.str();
}

std::string CsvString(const std::string& value) {
  std::string result = "\"";
  for (char character : value) {
    if (character == '"') result += '"';
    result += character;
  }
  return result + '"';
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool SaveSample(const fs::path& folder, size_t index, const uint8_t* input,
                const float* blurred, const uint8_t* edges, int width,
                int height, std::string* error) {
  const size_t plane = static_cast<size_t>(width) * height;
  const std::string prefix = (folder / std::to_string(index)).string();
  return WritePgm(prefix + "_input.pgm", width, height,
                  std::vector<uint8_t>(input, input + plane), error) &&
         WritePgm(prefix + "_blur.pgm", width, height,
                  Quantize(std::vector<float>(blurred, blurred + plane)), error) &&
         WritePgm(prefix + "_edges.pgm", width, height,
                  std::vector<uint8_t>(edges, edges + plane), error);
}

int Run(const Options& options) {
  std::error_code filesystem_error;
  std::vector<fs::path> files;
  fs::directory_iterator iterator(options.input, filesystem_error), end;
  if (filesystem_error) {
    std::cerr << "Cannot open input directory: " << filesystem_error.message()
              << '\n';
    return 1;
  }
  for (; iterator != end; iterator.increment(filesystem_error)) {
    if (filesystem_error) break;
    if (iterator->is_regular_file(filesystem_error) &&
        iterator->path().extension() == ".pgm") files.push_back(iterator->path());
  }
  if (filesystem_error || files.empty()) {
    std::cerr << "Input must contain readable .pgm files\n";
    return 1;
  }
  std::sort(files.begin(), files.end());
  if (options.limit > 0 && files.size() > static_cast<size_t>(options.limit))
    files.resize(options.limit);
  const fs::path output(options.output);
  fs::create_directories(output / "samples", filesystem_error);
  if (filesystem_error || fs::exists(output / "images.csv") ||
      fs::exists(output / "summary.json")) {
    std::cerr << "Output must be a new run directory; choose another --output\n";
    return 1;
  }
  std::string error;
  Image first;
  if (!ReadPgm(files[0].string(), &first, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  const int width = first.width, height = first.height;
  const size_t plane = first.pixels.size();
  GpuPipeline pipeline;
  if (!pipeline.Initialize(width, height, options.batch, &error)) {
    std::cerr << error << "\nA real CUDA GPU is required; no CPU fallback.\n";
    return 1;
  }
  std::ofstream images(output / "images.csv");
  std::ofstream batches(output / "batches.csv");
  if (!images || !batches) {
    std::cerr << "Cannot open evidence CSV files\n";
    return 1;
  }
  images << "image_index,filename,width,height,batch_id,max_blur_error,"
            "edge_mismatches,edge_pixels,sample_index\n";
  batches << "batch_id,images,upload_ms,horizontal_ms,vertical_ms,sobel_ms,"
             "download_ms,cpu_reference_ms\n";
  images << std::fixed << std::setprecision(6);
  batches << std::fixed << std::setprecision(6);
  const DeviceInfo& info = pipeline.device_info();
  std::cout << "Backend: CUDA\nGPU: " << info.name << "\nImages: "
            << files.size() << "\nDimensions: " << width << 'x' << height
            << "\nBatch capacity: " << options.batch << '\n';
  double upload_ms = 0, horizontal_ms = 0, vertical_ms = 0, sobel_ms = 0;
  double download_ms = 0, cpu_ms = 0;
  float max_error = 0;
  size_t batch_id = 0;
  const auto begin = Clock::now();
  for (size_t start = 0; start < files.size(); start += options.batch, ++batch_id) {
    const int count = static_cast<int>(
        std::min(static_cast<size_t>(options.batch), files.size() - start));
    std::vector<uint8_t> input;
    input.reserve(plane * count);
    for (int local = 0; local < count; ++local) {
      Image image;
      if (!ReadPgm(files[start + local].string(), &image, &error) ||
          image.width != width || image.height != height) {
        std::cerr << "Every image must have equal dimensions: "
                  << files[start + local] << ' ' << error << '\n';
        return 1;
      }
      input.insert(input.end(), image.pixels.begin(), image.pixels.end());
    }
    BatchResult gpu;
    // Exclude first-use warmup from event-based timing totals.
    if (start == 0 && !pipeline.Process(input, count, options.threshold,
                                       &gpu, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    if (!pipeline.Process(input, count, options.threshold, &gpu, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::vector<float> reference_blur;
    std::vector<uint8_t> reference_edges;
    const auto cpu_start = Clock::now();
    ProcessCpu(input.data(), width, height, count, options.threshold,
                &reference_blur, &reference_edges);
    const double reference_ms = Milliseconds(cpu_start, Clock::now());
    for (int local = 0; local < count; ++local) {
      float image_error = 0;
      size_t mismatches = 0, edge_pixels = 0;
      const size_t offset = plane * local;
      for (size_t pixel = 0; pixel < plane; ++pixel) {
        const size_t index = offset + pixel;
        if (!std::isfinite(gpu.blurred[index])) {
          std::cerr << "Nonfinite GPU output\n";
          return 1;
        }
        image_error = std::max(image_error,
                              std::abs(gpu.blurred[index] - reference_blur[index]));
        mismatches += gpu.edges[index] != reference_edges[index];
        edge_pixels += gpu.edges[index] != 0;
      }
      if (image_error > 0.0001f || mismatches != 0) {
        std::cerr << "GPU reference mismatch: " << files[start + local]
                  << " blur_error=" << image_error
                  << " edge_mismatches=" << mismatches << '\n';
        return 1;
      }
      const size_t index = start + local;
      const bool sample = index < static_cast<size_t>(options.samples);
      if (sample && !SaveSample(output / "samples", index, input.data() + offset,
                                gpu.blurred.data() + offset,
                                gpu.edges.data() + offset, width, height, &error)) {
        std::cerr << error << '\n';
        return 1;
      }
      images << index << ',' << CsvString(files[index].filename().string())
             << ',' << width << ',' << height << ',' << batch_id << ','
             << image_error << ',' << mismatches << ',' << edge_pixels << ','
             << (sample ? static_cast<int>(index) : -1) << '\n';
      max_error = std::max(max_error, image_error);
    }
    const GpuTimings& times = gpu.timings;
    upload_ms += times.upload_ms;
    horizontal_ms += times.horizontal_ms;
    vertical_ms += times.vertical_ms;
    sobel_ms += times.sobel_ms;
    download_ms += times.download_ms;
    cpu_ms += reference_ms;
    batches << batch_id << ',' << count << ',' << times.upload_ms << ','
            << times.horizontal_ms << ',' << times.vertical_ms << ','
            << times.sobel_ms << ',' << times.download_ms << ','
            << reference_ms << '\n';
    std::cout << "Batch " << batch_id << ": validated " << count << " images\n";
  }
  const double wall_ms = Milliseconds(begin, Clock::now());
  images.close();
  batches.close();
  if (!images || !batches) {
    std::cerr << "Evidence CSV write failed\n";
    return 1;
  }
  std::ofstream summary(output / "summary.json");
  summary << std::fixed << std::setprecision(6)
          << "{\n  \"status\": \"passed\",\n  \"backend\": \"CUDA\",\n"
          << "  \"device\": " << JsonString(info.name) << ",\n"
          << "  \"compute_capability\": "
          << JsonString(std::to_string(info.compute_major) + "." +
                        std::to_string(info.compute_minor)) << ",\n"
          << "  \"driver_version\": " << info.driver_version << ",\n"
          << "  \"runtime_version\": " << info.runtime_version << ",\n"
          << "  \"images\": " << files.size() << ",\n"
          << "  \"validated_images\": " << files.size() << ",\n"
          << "  \"width\": " << width << ", \"height\": " << height << ",\n"
          << "  \"pixels\": " << plane * files.size() << ",\n"
          << "  \"batch_capacity\": " << options.batch << ",\n"
          << "  \"batches\": " << batch_id << ",\n"
          << "  \"threshold\": " << options.threshold << ",\n"
          << "  \"warmup_batches\": 1,\n"
          << "  \"max_blur_error\": " << max_error << ",\n"
          << "  \"edge_mismatches\": 0,\n"
          << "  \"upload_ms\": " << upload_ms << ",\n"
          << "  \"horizontal_ms\": " << horizontal_ms << ",\n"
          << "  \"vertical_ms\": " << vertical_ms << ",\n"
          << "  \"sobel_ms\": " << sobel_ms << ",\n"
          << "  \"download_ms\": " << download_ms << ",\n"
          << "  \"cpu_reference_ms\": " << cpu_ms << ",\n"
          << "  \"loop_wall_ms\": " << wall_ms << "\n}\n";
  summary.close();
  if (!summary) {
    std::cerr << "Summary write failed\n";
    return 1;
  }
  std::cout << "PASS: all " << files.size()
            << " images matched the CPU reference. Evidence: " << output << '\n';
  return 0;
}

}  // namespace
}  // namespace batch_image

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Usage: batch_image [--input DIR] [--output NEW_DIR] "
                 "[--batch 1..256] [--threshold 0<T<=4096] "
                 "[--limit N] [--samples 1..64]\n"
                 "Input: equal-sized P5 PGM images, 8-bit grayscale. "
                 "A CUDA GPU is mandatory.\n";
    return 0;
  }
  batch_image::Options options;
  if (!batch_image::ParseOptions(argc, argv, &options)) {
    std::cerr << "Invalid arguments. Use --help.\n";
    return 1;
  }
  return batch_image::Run(options);
}
