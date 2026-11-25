#include "image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>

namespace batch_image {
namespace {

bool ReadToken(std::istream* stream, std::string* token) {
  token->clear();
  char character;
  while (stream->get(character)) {
    if (character == '#') {
      stream->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else if (!std::isspace(static_cast<unsigned char>(character))) {
      token->push_back(character);
      break;
    }
  }
  while (stream->get(character)) {
    if (std::isspace(static_cast<unsigned char>(character))) {
      if (character == '\r' && stream->peek() == '\n') stream->get();
      return !token->empty();
    }
    if (character == '#') {
      stream->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      return !token->empty();
    }
    token->push_back(character);
  }
  return !token->empty();
}

bool ParseDimension(const std::string& text, int* value) {
  if (text.empty() || text.size() > 5) return false;
  int number = 0;
  for (char digit : text) {
    if (digit < '0' || digit > '9') return false;
    number = number * 10 + digit - '0';
  }
  if (number < 1 || number > 8192) return false;
  *value = number;
  return true;
}

}  // namespace

bool ReadPgm(const std::string& path, Image* image, std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  std::string magic, width, height, maximum;
  if (!stream || !ReadToken(&stream, &magic) || magic != "P5" ||
      !ReadToken(&stream, &width) || !ReadToken(&stream, &height) ||
      !ReadToken(&stream, &maximum) || maximum != "255" ||
      !ParseDimension(width, &image->width) ||
      !ParseDimension(height, &image->height)) {
    *error = "Expected an 8-bit binary PGM (P5, max=255): " + path;
    return false;
  }
  image->pixels.resize(static_cast<size_t>(image->width) * image->height);
  stream.read(reinterpret_cast<char*>(image->pixels.data()),
              static_cast<std::streamsize>(image->pixels.size()));
  if (stream.gcount() != static_cast<std::streamsize>(image->pixels.size())) {
    *error = "Truncated image: " + path;
    return false;
  }
  if (stream.peek() != std::char_traits<char>::eof()) {
    *error = "Unexpected bytes after the PGM raster: " + path;
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels, std::string* error) {
  if (width <= 0 || height <= 0 ||
      pixels.size() != static_cast<size_t>(width) * height) {
    *error = "Image dimensions do not match output pixels";
    return false;
  }
  std::ofstream stream(path, std::ios::binary);
  stream << "P5\n" << width << ' ' << height << "\n255\n";
  stream.write(reinterpret_cast<const char*>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));
  stream.close();
  if (!stream) {
    *error = "Could not write " + path;
    return false;
  }
  return true;
}

std::vector<uint8_t> Quantize(const std::vector<float>& pixels) {
  std::vector<uint8_t> result(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    result[i] = static_cast<uint8_t>(
        std::lround(std::max(0.0f, std::min(255.0f, pixels[i]))));
  }
  return result;
}

}  // namespace batch_image
