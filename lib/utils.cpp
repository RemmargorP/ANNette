#include "utils.hpp"

namespace ANNette {
  float rand01() {
    return rand() * 1.0 / RAND_MAX;
  }
}
