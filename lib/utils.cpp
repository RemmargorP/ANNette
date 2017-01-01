#include "utils.hpp"

namespace Annette {
  float rand01() {
    return rand() * 1.0 / RAND_MAX;
  }
}
