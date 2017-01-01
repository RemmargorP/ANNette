#include <cstdlib>

#include "synapse.hpp"

namespace Annette {

  Synapse::Synapse(Neuron *from, Neuron *to) : from(from), to(to) {
    id = rand();
  }

  float Synapse::read() const {
    return value;
  }

  void Synapse::set(float value) {
    this->value = value;
  }

  int Synapse::getId() const {
    return id;
  }

  bool operator<(const Synapse *a, const Synapse *b) {
    if (a == nullptr) return 0;
    if (b == nullptr) return 1;
    return a->getId() < b->getId();
  }

};
