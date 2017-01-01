#include <cmath>

#include "utils.hpp"
#include "neuron.hpp"

namespace Annette {

  Neuron::Neuron() {
    value = 0;
    bias = rand01();
  }

  void Neuron::dependOn(Neuron *target) {
    if (in.find(target) != in.end()) in[target] = rand01();
  }

  void Neuron::deleteDependency(Neuron *target) {
    if (in.find(target) != in.end()) in.erase(target);
  }

  float Neuron::calculate() {
    value = bias;
    for (auto d : in) {
      value += d.first->getValue() * d.second;
    }
    return out = activate(value);
  }

  float Neuron::activate(float value) const {
    return 1 / (1 + exp(-value));
  }

  float Neuron::getValue() const {
    return out;
  }

}
