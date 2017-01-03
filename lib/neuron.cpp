#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.hpp"
#include "neuron.hpp"

namespace ANNette {

  Neuron::Neuron() {
    out = 0;
    bias = rand01();
  }

  Neuron::~Neuron() {
    in.clear();
  }

  void Neuron::dependOn(Neuron *target) {
    if (in.find(target) != in.end()) in[target] = rand01();
  }

  void Neuron::deleteDependency(Neuron *target) {
    if (in.find(target) != in.end()) in.erase(target);
  }

  float Neuron::getWeight(Neuron *target) {
    if (in.find(target) == in.end()) return 0;
    return in[target];
  }

  float Neuron::calculate() {
    float value = bias;
    for (auto d : in) {
      value += d.first->getValue() * d.second;
    }
    return out = activate(value);
  }

  float Neuron::activate(float value) const {
    return 1 / (1 + exp(-value));
  }

  float Neuron::activateDerivative() const {
    return out * (1 - out);
  }

  void Neuron::changeWeight(Neuron *target, float delta) {
    in[target] += delta;
  }

  void Neuron::changeBias(float delta) {
    bias += delta;
  }

  float Neuron::getValue() const {
    return out;
  }

  void Neuron::setValue(float v) {
    out = v;
  }

  float Neuron::getDelta() const {
    return delta;
  }

  void Neuron::setDelta(float v) {
    delta = v;
  }

  std::string Neuron::dump() const {
    std::stringstream res;
    res << "neuron " << (int*)this << ' ' << bias << ' ' << in.size() << ' ';
    for (auto d : in)
      res << d.first << ' ' << d.second << ' ';
    return res.str();
  }
}
