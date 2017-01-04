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
    res << "neuron " << (size_t)this << ' ' << bias << ' ' << in.size() << ' ';
    for (auto d : in)
      res << (size_t)d.first << ' ' << d.second << ' ';
    return res.str();
  }

  void Neuron::load(float bias, std::vector<std::pair<Neuron*, float>> deps) {
    // std::cerr << "Loading to neuron " << (size_t)this << ' ' << bias << ' ';
    // for (auto d : deps)
    //   std::cerr << (size_t)d.first << ' ' << d.second << ' ';
    // std::cerr << std::endl;

    this->bias = bias;

    this->in.clear();
    for (auto d : deps) {
      // std::cerr << "Setting weight for " << (size_t)d.first << "->" << d.second << ": ";
      this->in[d.first] = d.second;
      // std::cerr << this->in[d.first] << std::endl;
    }
  }
}
