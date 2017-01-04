#include <algorithm>
#include <sstream>

#include "layer.hpp"

namespace ANNette {

  Layer::Layer(size_t cnt) {
    neurons.resize(cnt);
    for (size_t i = 0; i < cnt; ++i) {
      neurons[i] = new Neuron();
    }
  }

  Layer::Layer(std::vector<Neuron*> d) {
    neurons = d;
  }

  void Layer::calculate() {
    for (size_t i = 0; i < neurons.size(); ++i) {
      neurons[i]->calculate();
    }
  }

  void Layer::dependOn(Layer *target) {
    if (target == nullptr) return;

    for (size_t i = 0; i < neurons.size(); ++i) {
      for (size_t j = 0; j < target->size(); ++j) {
        neurons[i]->dependOn(target->getNeuron(j));
      }
    }
  }

  void Layer::setValues(const std::vector<float> &v) {
    for (size_t i = 0; i < std::min(v.size(), neurons.size()); i++) {
      neurons[i]->setValue(v[i]);
    }
  }

  size_t Layer::size() const {
    return neurons.size();
  }

  Neuron* Layer::getNeuron(size_t i) const {
    if (i >= neurons.size()) return nullptr;
    return neurons[i];
  }

  std::string Layer::dump() const {
    std::stringstream res;
    res << "layer " << (size_t)this << ' ' << neurons.size();
    for (auto n : neurons)
      res << std::endl << n->dump();
    return res.str();
  }

}
